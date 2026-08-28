import { type AssistantMessage, type AssistantMessageEvent, EventStream } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { Agent, type AgentMessage, type ManagedQueueMaterializationHook, type StreamFn } from "../src/index.ts";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor(message: AssistantMessage) {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
		queueMicrotask(() => this.push({ type: "done", reason: "stop", message }));
	}
}

function assistant(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function user(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

function userTexts(messages: readonly AgentMessage[]): string[] {
	return messages.flatMap((message) => {
		if (message.role !== "user") return [];
		if (typeof message.content === "string") return [message.content];
		return message.content.flatMap((part) => (part.type === "text" ? [part.text] : []));
	});
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve = (): void => {};
	const promise = new Promise<void>((settle) => {
		resolve = settle;
	});
	return { promise, resolve };
}

function createAgent(
	hook: ManagedQueueMaterializationHook,
	overrides: { steeringMode?: "all" | "one-at-a-time"; streamFn?: StreamFn } = {},
): Agent {
	return new Agent({
		managedQueueMaterializationHook: hook,
		steeringMode: overrides.steeringMode,
		initialState: { messages: [user("initial"), assistant("initial response")] },
		streamFn: overrides.streamFn ?? (() => new MockAssistantStream(assistant("done"))),
	});
}

describe("Agent managed queue mirror", () => {
	it("keeps staged and admitted items non-drainable until fenced publish", async () => {
		const selectedBatches: string[][] = [];
		const providerContexts: AgentMessage[][] = [];
		const agent = createAgent(
			async (items) => {
				selectedBatches.push(items.map((item) => item.itemId));
				return items.map((item) => ({ type: "materialized", itemId: item.itemId, message: user("final") }));
			},
			{
				streamFn: (_model, context) => {
					providerContexts.push(context.messages as AgentMessage[]);
					return new MockAssistantStream(assistant("done"));
				},
			},
		);

		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("raw") });
		expect(staged.type).toBe("staged");
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		expect(agent.hasQueuedMessages()).toBe(false);
		await expect(agent.continue()).rejects.toThrow("Cannot continue from message role: assistant");

		expect(agent.admitManagedQueueItem(staged.ticket)).toBe("admitted");
		expect(agent.hasQueuedMessages()).toBe(false);
		await expect(agent.continue()).rejects.toThrow("Cannot continue from message role: assistant");

		expect(agent.publishManagedQueueItem(staged.ticket)).toBe("published");
		expect(agent.hasQueuedMessages()).toBe(true);
		await agent.continue();

		expect(selectedBatches).toEqual([["input_1"]]);
		expect(providerContexts).toHaveLength(1);
		expect(userTexts(providerContexts[0])).toContain("final");
		expect(userTexts(providerContexts[0])).not.toContain("raw");
		expect(agent.removeManagedQueueItem("input_1")).toBe("already_consumed");
	});

	it("preserves FIFO by refusing to drain a published item behind a provisional prefix", async () => {
		const batches: string[][] = [];
		const agent = createAgent(
			async (items) => {
				batches.push(items.map((item) => item.itemId));
				return items.map((item) => ({ type: "materialized", itemId: item.itemId, message: item.message }));
			},
			{ steeringMode: "all" },
		);
		const first = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("first") });
		const second = agent.stageManagedQueueItem({ itemId: "input_2", lane: "steer", message: user("second") });
		if (first.type !== "staged" || second.type !== "staged") throw new Error("Expected staged tickets");
		agent.admitManagedQueueItem(second.ticket);
		agent.publishManagedQueueItem(second.ticket);

		expect(agent.hasQueuedMessages()).toBe(false);
		agent.admitManagedQueueItem(first.ticket);
		agent.publishManagedQueueItem(first.ticket);
		await agent.continue();

		expect(batches).toEqual([["input_1", "input_2"]]);
	});

	it("linearizes selection before cancellation and accepts a host drop_cancelled result", async () => {
		const selected = deferred();
		const release = deferred();
		let providerCalls = 0;
		const agent = createAgent(
			async (items) => {
				selected.resolve();
				await release.promise;
				return items.map((item) => ({ type: "drop_cancelled", itemId: item.itemId }));
			},
			{
				streamFn: () => {
					providerCalls++;
					return new MockAssistantStream(assistant("unexpected"));
				},
			},
		);
		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("raw") });
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		agent.admitManagedQueueItem(staged.ticket);
		agent.publishManagedQueueItem(staged.ticket);

		const continuation = agent.continue();
		await selected.promise;
		expect(agent.getManagedQueueMirrorSnapshot()).toEqual([
			{ itemId: "input_1", lane: "steer", mirrorRevision: staged.ticket.mirrorRevision, phase: "selected" },
		]);
		expect(agent.removeManagedQueueItem("input_1")).toBe("already_selected");
		expect(providerCalls).toBe(0);

		release.resolve();
		await continuation;
		expect(providerCalls).toBe(0);
		expect(agent.getManagedQueueMirrorSnapshot()).toEqual([]);
		expect(agent.removeManagedQueueItem("input_1")).toBe("not_found");
	});

	it("holds Provider visibility until the awaited hook returns transformed material", async () => {
		const selected = deferred();
		const release = deferred();
		const contexts: AgentMessage[][] = [];
		let agent!: Agent;
		agent = createAgent(
			async (items, signal) => {
				expect(signal).toBe(agent.signal);
				selected.resolve();
				await release.promise;
				return items.map((item) => ({ type: "materialized", itemId: item.itemId, message: user("transformed") }));
			},
			{
				streamFn: (_model, context) => {
					contexts.push(context.messages as AgentMessage[]);
					return new MockAssistantStream(assistant("done"));
				},
			},
		);
		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "follow_up", message: user("raw") });
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		agent.admitManagedQueueItem(staged.ticket);
		agent.publishManagedQueueItem(staged.ticket);

		const continuation = agent.continue();
		await selected.promise;
		expect(contexts).toEqual([]);
		release.resolve();
		await continuation;

		expect(contexts).toHaveLength(1);
		expect(userTexts(contexts[0])).toContain("transformed");
		expect(userTexts(contexts[0])).not.toContain("raw");
	});

	it("fails the mirror closed when materialization rejects", async () => {
		let providerCalls = 0;
		const agent = createAgent(
			async () => {
				throw new Error("durable materialization failed");
			},
			{
				streamFn: () => {
					providerCalls++;
					return new MockAssistantStream(assistant("unexpected"));
				},
			},
		);
		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("raw") });
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		agent.admitManagedQueueItem(staged.ticket);
		agent.publishManagedQueueItem(staged.ticket);

		await agent.continue();

		expect(providerCalls).toBe(0);
		expect(agent.hasManagedQueueFailure()).toBe(true);
		expect(agent.getManagedQueueMirrorSnapshot()[0]?.phase).toBe("failed");
		expect(agent.state.errorMessage).toBe("durable materialization failed");
		expect(() => agent.stageManagedQueueItem({ itemId: "input_2", lane: "steer", message: user("new") })).toThrow(
			"Managed queue mirror is failed",
		);
	});

	it("rejects legacy queue mutation and fences stale admission tickets", () => {
		const agent = createAgent(async (items) =>
			items.map((item) => ({ type: "materialized", itemId: item.itemId, message: item.message })),
		);
		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("raw") });
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		const stale = { ...staged.ticket, mirrorRevision: staged.ticket.mirrorRevision + 1 };

		expect(agent.admitManagedQueueItem(stale)).toBe("stale");
		expect(agent.publishManagedQueueItem(stale)).toBe("stale");
		expect(agent.abortManagedQueueItem(stale)).toBe("stale");
		expect(() => agent.steer(user("legacy"))).toThrow("stable stage/admit/publish");
		expect(() => agent.followUp(user("legacy"))).toThrow("stable stage/admit/publish");
		expect(() => agent.clearAllQueues()).toThrow("stable stage/admit/publish");
	});
});
