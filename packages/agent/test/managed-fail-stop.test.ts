import { type AssistantMessage, type AssistantMessageEvent, EventStream } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { describe, expect, it, vi } from "vitest";
import { Agent, type AgentMessage, type AgentTool, ManagedAgentFailStopError, type StreamFn } from "../src/index.ts";

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
		queueMicrotask(() => this.push({ type: "done", reason: message.stopReason, message }));
	}
}

function assistant(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"] = "stop") {
	return {
		role: "assistant",
		content,
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
		stopReason,
		timestamp: Date.now(),
	} satisfies AssistantMessage;
}

function user(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

describe("managed Agent fail-stop", () => {
	it("fails the generation once when an awaited lifecycle listener rejects", async () => {
		const streamFn = vi.fn<StreamFn>(() => new MockAssistantStream(assistant([{ type: "text", text: "unused" }])));
		const failures: ManagedAgentFailStopError[] = [];
		const events: string[] = [];
		const agent = new Agent({
			streamFn,
			managedFailStopHandler: async (failure) => {
				failures.push(failure);
			},
		});
		agent.subscribe(async (event) => {
			events.push(event.type);
			if (event.type === "agent_start") throw new Error("durable lifecycle rejected");
		});

		await expect(agent.prompt("hello")).rejects.toMatchObject({
			name: "ManagedAgentFailStopError",
			phase: "lifecycle_listener",
		});
		expect(events).toEqual(["agent_start"]);
		expect(streamFn).not.toHaveBeenCalled();
		expect(failures).toHaveLength(1);

		await expect(agent.prompt("again")).rejects.toThrow("fail-stopped");
		expect(failures).toHaveLength(1);
	});

	it("fences before Provider visibility when managed queue materialization rejects", async () => {
		const streamFn = vi.fn<StreamFn>(() => new MockAssistantStream(assistant([{ type: "text", text: "unused" }])));
		const failures: ManagedAgentFailStopError[] = [];
		const agent = new Agent({
			initialState: { messages: [user("initial"), assistant([{ type: "text", text: "initial response" }])] },
			streamFn,
			managedQueueMaterializationHook: async () => {
				throw new Error("durable dequeue rejected");
			},
			managedFailStopHandler: async (failure) => {
				failures.push(failure);
			},
		});
		const staged = agent.stageManagedQueueItem({ itemId: "input_1", lane: "steer", message: user("raw") });
		expect(staged.type).toBe("staged");
		if (staged.type !== "staged") throw new Error("Expected staged ticket");
		expect(agent.admitManagedQueueItem(staged.ticket)).toBe("admitted");
		expect(agent.publishManagedQueueItem(staged.ticket)).toBe("published");

		await expect(agent.continue()).rejects.toMatchObject({
			name: "ManagedAgentFailStopError",
			phase: "queue_materialization",
		});
		expect(streamFn).not.toHaveBeenCalled();
		expect(failures).toHaveLength(1);
		expect(agent.getManagedQueueMirrorSnapshot()).toEqual([
			expect.objectContaining({ itemId: "input_1", phase: "failed" }),
		]);
		expect(() => agent.stageManagedQueueItem({ itemId: "input_2", lane: "steer", message: user("new") })).toThrow(
			"fail-stopped",
		);
	});

	it("does not convert a managed tool boundary failure into a tool result", async () => {
		const responses = [
			assistant([{ type: "toolCall", id: "call_1", name: "managed_tool", arguments: {} }], "toolUse"),
			assistant([{ type: "text", text: "must not run" }]),
		];
		const streamFn = vi.fn<StreamFn>(() => {
			const response = responses.shift();
			if (!response) throw new Error("Unexpected Provider call");
			return new MockAssistantStream(response);
		});
		const failures: ManagedAgentFailStopError[] = [];
		const events: string[] = [];
		const tool: AgentTool = {
			name: "managed_tool",
			label: "Managed tool",
			description: "Fails at the managed host gateway",
			parameters: Type.Object({}),
			execute: async () => {
				throw new ManagedAgentFailStopError("managed_host_boundary", "tool gateway rejected");
			},
		};
		const agent = new Agent({
			streamFn,
			initialState: { tools: [tool] },
			managedFailStopHandler: async (failure) => {
				failures.push(failure);
			},
		});
		agent.subscribe(async (event) => {
			events.push(event.type);
		});

		await expect(agent.prompt("run the tool")).rejects.toMatchObject({
			name: "ManagedAgentFailStopError",
			phase: "managed_host_boundary",
		});
		expect(streamFn).toHaveBeenCalledTimes(1);
		expect(failures).toHaveLength(1);
		expect(events).not.toContain("tool_execution_end");
		expect(events).not.toContain("agent_end");
	});

	it("remains locally fenced when the host fail-stop bridge rejects", async () => {
		const agent = new Agent({
			streamFn: () => new MockAssistantStream(assistant([{ type: "text", text: "unused" }])),
			managedFailStopHandler: async () => {
				throw new Error("durable generation fence failed");
			},
		});
		agent.subscribe(async (event) => {
			if (event.type === "agent_start") throw new Error("durable lifecycle rejected");
		});

		await expect(agent.prompt("hello")).rejects.toMatchObject({
			name: "ManagedAgentFailStopError",
			phase: "fail_stop_bridge",
		});
		await expect(agent.prompt("again")).rejects.toThrow("fail-stopped");
	});
});
