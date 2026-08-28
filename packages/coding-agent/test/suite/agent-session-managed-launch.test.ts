import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import type {
	ManagedAgentSessionLifecycleEvent,
	ManagedAgentSessionLifecycleSink,
} from "../../src/core/agent-session.ts";
import { createHarness, type Harness } from "./harness.ts";

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve = (): void => {};
	const promise = new Promise<void>((settle) => {
		resolve = settle;
	});
	return { promise, resolve };
}

describe("AgentSession managed paused launch", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("pauses after preflight and correlates every awaited lifecycle event", async () => {
		const lifecycle: ManagedAgentSessionLifecycleEvent[] = [];
		const sink: ManagedAgentSessionLifecycleSink = async (event) => {
			lifecycle.push(event);
		};
		const harness = await createHarness({ managedLifecycleSink: sink });
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("done")]);

		await expect(harness.session.prepareManagedPrompt("activity_1", "hello")).resolves.toEqual({
			outcome: "ready",
			activityToken: "activity_1",
		});
		expect(harness.session.hasPreparedManagedPrompt).toBe(true);
		expect(harness.session.messages).toEqual([]);
		expect(harness.getPendingResponseCount()).toBe(1);
		expect(lifecycle).toEqual([]);

		await harness.session.launchManagedPrompt("activity_1");

		expect(harness.session.hasPreparedManagedPrompt).toBe(false);
		expect(lifecycle[0]).toMatchObject({
			type: "agent_event",
			activityToken: "activity_1",
			event: { type: "agent_start" },
		});
		expect(lifecycle.at(-1)).toEqual({ type: "agent_settled", activityToken: "activity_1" });
		expect(lifecycle.every((event) => event.activityToken === "activity_1")).toBe(true);
		expect(harness.session.messages.map((message) => message.role)).toEqual(["user", "assistant"]);
	});

	it("awaits the correlated agent_start sink before the provider can observe the prompt", async () => {
		const startSeen = deferred();
		const releaseStart = deferred();
		const sink: ManagedAgentSessionLifecycleSink = async (event) => {
			if (event.type === "agent_event" && event.event.type === "agent_start") {
				startSeen.resolve();
				await releaseStart.promise;
			}
		};
		const harness = await createHarness({ managedLifecycleSink: sink });
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("done")]);
		await harness.session.prepareManagedPrompt("activity_2", "hello");

		const launch = harness.session.launchManagedPrompt("activity_2");
		await startSeen.promise;

		expect(harness.getPendingResponseCount()).toBe(1);
		expect(harness.session.messages).toEqual([]);
		releaseStart.resolve();
		await launch;
		expect(harness.getPendingResponseCount()).toBe(0);
	});

	it("keeps launch and waitForIdle pending until the correlated settlement sink drains", async () => {
		const settlementSeen = deferred();
		const releaseSettlement = deferred();
		const sink: ManagedAgentSessionLifecycleSink = async (event) => {
			if (event.type === "agent_settled") {
				settlementSeen.resolve();
				await releaseSettlement.promise;
			}
		};
		const harness = await createHarness({ managedLifecycleSink: sink });
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("done")]);
		await harness.session.prepareManagedPrompt("activity_3", "hello");

		let launchFinished = false;
		const launch = harness.session.launchManagedPrompt("activity_3").then(() => {
			launchFinished = true;
		});
		await settlementSeen.promise;
		let idleFinished = false;
		const idle = harness.session.waitForIdle().then(() => {
			idleFinished = true;
		});
		await Promise.resolve();

		expect(launchFinished).toBe(false);
		expect(idleFinished).toBe(false);
		expect(harness.events.some((event) => event.type === "agent_settled")).toBe(false);
		releaseSettlement.resolve();
		await Promise.all([launch, idle]);
		expect(harness.events.at(-1)?.type).toBe("agent_settled");
	});

	it("cancels only the matching prepared token and never permits token reuse", async () => {
		const harness = await createHarness({ managedLifecycleSink: async () => {} });
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("unused")]);
		await harness.session.prepareManagedPrompt("activity_4", "hello");

		expect(harness.session.cancelManagedPrompt("different")).toBe(false);
		expect(harness.session.cancelManagedPrompt("activity_4")).toBe(true);
		expect(harness.session.cancelManagedPrompt("activity_4")).toBe(false);
		await expect(harness.session.launchManagedPrompt("activity_4")).rejects.toThrow("No prepared managed prompt");
		await expect(harness.session.prepareManagedPrompt("activity_4", "again")).rejects.toThrow("was already used");
		expect(harness.getPendingResponseCount()).toBe(1);
	});

	it("returns handled without creating an activity when an extension command consumes preflight", async () => {
		const commandRuns: string[] = [];
		const lifecycle: ManagedAgentSessionLifecycleEvent[] = [];
		const harness = await createHarness({
			managedLifecycleSink: async (event) => {
				lifecycle.push(event);
			},
			extensionFactories: [
				(pi) => {
					pi.registerCommand("managed", {
						description: "managed command",
						handler: async (args) => {
							commandRuns.push(args);
						},
					});
				},
			],
		});
		harnesses.push(harness);

		await expect(harness.session.prepareManagedPrompt("activity_5", "/managed value")).resolves.toEqual({
			outcome: "handled",
			activityToken: "activity_5",
		});
		expect(commandRuns).toEqual(["value"]);
		expect(lifecycle).toEqual([]);
		expect(harness.session.hasPreparedManagedPrompt).toBe(false);
	});

	it("rejects legacy prompt entry while the managed lifecycle sink owns the session", async () => {
		const harness = await createHarness({ managedLifecycleSink: async () => {} });
		harnesses.push(harness);

		await expect(harness.session.prompt("legacy")).rejects.toThrow(
			"Managed lifecycle mode requires prepareManagedPrompt() followed by launchManagedPrompt()",
		);
	});

	it("bridges stable managed queue admission and rejects legacy queue mutation", async () => {
		const harness = await createHarness({
			managedLifecycleSink: async () => {},
			managedQueueMaterializationHook: async (items) =>
				items.map((item) => ({ type: "materialized", itemId: item.itemId, message: item.message })),
		});
		harnesses.push(harness);
		const staged = harness.session.stageManagedQueueItem({
			itemId: "input_1",
			lane: "steer",
			message: { role: "user", content: [{ type: "text", text: "queued" }], timestamp: Date.now() },
		});
		expect(staged.type).toBe("staged");
		if (staged.type !== "staged") throw new Error("Expected staged ticket");

		expect(harness.session.admitManagedQueueItem(staged.ticket)).toBe("admitted");
		expect(harness.session.publishManagedQueueItem(staged.ticket)).toBe("published");
		expect(harness.session.getManagedQueueMirrorSnapshot()).toEqual([
			{
				itemId: "input_1",
				lane: "steer",
				mirrorRevision: staged.ticket.mirrorRevision,
				phase: "published",
			},
		]);
		await expect(harness.session.steer("legacy")).rejects.toThrow("stageManagedQueueItem() admission");
		await expect(harness.session.followUp("legacy")).rejects.toThrow("stageManagedQueueItem() admission");
		expect(() => harness.session.clearQueue()).toThrow("per-item removeManagedQueueItem() cancellation");
		await expect(harness.session.prepareManagedPrompt("activity_6", "new activity")).rejects.toThrow(
			"cannot start while queued messages remain",
		);
		expect(harness.session.removeManagedQueueItem("input_1")).toBe("removed");
		expect(harness.session.getManagedQueueMirrorSnapshot()).toEqual([]);
	});
});
