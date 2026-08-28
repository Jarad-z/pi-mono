import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getModel } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createAgentSession } from "../src/core/sdk.ts";
import {
	type ManagedSessionEntryAppendRequest,
	type ManagedSessionEntryReservationRequest,
	type ManagedSessionLeafCasRequest,
	type ManagedSessionStore,
	type ManagedSessionStoreSnapshot,
	SessionManager,
} from "../src/core/session-manager.ts";

class EmptyManagedStore implements ManagedSessionStore {
	readonly load = vi.fn(
		async (): Promise<ManagedSessionStoreSnapshot> => ({
			header: {
				type: "session",
				version: 3,
				id: "managed_sdk_session",
				timestamp: "2026-08-29T00:00:00.000Z",
				cwd: "D:/workspace",
			},
			entries: [],
			leafId: null,
		}),
	);
	readonly reserve = vi.fn(async (_request: ManagedSessionEntryReservationRequest) => ({ entryId: "unused" }));
	readonly append = vi.fn(async (_request: ManagedSessionEntryAppendRequest) => {
		throw new Error("Unexpected managed append during SDK hydrate");
	});
	readonly setLeaf = vi.fn(async (_request: ManagedSessionLeafCasRequest) => {});

	loadSnapshot(): Promise<ManagedSessionStoreSnapshot> {
		return this.load();
	}

	reserveEntry(request: ManagedSessionEntryReservationRequest): Promise<{ entryId: string }> {
		return this.reserve(request);
	}

	appendEntry(request: ManagedSessionEntryAppendRequest) {
		return this.append(request);
	}

	compareAndSetLeaf(request: ManagedSessionLeafCasRequest): Promise<void> {
		return this.setLeaf(request);
	}
}

describe("createAgentSession managed host composition", () => {
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-sdk-managed-host-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(tempDir, { recursive: true, force: true });
	});

	it("installs every managed bridge without writing default metadata during hydrate", async () => {
		const store = new EmptyManagedStore();
		const sessionManager = await SessionManager.managed(store);
		const queueBarrier = vi.fn(async () => []);
		const lifecycleSink = vi.fn(async () => {});
		const failStopSink = vi.fn(async () => {});
		const model = getModel("anthropic", "claude-sonnet-4-5");
		expect(model).toBeTruthy();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir: tempDir,
			model: model!,
			sessionManager,
			managedHost: {
				managedLifecycleSink: lifecycleSink,
				managedQueueMaterializationHook: queueBarrier,
				managedExtensionHost: {
					getActivityBinding: () => undefined,
					dispatchExtensionAction: async (_request, execute) => execute(),
					executeTool: async (_scope, execute) => execute(),
				},
				managedFailStopSink: failStopSink,
			},
		});

		expect(session.sessionManager).toBe(sessionManager);
		expect(session.agent.managedQueueMaterializationHook).toBe(queueBarrier);
		expect(session.agent.managedFailStopHandler).toBeTypeOf("function");
		expect(store.reserve).not.toHaveBeenCalled();
		expect(store.append).not.toHaveBeenCalled();
		expect(store.setLeaf).not.toHaveBeenCalled();
		session.dispose();
	});

	it("rejects a managed store when the complete host bridge is absent", async () => {
		const sessionManager = await SessionManager.managed(new EmptyManagedStore());
		const model = getModel("anthropic", "claude-sonnet-4-5");

		await expect(
			createAgentSession({
				cwd: tempDir,
				agentDir: tempDir,
				model: model!,
				sessionManager,
			}),
		).rejects.toThrow("SessionManager.managed() and managedHost");
	});

	it("rejects a managed host bridge paired with a legacy SessionManager", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5");

		await expect(
			createAgentSession({
				cwd: tempDir,
				agentDir: tempDir,
				model: model!,
				sessionManager: SessionManager.inMemory(tempDir),
				managedHost: {
					managedLifecycleSink: async () => {},
					managedQueueMaterializationHook: async () => [],
					managedExtensionHost: {
						getActivityBinding: () => undefined,
						dispatchExtensionAction: async (_request, execute) => execute(),
						executeTool: async (_scope, execute) => execute(),
					},
					managedFailStopSink: async () => {},
				},
			}),
		).rejects.toThrow("SessionManager.managed() and managedHost");
	});
});
