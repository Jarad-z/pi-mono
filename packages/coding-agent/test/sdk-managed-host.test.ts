import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ManagedProviderAttemptGateway } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, createAssistantMessageEventStream, getModel } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import {
	type ManagedSessionEntryAppendRequest,
	type ManagedSessionEntryReservationRequest,
	type ManagedSessionLeafCasRequest,
	type ManagedSessionStore,
	type ManagedSessionStoreSnapshot,
	type SessionEntry,
	SessionManager,
} from "../src/core/session-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

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

class RecordingManagedStore implements ManagedSessionStore {
	readonly entries: SessionEntry[] = [];
	readonly reserved = new Set<string>();
	readonly order: string[];
	leafId: string | null = null;
	private nextId = 1;

	constructor(order: string[]) {
		this.order = order;
	}

	async loadSnapshot(): Promise<ManagedSessionStoreSnapshot> {
		return {
			header: {
				type: "session",
				version: 3,
				id: "managed_sdk_session",
				timestamp: "2026-08-29T00:00:00.000Z",
				cwd: "D:/workspace",
			},
			entries: this.entries,
			leafId: this.leafId,
		};
	}

	async reserveEntry(_request: ManagedSessionEntryReservationRequest): Promise<{ entryId: string }> {
		const entryId = `managed_sdk_entry_${this.nextId++}`;
		this.reserved.add(entryId);
		return { entryId };
	}

	async appendEntry(request: ManagedSessionEntryAppendRequest): Promise<SessionEntry> {
		if (request.expectedLeafId !== this.leafId || !this.reserved.delete(request.entry.id)) {
			throw new Error("Managed SDK store append CAS failed");
		}
		this.entries.push(request.entry);
		this.leafId = request.entry.id;
		this.order.push(`store:${request.entry.type === "message" ? request.entry.message.role : request.entry.type}`);
		return request.entry;
	}

	async compareAndSetLeaf(request: ManagedSessionLeafCasRequest): Promise<void> {
		if (request.expectedLeafId !== this.leafId) throw new Error("Managed SDK store leaf CAS failed");
		this.leafId = request.nextLeafId;
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
		const providerAttemptGateway: ManagedProviderAttemptGateway = {
			dispatch: async (request, execute) => ({
				receipt: {
					attemptId: `sdk-attempt-${request.requestId}`,
					attemptVersion: 1,
					purpose: request.purpose,
				},
				stream: await execute(),
			}),
			settle: async () => {},
		};
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
				managedProviderAttemptGateway: providerAttemptGateway,
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
		expect(session.agent.managedProviderAttemptGateway).toBe(providerAttemptGateway);
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

	it("drives one fake-provider Prompt through the complete managed SDK host", async () => {
		const order: string[] = [];
		const store = new RecordingManagedStore(order);
		store.reserved.add("managed_sdk_initial_entry");
		const sessionManager = await SessionManager.managed(store);
		const model = getModel("anthropic", "claude-sonnet-4-5");
		expect(model).toBeTruthy();
		if (!model) throw new Error("Managed SDK test model is unavailable");

		const authStorage = AuthStorage.inMemory({
			[model.provider]: { type: "api_key", key: "managed-sdk-key" },
		});
		const modelRegistry = await createInMemoryModelRegistry(authStorage);
		const providerAttemptGateway: ManagedProviderAttemptGateway = {
			dispatch: async (request, execute) => {
				order.push("provider-attempt:dispatched");
				return {
					receipt: {
						attemptId: `sdk-attempt-${request.requestId}`,
						attemptVersion: 1,
						purpose: request.purpose,
					},
					stream: await execute(),
				};
			},
			settle: async (receipt, result) => {
				order.push(`provider-attempt:settled:${receipt.attemptId}:${result.responseEntryId}`);
			},
		};
		modelRegistry.registerProvider(model.provider, {
			api: model.api,
			streamSimple: () => {
				order.push("provider");
				const stream = createAssistantMessageEventStream();
				const message: AssistantMessage = {
					role: "assistant",
					content: [{ type: "text", text: "done" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
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
				stream.end(message);
				return stream;
			},
		});

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir: tempDir,
			model,
			modelRuntime: getModelRuntime(modelRegistry),
			noTools: "all",
			sessionManager,
			managedHost: {
				managedLifecycleSink: async (event) => {
					const eventType = event.type === "agent_event" ? event.event.type : event.type;
					const role =
						event.type === "agent_event" && "message" in event.event ? event.event.message.role : undefined;
					order.push(`lifecycle:${eventType}${role ? `:${role}` : ""}`);
				},
				managedQueueMaterializationHook: async () => [],
				managedProviderAttemptGateway: providerAttemptGateway,
				managedExtensionHost: {
					getActivityBinding: () => ({
						generationId: "generation_sdk",
						generationLeaseToken: "lease_sdk",
						activityToken: "activity_sdk",
						runId: "run_sdk",
						coreInvocationId: "core_sdk",
					}),
					dispatchExtensionAction: async (_request, execute) => execute(),
					executeTool: async (_scope, execute) => execute(),
				},
				managedFailStopSink: async () => {},
			},
		});

		try {
			await expect(
				session.prepareManagedPrompt("activity_sdk", "hello", {
					reservedEntryId: "managed_sdk_initial_entry",
				}),
			).resolves.toEqual({ outcome: "ready", activityToken: "activity_sdk" });
			await session.launchManagedPrompt("activity_sdk");

			expect(store.entries.map((entry) => entry.id)).toEqual(["managed_sdk_initial_entry", "managed_sdk_entry_1"]);
			expect(order).toContain("provider");
			expect(order.indexOf("lifecycle:agent_start")).toBeLessThan(order.indexOf("provider"));
			expect(order.indexOf("lifecycle:message_end:user")).toBeLessThan(order.indexOf("store:user"));
			expect(order.indexOf("lifecycle:message_end:assistant")).toBeLessThan(order.indexOf("store:assistant"));
			expect(order.indexOf("store:assistant")).toBeLessThan(
				order.findIndex((item) => item.startsWith("provider-attempt:settled:sdk-attempt-1:managed_sdk_entry_1")),
			);
			expect(order.at(-1)).toBe("lifecycle:agent_settled");
		} finally {
			session.dispose();
			modelRegistry.unregisterProvider(model.provider);
		}
	});

	it("commits a host-reserved managed compaction before completing its ProviderAttempt", async () => {
		const order: string[] = [];
		const store = new RecordingManagedStore(order);
		const message = (
			id: string,
			parentId: string | null,
			role: "user" | "assistant",
			text: string,
		): SessionEntry => ({
			type: "message",
			id,
			parentId,
			timestamp: "2026-08-29T00:00:00.000Z",
			message:
				role === "user"
					? { role, content: [{ type: "text", text }], timestamp: 1 }
					: {
							role,
							content: [{ type: "text", text }],
							api: "anthropic-messages",
							provider: "anthropic",
							model: "claude-sonnet-4-5",
							usage: {
								input: 0,
								output: 0,
								cacheRead: 0,
								cacheWrite: 0,
								totalTokens: 0,
								cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
							},
							stopReason: "stop",
							timestamp: 1,
						},
		});
		store.entries.push(
			message("entry-1", null, "user", "old request ".repeat(10_000)),
			message("entry-2", "entry-1", "assistant", "old result"),
			message("entry-3", "entry-2", "user", "middle request ".repeat(10_000)),
			message("entry-4", "entry-3", "assistant", "middle result"),
			message("entry-5", "entry-4", "user", "recent request"),
			message("entry-6", "entry-5", "assistant", "recent result"),
		);
		store.leafId = "entry-6";
		store.reserved.add("entry-managed-compaction");
		const sessionManager = await SessionManager.managed(store);
		const model = getModel("anthropic", "claude-sonnet-4-5");
		if (!model) throw new Error("Managed SDK test model is unavailable");
		const authStorage = AuthStorage.inMemory({
			[model.provider]: { type: "api_key", key: "managed-sdk-key" },
		});
		const modelRegistry = await createInMemoryModelRegistry(authStorage);
		modelRegistry.registerProvider(model.provider, {
			api: model.api,
			streamSimple: () => {
				order.push("provider");
				const stream = createAssistantMessageEventStream();
				stream.end({
					role: "assistant",
					content: [{ type: "text", text: "managed summary" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: {
						input: 10,
						output: 5,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 15,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "stop",
					timestamp: Date.now(),
				});
				return stream;
			},
		});
		const providerAttemptGateway: ManagedProviderAttemptGateway = {
			dispatch: async (request, execute) => {
				order.push(`provider-attempt:dispatched:${request.requestId}:${request.purpose}`);
				return {
					receipt: {
						attemptId: `sdk-maintenance-${request.requestId}`,
						attemptVersion: 1,
						purpose: request.purpose,
					},
					stream: await execute(),
				};
			},
			settle: async (receipt, settlement) => {
				order.push(
					`provider-attempt:settled:${receipt.attemptId}:${settlement.status}:${settlement.responseEntryId ?? "none"}`,
				);
			},
		};
		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir: tempDir,
			model,
			modelRuntime: getModelRuntime(modelRegistry),
			noTools: "all",
			sessionManager,
			managedHost: {
				managedLifecycleSink: async () => {},
				managedQueueMaterializationHook: async () => [],
				managedProviderAttemptGateway: providerAttemptGateway,
				managedExtensionHost: {
					getActivityBinding: () => ({
						generationId: "generation_sdk",
						generationLeaseToken: "lease_sdk",
						activityToken: "maintenance_sdk",
						maintenanceActivityId: "maintenance-activity-sdk",
						maintenanceCoreId: "maintenance-core-sdk",
					}),
					dispatchExtensionAction: async (_request, execute) => execute(),
					executeTool: async (_scope, execute) => execute(),
				},
				managedFailStopSink: async () => {},
			},
		});

		try {
			await expect(session.compact()).rejects.toThrow("requires compactManaged()");
			const result = await session.compactManaged({
				activityToken: "maintenance_sdk",
				reservedEntryId: "entry-managed-compaction",
			});
			expect(result.summary).toContain("managed summary");
			expect(store.entries.at(-1)).toMatchObject({
				type: "compaction",
				id: "entry-managed-compaction",
				parentId: "entry-6",
			});
			expect(order).toContain("provider-attempt:dispatched:maintenance_sdk:summary:1:manual_compaction");
			expect(order.indexOf("store:compaction")).toBeLessThan(
				order.indexOf(
					"provider-attempt:settled:sdk-maintenance-maintenance_sdk:summary:1:completed:entry-managed-compaction",
				),
			);
		} finally {
			session.dispose();
			modelRegistry.unregisterProvider(model.provider);
		}
	});
});
