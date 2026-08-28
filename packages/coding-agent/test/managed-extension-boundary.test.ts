import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createEventBus } from "../src/core/event-bus.ts";
import { createExtensionRuntime, loadExtensionFromFactory } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import type {
	ExtensionActions,
	ExtensionContextActions,
	ManagedExtensionActionRequest,
	ManagedExtensionHost,
	ManagedHostActivityBinding,
	ManagedToolActivityScope,
} from "../src/core/extensions/types.ts";
import { wrapRegisteredTool } from "../src/core/extensions/wrapper.ts";
import type { ModelRegistry } from "../src/core/model-registry.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { createInMemoryModelRegistry } from "./model-runtime-test-utils.ts";

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve = (): void => {};
	const promise = new Promise<void>((settle) => {
		resolve = settle;
	});
	return { promise, resolve };
}

describe("managed extension boundary", () => {
	let sessionManager: SessionManager;
	let modelRegistry: ModelRegistry;
	let binding: ManagedHostActivityBinding;
	let actionRequests: ManagedExtensionActionRequest[];
	let toolScopes: ManagedToolActivityScope[];
	let managedHost: ManagedExtensionHost;
	let extensionActions: ExtensionActions;
	const localMutations: string[] = [];

	const contextActions: ExtensionContextActions = {
		getModel: () => undefined,
		getScopedModels: () => [],
		isIdle: () => false,
		isProjectTrusted: () => true,
		getSignal: () => undefined,
		abort: () => {
			localMutations.push("abort");
		},
		hasPendingMessages: () => false,
		shutdown: () => {
			localMutations.push("shutdown");
		},
		getContextUsage: () => undefined,
		compact: () => {
			localMutations.push("compact");
		},
		getSystemPrompt: () => "",
	};

	beforeEach(async () => {
		sessionManager = SessionManager.inMemory();
		modelRegistry = await createInMemoryModelRegistry(AuthStorage.inMemory());
		binding = {
			generationId: "generation_1",
			generationLeaseToken: "lease_1",
			activityToken: "activity_1",
			runId: "run_1",
			coreInvocationId: "core_1",
		};
		actionRequests = [];
		toolScopes = [];
		localMutations.length = 0;
		managedHost = {
			getActivityBinding: () => binding,
			dispatchExtensionAction: async (request, execute) => {
				actionRequests.push(request);
				return execute();
			},
			executeTool: async (scope, execute) => {
				toolScopes.push(scope);
				return execute();
			},
		};
		extensionActions = {
			sendMessage: () => {
				localMutations.push("send_message");
			},
			sendUserMessage: () => {
				localMutations.push("send_user_message");
			},
			appendEntry: (customType) => {
				localMutations.push(`append_entry:${customType}`);
			},
			setSessionName: (name) => {
				localMutations.push(`set_session_name:${name}`);
			},
			getSessionName: () => undefined,
			setLabel: () => {
				localMutations.push("set_label");
			},
			exec: async () => ({ stdout: "", stderr: "", code: 0, killed: false }),
			getActiveTools: () => [],
			getAllTools: () => [],
			setActiveTools: () => {
				localMutations.push("set_active_tools");
			},
			refreshTools: () => {},
			getCommands: () => [],
			setModel: async () => false,
			getThinkingLevel: () => "off",
			setThinkingLevel: () => {
				localMutations.push("set_thinking_level");
			},
		};
	});

	async function createRunner(
		factory: Parameters<typeof loadExtensionFromFactory>[0],
		extensionPath = "<inline:managed>",
	): Promise<ExtensionRunner> {
		const runtime = createExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			factory,
			"D:/workspace",
			createEventBus(),
			runtime,
			extensionPath,
		);
		const runner = new ExtensionRunner(
			[extension],
			runtime,
			"D:/workspace",
			sessionManager,
			modelRegistry,
			managedHost,
		);
		runner.bindCore(extensionActions, contextActions);
		return runner;
	}

	it("drains fire-and-forget admissions before an event scope returns", async () => {
		const release = deferred();
		managedHost.dispatchExtensionAction = async (request, execute) => {
			actionRequests.push(request);
			await release.promise;
			return execute();
		};
		const runner = await createRunner((pi) => {
			pi.on("agent_end", () => {
				pi.appendEntry("audit", { ok: true });
				pi.setSessionName("managed");
			});
		});

		let settled = false;
		const emitted = runner.emit({ type: "agent_end", messages: [] }).then(() => {
			settled = true;
		});
		await Promise.resolve();

		expect(actionRequests.map((request) => request.action.type)).toEqual(["append_entry", "set_session_name"]);
		expect(localMutations).toEqual([]);
		expect(settled).toBe(false);
		expect(actionRequests[0].scope).toMatchObject({
			extensionPath: "<inline:managed>",
			eventType: "agent_end",
			handlerIndex: 0,
			actionIndex: 1,
			binding,
		});
		expect(actionRequests[1].scope.scopeId).toBe(actionRequests[0].scope.scopeId);
		expect(actionRequests[1].scope.actionIndex).toBe(2);

		release.resolve();
		await emitted;
		expect(localMutations).toEqual(["append_entry:audit", "set_session_name:managed"]);
	});

	it("fails closed on gateway rejection and does not run later handlers", async () => {
		managedHost.dispatchExtensionAction = async () => {
			throw new Error("durable admission rejected");
		};
		const secondHandler = vi.fn();
		const runner = await createRunner((pi) => {
			pi.on("agent_end", () => pi.appendEntry("first"));
			pi.on("agent_end", secondHandler);
		});

		await expect(runner.emit({ type: "agent_end", messages: [] })).rejects.toThrow("durable admission rejected");
		expect(secondHandler).not.toHaveBeenCalled();
		expect(localMutations).toEqual([]);
	});

	it("fences an action when its activity binding changes inside the handler", async () => {
		let bindingReads = 0;
		managedHost.getActivityBinding = () =>
			++bindingReads === 1 ? binding : { ...binding, activityToken: "stale_activity" };
		const runner = await createRunner((pi) => {
			pi.on("agent_end", () => pi.appendEntry("blocked"));
		});

		await expect(runner.emit({ type: "agent_end", messages: [] })).rejects.toThrow("crossed an activity fence");
		expect(actionRequests).toEqual([]);
		expect(localMutations).toEqual([]);
	});

	it("rechecks the activity fence when the host executes an admitted action", async () => {
		const admitted = deferred();
		const release = deferred();
		managedHost.dispatchExtensionAction = async (request, execute) => {
			actionRequests.push(request);
			admitted.resolve();
			await release.promise;
			return execute();
		};
		const runner = await createRunner((pi) => {
			pi.on("agent_end", () => pi.appendEntry("blocked"));
		});

		const emitted = runner.emit({ type: "agent_end", messages: [] });
		await admitted.promise;
		binding = { ...binding, activityToken: "stale_activity" };
		release.resolve();

		await expect(emitted).rejects.toThrow("crossed an activity fence before execution");
		expect(localMutations).toEqual([]);
	});

	it("rejects captured mutation APIs after their scope closes", async () => {
		let lateMutation: (() => void) | undefined;
		const runner = await createRunner((pi) => {
			pi.on("agent_end", () => {
				lateMutation = () => pi.appendEntry("late");
			});
		});

		await runner.emit({ type: "agent_end", messages: [] });
		expect(() => lateMutation?.()).toThrow("outside an active ExtensionActionScope");
		expect(actionRequests).toEqual([]);
	});

	it("allows pure event reads without an activity but rejects mutations", async () => {
		managedHost.getActivityBinding = () => undefined;
		const pureHandler = vi.fn();
		const readRunner = await createRunner((pi) => {
			pi.on("session_start", (_event, context) => {
				context.sessionManager.getSessionId();
				context.modelRegistry.getAll();
				pureHandler();
			});
		});

		await readRunner.emit({ type: "session_start", reason: "startup" });
		expect(pureHandler).toHaveBeenCalledOnce();

		const mutationRunner = await createRunner((pi) => {
			pi.on("session_start", () => pi.appendEntry("blocked"));
		});
		await expect(mutationRunner.emit({ type: "session_start", reason: "startup" })).rejects.toThrow(
			"outside an active ExtensionActionScope",
		);
	});

	it("exposes object-level read-only session and model facades", async () => {
		const entryId = sessionManager.appendCustomEntry("state", { value: 1 });
		const runner = await createRunner((_pi) => {});
		const context = runner.createContext();

		expect(context.sessionManager.getSessionId()).toBe(sessionManager.getSessionId());
		expect("appendCustomEntry" in context.sessionManager).toBe(false);
		const entry = context.sessionManager.getEntry(entryId);
		if (entry?.type !== "custom") throw new Error("expected custom entry");
		(entry.data as { value: number }).value = 2;
		expect(sessionManager.getEntry(entryId)).toMatchObject({ data: { value: 1 } });

		expect(() => context.modelRegistry.unregisterProvider("blocked")).toThrow("not a pure catalogue read");
		const model = context.modelRegistry.getAll()[0];
		expect(model).toBeDefined();
		const originalName = model.name;
		model.name = "mutated outside host";
		expect(modelRegistry.find(model.provider, model.id)?.name).toBe(originalName);
	});

	it("resolves a fresh Run/CoreInvocation scope for every tool execution", async () => {
		const runner = await createRunner((pi) => {
			pi.registerTool({
				name: "scoped_tool",
				label: "Scoped tool",
				description: "Verifies managed scope",
				parameters: Type.Object({}),
				execute: async () => {
					pi.appendEntry("tool_audit");
					return { content: [{ type: "text", text: "ok" }], details: {} };
				},
			});
		});
		const registered = runner.getAllRegisteredTools()[0];
		const tool = wrapRegisteredTool(registered, runner);

		await tool.execute("call_1", {}, undefined, undefined);
		binding = { ...binding, runId: "run_2", coreInvocationId: "core_2", activityToken: "activity_2" };
		await tool.execute("call_2", {}, undefined, undefined);

		expect(toolScopes).toMatchObject([
			{ toolCallId: "call_1", activityToken: "activity_1", runId: "run_1", coreInvocationId: "core_1" },
			{ toolCallId: "call_2", activityToken: "activity_2", runId: "run_2", coreInvocationId: "core_2" },
		]);
		expect(actionRequests.map((request) => request.scope.binding.activityToken)).toEqual([
			"activity_1",
			"activity_2",
		]);
	});

	it("routes command-context mutations through the command scope", async () => {
		const reload = vi.fn(async () => {});
		const runner = await createRunner((pi) => {
			pi.registerCommand("reload-managed", {
				handler: async (_args, ctx) => ctx.reload(),
			});
		});
		runner.bindCommandContext({
			waitForIdle: async () => {},
			newSession: async () => ({ cancelled: false }),
			fork: async () => ({ cancelled: false }),
			navigateTree: async () => ({ cancelled: false }),
			switchSession: async () => ({ cancelled: false }),
			reload,
		});

		await runner.getCommand("reload-managed")?.handler("", runner.createCommandContext());

		expect(actionRequests).toHaveLength(1);
		expect(actionRequests[0]).toMatchObject({
			scope: { eventType: "command:reload-managed", extensionPath: "<inline:managed>" },
			action: { type: "reload" },
		});
		expect(reload).toHaveBeenCalledOnce();
	});
});
