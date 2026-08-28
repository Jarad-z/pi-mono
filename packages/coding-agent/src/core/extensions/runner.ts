/**
 * Extension runner - executes extensions and manages their lifecycle.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { type AgentMessage, type AgentToolResult, ManagedAgentFailStopError } from "@earendil-works/pi-agent-core";
import type { Api, ImageContent, Model, Provider, ProviderHeaders } from "@earendil-works/pi-ai";
import type { KeyId } from "@earendil-works/pi-tui";
import { type Theme, theme } from "../../modes/interactive/theme/theme.ts";
import type { ResourceDiagnostic } from "../diagnostics.ts";
import type { KeybindingsConfig } from "../keybindings.ts";
import type { ModelRegistry } from "../model-registry.ts";
import type { ScopedModel } from "../model-resolver.ts";
import type { ReadonlySessionManager, SessionManager } from "../session-manager.ts";
import type { BuildSystemPromptOptions } from "../system-prompt.ts";
import type {
	BeforeAgentStartEvent,
	BeforeAgentStartEventResult,
	BeforeProviderHeadersEvent,
	BeforeProviderRequestEvent,
	CompactOptions,
	ContextEvent,
	ContextEventResult,
	ContextUsage,
	EntryRenderer,
	Extension,
	ExtensionActions,
	ExtensionCommandContext,
	ExtensionCommandContextActions,
	ExtensionContext,
	ExtensionContextActions,
	ExtensionError,
	ExtensionEvent,
	ExtensionFlag,
	ExtensionMode,
	ExtensionRuntime,
	ExtensionShortcut,
	ExtensionUIContext,
	InputEvent,
	InputEventResult,
	InputSource,
	LoadExtensionsResult,
	ManagedExtensionAction,
	ManagedExtensionActionRequest,
	ManagedExtensionHost,
	ManagedHostActivityBinding,
	ManagedToolActivityScope,
	MarkdownTransformer,
	MessageEndEvent,
	MessageEndEventResult,
	MessageRenderer,
	ProjectTrustContext,
	ProjectTrustEvent,
	ProjectTrustEventResult,
	ProviderConfig,
	RegisteredCommand,
	RegisteredTool,
	ReplacedSessionContext,
	ResolvedCommand,
	ResourcesDiscoverEvent,
	ResourcesDiscoverResult,
	SessionBeforeCompactResult,
	SessionBeforeForkResult,
	SessionBeforeSwitchResult,
	SessionBeforeTreeResult,
	SessionShutdownEvent,
	ToolCallEvent,
	ToolCallEventResult,
	ToolResultEvent,
	ToolResultEventResult,
	UserBashEvent,
	UserBashEventResult,
} from "./types.ts";

interface ActiveManagedActionScope {
	readonly scopeId: string;
	readonly extensionPath: string;
	readonly eventType: string;
	readonly handlerIndex: number;
	readonly binding: ManagedHostActivityBinding;
	readonly pending: Promise<unknown>[];
	nextActionIndex: number;
	open: boolean;
}

class ManagedExtensionBoundaryError extends ManagedAgentFailStopError {
	constructor(message: string, options?: ErrorOptions) {
		super("managed_host_boundary", message, options);
		this.name = "ManagedExtensionBoundaryError";
	}
}

function sameManagedBinding(left: ManagedHostActivityBinding, right: ManagedHostActivityBinding): boolean {
	return (
		left.generationId === right.generationId &&
		left.generationLeaseToken === right.generationLeaseToken &&
		left.activityToken === right.activityToken &&
		left.runId === right.runId &&
		left.coreInvocationId === right.coreInvocationId
	);
}

function createReadonlySessionManager(manager: SessionManager): ReadonlySessionManager {
	return Object.freeze({
		getCwd: () => manager.getCwd(),
		getSessionDir: () => manager.getSessionDir(),
		getSessionId: () => manager.getSessionId(),
		getSessionFile: () => manager.getSessionFile(),
		getLeafId: () => manager.getLeafId(),
		getLeafEntry: () => {
			const entry = manager.getLeafEntry();
			return entry ? structuredClone(entry) : undefined;
		},
		getEntry: (id: string) => {
			const entry = manager.getEntry(id);
			return entry ? structuredClone(entry) : undefined;
		},
		getLabel: (id: string) => manager.getLabel(id),
		getBranch: (leafId?: string) => structuredClone(manager.getBranch(leafId)),
		buildContextEntries: () => structuredClone(manager.buildContextEntries()),
		getHeader: () => {
			const header = manager.getHeader();
			return header ? structuredClone(header) : null;
		},
		getEntries: () => structuredClone(manager.getEntries()),
		getTree: () => structuredClone(manager.getTree()),
		getSessionName: () => manager.getSessionName(),
	});
}

function createReadonlyModelRegistry(registry: ModelRegistry): ModelRegistry {
	return new Proxy(registry, {
		get(_target, property) {
			switch (property) {
				case "getError":
					return () => registry.getError();
				case "getAll":
					return () => structuredClone(registry.getAll());
				case "getAvailable":
					return () => structuredClone(registry.getAvailable());
				case "find":
					return (provider: string, modelId: string) => {
						const model = registry.find(provider, modelId);
						return model ? structuredClone(model) : undefined;
					};
				case "hasConfiguredAuth":
					return (model: Model<Api>) => registry.hasConfiguredAuth(model);
				case "getProviderAuthStatus":
					return (provider: string) => structuredClone(registry.getProviderAuthStatus(provider));
				case "getProviderDisplayName":
					return (provider: string) => registry.getProviderDisplayName(provider);
				case "isUsingOAuth":
					return (model: Model<Api>) => registry.isUsingOAuth(model);
				case "getRegisteredProviderIds":
					return () => [...registry.getRegisteredProviderIds()];
				default:
					throw new ManagedExtensionBoundaryError(
						`Managed extension modelRegistry.${String(property)} is not a pure catalogue read`,
					);
			}
		},
		set() {
			throw new ManagedExtensionBoundaryError("Managed extension modelRegistry is read-only");
		},
	});
}

// Extension shortcuts compete with canonical keybinding ids from keybindings.json.
// Only editor-global shortcuts are reserved here. Picker-specific bindings are not.
const RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS = [
	"app.interrupt",
	"app.clear",
	"app.exit",
	"app.suspend",
	"app.thinking.cycle",
	"app.model.cycleForward",
	"app.model.cycleBackward",
	"app.model.select",
	"app.tools.expand",
	"app.thinking.toggle",
	"app.editor.external",
	"app.message.copy",
	"app.message.followUp",
	"tui.input.submit",
	"tui.select.confirm",
	"tui.select.cancel",
	"tui.input.copy",
	"tui.editor.deleteToLineEnd",
] as const;

type BuiltInKeyBindings = Partial<Record<KeyId, { keybinding: string; restrictOverride: boolean }>>;

const buildBuiltinKeybindings = (resolvedKeybindings: KeybindingsConfig): BuiltInKeyBindings => {
	const builtinKeybindings = {} as BuiltInKeyBindings;
	for (const [keybinding, keys] of Object.entries(resolvedKeybindings)) {
		if (keys === undefined) continue;
		const keyList = Array.isArray(keys) ? keys : [keys];
		const restrictOverride = (RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS as readonly string[]).includes(keybinding);
		for (const key of keyList) {
			const normalizedKey = key.toLowerCase() as KeyId;
			// If multiple actions bind the same key, the reserved action wins so extensions
			// remain blocked by reserved shortcuts regardless of iteration order.
			const existing = builtinKeybindings[normalizedKey];
			if (existing?.restrictOverride && !restrictOverride) continue;
			builtinKeybindings[normalizedKey] = {
				keybinding,
				restrictOverride,
			};
		}
	}
	return builtinKeybindings;
};

/** Combined result from all before_agent_start handlers */
interface BeforeAgentStartCombinedResult {
	messages?: NonNullable<BeforeAgentStartEventResult["message"]>[];
	systemPrompt?: string;
}

/**
 * Events handled by the generic emit() method.
 * Events with dedicated emitXxx() methods are excluded for stronger type safety.
 */
type RunnerEmitEvent = Exclude<
	ExtensionEvent,
	| ToolCallEvent
	| ProjectTrustEvent
	| ToolResultEvent
	| UserBashEvent
	| ContextEvent
	| BeforeProviderRequestEvent
	| BeforeProviderHeadersEvent
	| BeforeAgentStartEvent
	| MessageEndEvent
	| ResourcesDiscoverEvent
	| InputEvent
>;

type SessionBeforeEvent = Extract<
	RunnerEmitEvent,
	{ type: "session_before_switch" | "session_before_fork" | "session_before_compact" | "session_before_tree" }
>;

type SessionBeforeEventResult =
	| SessionBeforeSwitchResult
	| SessionBeforeForkResult
	| SessionBeforeCompactResult
	| SessionBeforeTreeResult;

type RunnerEmitResult<TEvent extends RunnerEmitEvent> = TEvent extends { type: "session_before_switch" }
	? SessionBeforeSwitchResult | undefined
	: TEvent extends { type: "session_before_fork" }
		? SessionBeforeForkResult | undefined
		: TEvent extends { type: "session_before_compact" }
			? SessionBeforeCompactResult | undefined
			: TEvent extends { type: "session_before_tree" }
				? SessionBeforeTreeResult | undefined
				: undefined;

export type ExtensionErrorListener = (error: ExtensionError) => void;

export type NewSessionHandler = (options?: {
	parentSession?: string;
	setup?: (sessionManager: SessionManager) => Promise<void>;
	withSession?: (ctx: ReplacedSessionContext) => Promise<void>;
}) => Promise<{ cancelled: boolean }>;

export type ForkHandler = (
	entryId: string,
	options?: { position?: "before" | "at"; withSession?: (ctx: ReplacedSessionContext) => Promise<void> },
) => Promise<{ cancelled: boolean }>;

export type NavigateTreeHandler = (
	targetId: string,
	options?: { summarize?: boolean; customInstructions?: string; replaceInstructions?: boolean; label?: string },
) => Promise<{ cancelled: boolean }>;

export type SwitchSessionHandler = (
	sessionPath: string,
	options?: { withSession?: (ctx: ReplacedSessionContext) => Promise<void> },
) => Promise<{ cancelled: boolean }>;

export type ReloadHandler = () => Promise<void>;

export type ShutdownHandler = () => void;

/**
 * Helper function to emit session_shutdown event to extensions.
 * Returns true if the event was emitted, false if there were no handlers.
 */
export async function emitSessionShutdownEvent(
	extensionRunner: ExtensionRunner,
	event: SessionShutdownEvent,
): Promise<boolean> {
	if (extensionRunner.hasHandlers("session_shutdown")) {
		await extensionRunner.emit(event);
		return true;
	}
	return false;
}

export async function emitProjectTrustEvent(
	extensionsResult: LoadExtensionsResult,
	event: ProjectTrustEvent,
	ctx: ProjectTrustContext,
): Promise<{ result?: ProjectTrustEventResult; errors: ExtensionError[] }> {
	const errors: ExtensionError[] = [];
	for (const ext of extensionsResult.extensions) {
		// A single extension may register multiple handlers for the same event.
		// The first project_trust handler that returns yes/no wins; undecided falls through.
		const handlers = ext.handlers.get("project_trust");
		if (!handlers || handlers.length === 0) continue;

		for (const handler of handlers) {
			try {
				const handlerResult = (await handler(event, ctx)) as ProjectTrustEventResult;
				if (handlerResult.trusted === "undecided") {
					continue;
				}
				return { result: handlerResult, errors };
			} catch (error) {
				errors.push({
					extensionPath: ext.path,
					event: event.type,
					error: error instanceof Error ? error.message : String(error),
					stack: error instanceof Error ? error.stack : undefined,
				});
			}
		}
	}
	return { errors };
}

const noOpUIContext: ExtensionUIContext = {
	select: async () => undefined,
	confirm: async () => false,
	input: async () => undefined,
	notify: () => {},
	onTerminalInput: () => () => {},
	setStatus: () => {},
	setWorkingMessage: () => {},
	setWorkingVisible: () => {},
	setWorkingIndicator: () => {},
	setHiddenThinkingLabel: () => {},
	setWidget: () => {},
	setFooter: () => {},
	setHeader: () => {},
	setTitle: () => {},
	custom: async () => undefined as never,
	pasteToEditor: () => {},
	setEditorText: () => {},
	getEditorText: () => "",
	editor: async () => undefined,
	addAutocompleteProvider: () => {},
	setEditorComponent: () => {},
	getEditorComponent: () => undefined,
	get theme() {
		return theme;
	},
	getAllThemes: () => [],
	getTheme: () => undefined,
	setTheme: (_theme: string | Theme) => ({ success: false, error: "UI not available" }),
	getToolsExpanded: () => false,
	setToolsExpanded: () => {},
};

export class ExtensionRunner {
	private extensions: Extension[];
	private runtime: ExtensionRuntime;
	private uiContext: ExtensionUIContext;
	private mode: ExtensionMode = "print";
	private cwd: string;
	private modelRegistry: ModelRegistry;
	private readonly readonlySessionManager: ReadonlySessionManager;
	private readonly readonlyModelRegistry: ModelRegistry;
	private readonly managedHost: ManagedExtensionHost | undefined;
	private readonly managedActionStorage = new AsyncLocalStorage<ActiveManagedActionScope>();
	private managedScopeSequence = 0;
	private errorListeners: Set<ExtensionErrorListener> = new Set();
	private getModel: () => Model<any> | undefined = () => undefined;
	private getScopedModels: () => readonly ScopedModel[] = () => [];
	private isIdleFn: () => boolean = () => true;
	private isProjectTrustedFn: () => boolean = () => true;
	private getSignalFn: () => AbortSignal | undefined = () => undefined;
	private waitForIdleFn: () => Promise<void> = async () => {};
	private abortFn: () => void = () => {};
	private hasPendingMessagesFn: () => boolean = () => false;
	private getContextUsageFn: () => ContextUsage | undefined = () => undefined;
	private compactFn: (options?: CompactOptions) => void = () => {};
	private getSystemPromptFn: () => string = () => "";
	private getSystemPromptOptionsFn: () => BuildSystemPromptOptions = () => ({ cwd: this.cwd });
	private newSessionHandler: NewSessionHandler = async () => ({ cancelled: false });
	private forkHandler: ForkHandler = async () => ({ cancelled: false });
	private navigateTreeHandler: NavigateTreeHandler = async () => ({ cancelled: false });
	private switchSessionHandler: SwitchSessionHandler = async () => ({ cancelled: false });
	private reloadHandler: ReloadHandler = async () => {};
	private shutdownHandler: ShutdownHandler = () => {};
	private shortcutDiagnostics: ResourceDiagnostic[] = [];
	private commandDiagnostics: ResourceDiagnostic[] = [];
	private staleMessage: string | undefined;

	constructor(
		extensions: Extension[],
		runtime: ExtensionRuntime,
		cwd: string,
		sessionManager: SessionManager,
		modelRegistry: ModelRegistry,
		managedHost?: ManagedExtensionHost,
	) {
		this.extensions = extensions;
		this.runtime = runtime;
		this.uiContext = noOpUIContext;
		this.cwd = cwd;
		this.modelRegistry = modelRegistry;
		this.readonlySessionManager = createReadonlySessionManager(sessionManager);
		this.readonlyModelRegistry = managedHost ? createReadonlyModelRegistry(modelRegistry) : modelRegistry;
		this.managedHost = managedHost;
	}

	private requireManagedBinding(): ManagedHostActivityBinding {
		const binding = this.managedHost?.getActivityBinding();
		if (!binding || !binding.generationId || !binding.generationLeaseToken || !binding.activityToken) {
			throw new ManagedExtensionBoundaryError("Managed extension scope has no active host activity binding");
		}
		return binding;
	}

	private async runManagedScope<T>(
		extensionPath: string,
		eventType: string,
		handlerIndex: number,
		callback: () => T | Promise<T>,
		binding?: ManagedHostActivityBinding,
	): Promise<T> {
		if (!this.managedHost) {
			return await callback();
		}
		const resolvedBinding = binding ?? this.managedHost.getActivityBinding();
		if (!resolvedBinding) {
			try {
				return await this.managedActionStorage.exit(callback);
			} catch (error) {
				if (error instanceof ManagedExtensionBoundaryError) throw error;
				throw new ManagedExtensionBoundaryError(
					`Managed extension read-only scope ${eventType} failed: ${error instanceof Error ? error.message : String(error)}`,
					{ cause: error },
				);
			}
		}
		if (!resolvedBinding.generationId || !resolvedBinding.generationLeaseToken || !resolvedBinding.activityToken) {
			throw new ManagedExtensionBoundaryError("Managed extension scope received an invalid host activity binding");
		}
		const scope: ActiveManagedActionScope = {
			scopeId: `${resolvedBinding.activityToken}:${++this.managedScopeSequence}`,
			extensionPath,
			eventType,
			handlerIndex,
			binding: resolvedBinding,
			pending: [],
			nextActionIndex: 0,
			open: true,
		};

		try {
			const result = await this.managedActionStorage.run(scope, callback);
			let drained = 0;
			while (drained < scope.pending.length) {
				const pending = scope.pending.slice(drained);
				drained = scope.pending.length;
				await Promise.all(pending);
			}
			return result;
		} catch (error) {
			if (error instanceof ManagedExtensionBoundaryError) throw error;
			throw new ManagedExtensionBoundaryError(
				`Managed extension scope ${scope.scopeId} failed: ${error instanceof Error ? error.message : String(error)}`,
				{ cause: error },
			);
		} finally {
			scope.open = false;
		}
	}

	private async invokeExtensionHandler<T>(
		extensionPath: string,
		eventType: string,
		handlerIndex: number,
		callback: () => T | Promise<T>,
	): Promise<T> {
		return this.runManagedScope(extensionPath, eventType, handlerIndex, callback);
	}

	private rethrowManagedBoundary(error: unknown): void {
		if (error instanceof ManagedExtensionBoundaryError) throw error;
	}

	private dispatchManagedAction<T>(
		extensionPath: string | undefined,
		action: ManagedExtensionAction,
		execute: () => T | Promise<T>,
	): Promise<T> {
		if (!this.managedHost) {
			return Promise.resolve(execute());
		}

		const active = this.managedActionStorage.getStore();
		if (!active || !active.open) {
			throw new ManagedExtensionBoundaryError(
				`Managed extension action ${action.type} was invoked outside an active ExtensionActionScope`,
			);
		}
		if (extensionPath !== undefined && extensionPath !== active.extensionPath) {
			throw new ManagedExtensionBoundaryError(
				`Managed extension identity mismatch: expected ${active.extensionPath}, received ${extensionPath}`,
			);
		}

		const currentBinding = this.requireManagedBinding();
		if (!sameManagedBinding(active.binding, currentBinding)) {
			throw new ManagedExtensionBoundaryError(`Managed extension scope ${active.scopeId} crossed an activity fence`);
		}

		const request: ManagedExtensionActionRequest = {
			scope: {
				scopeId: active.scopeId,
				extensionPath: active.extensionPath,
				eventType: active.eventType,
				handlerIndex: active.handlerIndex,
				actionIndex: ++active.nextActionIndex,
				binding: active.binding,
			},
			action,
		};
		const executeAdmittedAction = async (): Promise<T> => {
			if (!active.open) {
				throw new ManagedExtensionBoundaryError(
					`Managed extension scope ${active.scopeId} closed before admitted action ${action.type} executed`,
				);
			}
			const executionBinding = this.requireManagedBinding();
			if (!sameManagedBinding(active.binding, executionBinding)) {
				throw new ManagedExtensionBoundaryError(
					`Managed extension scope ${active.scopeId} crossed an activity fence before execution`,
				);
			}
			return await execute();
		};
		const promise = this.managedHost.dispatchExtensionAction(request, executeAdmittedAction);
		active.pending.push(promise);
		void promise.catch(() => {});
		return promise;
	}

	private startExtensionAction(extensionPath: string, event: string, promise: Promise<unknown>): void {
		if (this.managedHost) return;
		void promise.catch((error) => {
			this.emitError({
				extensionPath,
				event,
				error: error instanceof Error ? error.message : String(error),
				stack: error instanceof Error ? error.stack : undefined,
			});
		});
	}

	async executeTool<TDetails>(
		registeredTool: RegisteredTool,
		toolCallId: string,
		execute: () => Promise<AgentToolResult<TDetails>>,
	): Promise<AgentToolResult<TDetails>> {
		if (!this.managedHost) return execute();

		const binding = this.requireManagedBinding();
		if (!binding.runId || !binding.coreInvocationId) {
			throw new ManagedExtensionBoundaryError(
				`Managed tool ${registeredTool.definition.name} has no active Run/CoreInvocation binding`,
			);
		}
		const scope: ManagedToolActivityScope = {
			...binding,
			runId: binding.runId,
			coreInvocationId: binding.coreInvocationId,
			toolCallId,
			toolName: registeredTool.definition.name,
			toolSource: registeredTool.sourceInfo,
		};
		return this.runManagedScope(
			registeredTool.sourceInfo.path,
			`tool:${registeredTool.definition.name}:${toolCallId}`,
			0,
			() => this.managedHost!.executeTool(scope, execute),
			binding,
		);
	}

	bindCore(
		actions: ExtensionActions,
		contextActions: ExtensionContextActions,
		providerActions?: {
			registerProvider?: (name: string, config: ProviderConfig) => void;
			registerNativeProvider?: (provider: Provider) => void;
			unregisterProvider?: (name: string) => void;
		},
	): void {
		// Per-extension API facades preserve their identity at every mutation boundary.
		this.runtime.sendMessage = (extensionPath, message, options) => {
			this.startExtensionAction(
				extensionPath,
				"send_message",
				this.dispatchManagedAction(extensionPath, { type: "send_message", message, options }, () =>
					actions.sendMessage(message, options),
				),
			);
		};
		this.runtime.sendUserMessage = (extensionPath, content, options) => {
			this.startExtensionAction(
				extensionPath,
				"send_user_message",
				this.dispatchManagedAction(extensionPath, { type: "send_user_message", content, options }, () =>
					actions.sendUserMessage(content, options),
				),
			);
		};
		this.runtime.appendEntry = (extensionPath, customType, data) => {
			this.startExtensionAction(
				extensionPath,
				"append_entry",
				this.dispatchManagedAction(extensionPath, { type: "append_entry", customType, data }, () =>
					actions.appendEntry(customType, data),
				),
			);
		};
		this.runtime.setSessionName = (extensionPath, name) => {
			this.startExtensionAction(
				extensionPath,
				"set_session_name",
				this.dispatchManagedAction(extensionPath, { type: "set_session_name", name }, () =>
					actions.setSessionName(name),
				),
			);
		};
		this.runtime.getSessionName = actions.getSessionName;
		this.runtime.setLabel = (extensionPath, entryId, label) => {
			this.startExtensionAction(
				extensionPath,
				"set_label",
				this.dispatchManagedAction(extensionPath, { type: "set_label", entryId, label }, () =>
					actions.setLabel(entryId, label),
				),
			);
		};
		this.runtime.exec = (extensionPath, command, args, options) =>
			this.dispatchManagedAction(extensionPath, { type: "exec", command, args, options }, () =>
				actions.exec(command, args, options),
			);
		this.runtime.getActiveTools = actions.getActiveTools;
		this.runtime.getAllTools = actions.getAllTools;
		this.runtime.setActiveTools = (extensionPath, toolNames) => {
			this.startExtensionAction(
				extensionPath,
				"set_active_tools",
				this.dispatchManagedAction(extensionPath, { type: "set_active_tools", toolNames }, () =>
					actions.setActiveTools(toolNames),
				),
			);
		};
		this.runtime.refreshTools = actions.refreshTools;
		this.runtime.getCommands = actions.getCommands;
		this.runtime.setModel = (extensionPath, model) =>
			this.dispatchManagedAction(extensionPath, { type: "set_model", model }, () => actions.setModel(model));
		this.runtime.getThinkingLevel = actions.getThinkingLevel;
		this.runtime.setThinkingLevel = (extensionPath, level) => {
			this.startExtensionAction(
				extensionPath,
				"set_thinking_level",
				this.dispatchManagedAction(extensionPath, { type: "set_thinking_level", level }, () =>
					actions.setThinkingLevel(level),
				),
			);
		};

		// Context actions (required)
		this.getModel = contextActions.getModel;
		this.getScopedModels = contextActions.getScopedModels;
		this.isIdleFn = contextActions.isIdle;
		this.isProjectTrustedFn = contextActions.isProjectTrusted;
		this.getSignalFn = contextActions.getSignal;
		this.abortFn = contextActions.abort;
		this.hasPendingMessagesFn = contextActions.hasPendingMessages;
		this.shutdownHandler = contextActions.shutdown;
		this.getContextUsageFn = contextActions.getContextUsage;
		this.compactFn = contextActions.compact;
		this.getSystemPromptFn = contextActions.getSystemPrompt;
		this.getSystemPromptOptionsFn = contextActions.getSystemPromptOptions ?? (() => ({ cwd: this.cwd }));

		// Flush provider registrations queued during extension loading
		if (
			this.managedHost &&
			(this.runtime.pendingProviderRegistrations.length > 0 ||
				this.runtime.pendingNativeProviderRegistrations.length > 0)
		) {
			throw new ManagedExtensionBoundaryError(
				"Managed extensions cannot register providers during module initialization",
			);
		}
		for (const { name, config, extensionPath } of this.runtime.pendingProviderRegistrations) {
			try {
				if (providerActions?.registerProvider) {
					providerActions.registerProvider(name, config);
				} else {
					this.modelRegistry.registerProvider(name, config);
				}
			} catch (err) {
				this.emitError({
					extensionPath,
					event: "register_provider",
					error: err instanceof Error ? err.message : String(err),
					stack: err instanceof Error ? err.stack : undefined,
				});
			}
		}
		this.runtime.pendingProviderRegistrations = [];
		for (const { provider, extensionPath } of this.runtime.pendingNativeProviderRegistrations) {
			try {
				if (providerActions?.registerNativeProvider) {
					providerActions.registerNativeProvider(provider);
				} else {
					this.modelRegistry.registerProvider(provider);
				}
			} catch (err) {
				this.emitError({
					extensionPath,
					event: "register_provider",
					error: err instanceof Error ? err.message : String(err),
					stack: err instanceof Error ? err.stack : undefined,
				});
			}
		}
		this.runtime.pendingNativeProviderRegistrations = [];

		// From this point on, provider registration/unregistration takes effect immediately
		// without requiring a /reload.
		this.runtime.registerProvider = (name, config, extensionPath) => {
			this.startExtensionAction(
				extensionPath,
				"register_provider",
				this.dispatchManagedAction(extensionPath, { type: "register_provider", name, config }, () => {
					if (providerActions?.registerProvider) {
						providerActions.registerProvider(name, config);
						return;
					}
					this.modelRegistry.registerProvider(name, config);
				}),
			);
		};
		this.runtime.registerNativeProvider = (provider, extensionPath) => {
			this.startExtensionAction(
				extensionPath,
				"register_native_provider",
				this.dispatchManagedAction(extensionPath, { type: "register_native_provider", provider }, () => {
					if (providerActions?.registerNativeProvider) {
						providerActions.registerNativeProvider(provider);
						return;
					}
					this.modelRegistry.registerProvider(provider);
				}),
			);
		};
		this.runtime.unregisterProvider = (name, extensionPath) => {
			this.startExtensionAction(
				extensionPath,
				"unregister_provider",
				this.dispatchManagedAction(extensionPath, { type: "unregister_provider", name }, () => {
					if (providerActions?.unregisterProvider) {
						providerActions.unregisterProvider(name);
						return;
					}
					this.modelRegistry.unregisterProvider(name);
				}),
			);
		};
	}

	bindCommandContext(actions?: ExtensionCommandContextActions): void {
		if (actions) {
			this.waitForIdleFn = actions.waitForIdle;
			this.newSessionHandler = actions.newSession;
			this.forkHandler = actions.fork;
			this.navigateTreeHandler = actions.navigateTree;
			this.switchSessionHandler = actions.switchSession;
			this.reloadHandler = actions.reload;
			return;
		}

		this.waitForIdleFn = async () => {};
		this.newSessionHandler = async () => ({ cancelled: false });
		this.forkHandler = async () => ({ cancelled: false });
		this.navigateTreeHandler = async () => ({ cancelled: false });
		this.switchSessionHandler = async () => ({ cancelled: false });
		this.reloadHandler = async () => {};
	}

	setUIContext(uiContext?: ExtensionUIContext, mode: ExtensionMode = "print"): void {
		this.uiContext = uiContext ?? noOpUIContext;
		this.mode = mode;
	}

	getUIContext(): ExtensionUIContext {
		return this.uiContext;
	}

	hasUI(): boolean {
		return this.uiContext !== noOpUIContext;
	}

	getExtensionPaths(): string[] {
		return this.extensions.map((e) => e.path);
	}

	/** Get all registered tools from all extensions (first registration per name wins). */
	getAllRegisteredTools(): RegisteredTool[] {
		const toolsByName = new Map<string, RegisteredTool>();
		for (const ext of this.extensions) {
			for (const tool of ext.tools.values()) {
				if (!toolsByName.has(tool.definition.name)) {
					toolsByName.set(tool.definition.name, tool);
				}
			}
		}
		return Array.from(toolsByName.values());
	}

	/** Get a tool definition by name. Returns undefined if not found. */
	getToolDefinition(toolName: string): RegisteredTool["definition"] | undefined {
		for (const ext of this.extensions) {
			const tool = ext.tools.get(toolName);
			if (tool) {
				return tool.definition;
			}
		}
		return undefined;
	}

	getFlags(): Map<string, ExtensionFlag> {
		const allFlags = new Map<string, ExtensionFlag>();
		for (const ext of this.extensions) {
			for (const [name, flag] of ext.flags) {
				if (!allFlags.has(name)) {
					allFlags.set(name, flag);
				}
			}
		}
		return allFlags;
	}

	setFlagValue(name: string, value: boolean | string): void {
		this.runtime.flagValues.set(name, value);
	}

	getFlagValues(): Map<string, boolean | string> {
		return new Map(this.runtime.flagValues);
	}

	getShortcuts(resolvedKeybindings: KeybindingsConfig): Map<KeyId, ExtensionShortcut> {
		this.shortcutDiagnostics = [];
		const builtinKeybindings = buildBuiltinKeybindings(resolvedKeybindings);
		const extensionShortcuts = new Map<KeyId, ExtensionShortcut>();

		const addDiagnostic = (message: string, extensionPath: string) => {
			this.shortcutDiagnostics.push({ type: "warning", message, path: extensionPath });
			if (!this.hasUI()) {
				console.warn(message);
			}
		};

		for (const ext of this.extensions) {
			for (const [key, shortcut] of ext.shortcuts) {
				const normalizedKey = key.toLowerCase() as KeyId;

				const builtInKeybinding = builtinKeybindings[normalizedKey];
				if (builtInKeybinding?.restrictOverride === true) {
					addDiagnostic(
						`Extension shortcut '${key}' from ${shortcut.extensionPath} conflicts with built-in shortcut. Skipping.`,
						shortcut.extensionPath,
					);
					continue;
				}

				if (builtInKeybinding?.restrictOverride === false) {
					addDiagnostic(
						`Extension shortcut conflict: '${key}' is built-in shortcut for ${builtInKeybinding.keybinding} and ${shortcut.extensionPath}. Using ${shortcut.extensionPath}.`,
						shortcut.extensionPath,
					);
				}

				const existingExtensionShortcut = extensionShortcuts.get(normalizedKey);
				if (existingExtensionShortcut) {
					addDiagnostic(
						`Extension shortcut conflict: '${key}' registered by both ${existingExtensionShortcut.extensionPath} and ${shortcut.extensionPath}. Using ${shortcut.extensionPath}.`,
						shortcut.extensionPath,
					);
				}
				extensionShortcuts.set(normalizedKey, {
					...shortcut,
					handler: (ctx) =>
						this.runManagedScope(shortcut.extensionPath, `shortcut:${normalizedKey}`, 0, () =>
							shortcut.handler(ctx),
						),
				});
			}
		}
		return extensionShortcuts;
	}

	getShortcutDiagnostics(): ResourceDiagnostic[] {
		return this.shortcutDiagnostics;
	}

	invalidate(
		message = "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().",
	): void {
		if (!this.staleMessage) {
			this.staleMessage = message;
			this.runtime.invalidate(message);
		}
	}

	private assertActive(): void {
		if (this.staleMessage) {
			throw new Error(this.staleMessage);
		}
	}

	onError(listener: ExtensionErrorListener): () => void {
		this.errorListeners.add(listener);
		return () => this.errorListeners.delete(listener);
	}

	emitError(error: ExtensionError): void {
		for (const listener of this.errorListeners) {
			listener(error);
		}
	}

	hasHandlers(eventType: string): boolean {
		for (const ext of this.extensions) {
			const handlers = ext.handlers.get(eventType);
			if (handlers && handlers.length > 0) {
				return true;
			}
		}
		return false;
	}

	getMessageRenderer(customType: string): MessageRenderer | undefined {
		for (const ext of this.extensions) {
			const renderer = ext.messageRenderers.get(customType);
			if (renderer) {
				return renderer;
			}
		}
		return undefined;
	}

	getMarkdownTransformers(): MarkdownTransformer[] {
		return this.extensions.flatMap((ext) => (ext.markdownTransformer ? [ext.markdownTransformer] : []));
	}

	getEntryRenderer(customType: string): EntryRenderer | undefined {
		for (const ext of this.extensions) {
			const renderer = ext.entryRenderers?.get(customType);
			if (renderer) {
				return renderer;
			}
		}
		return undefined;
	}

	private resolveRegisteredCommands(): ResolvedCommand[] {
		const commands: RegisteredCommand[] = [];
		const counts = new Map<string, number>();

		for (const ext of this.extensions) {
			for (const command of ext.commands.values()) {
				commands.push(command);
				counts.set(command.name, (counts.get(command.name) ?? 0) + 1);
			}
		}

		const seen = new Map<string, number>();
		const takenInvocationNames = new Set<string>();

		return commands.map((command) => {
			const occurrence = (seen.get(command.name) ?? 0) + 1;
			seen.set(command.name, occurrence);

			let invocationName = (counts.get(command.name) ?? 0) > 1 ? `${command.name}:${occurrence}` : command.name;

			if (takenInvocationNames.has(invocationName)) {
				let suffix = occurrence;
				do {
					suffix++;
					invocationName = `${command.name}:${suffix}`;
				} while (takenInvocationNames.has(invocationName));
			}

			takenInvocationNames.add(invocationName);
			return {
				...command,
				invocationName,
				handler: (args, ctx) =>
					this.runManagedScope(command.sourceInfo.path, `command:${invocationName}`, 0, () =>
						command.handler(args, ctx),
					),
			};
		});
	}

	getModelRegistry(): ModelRegistry {
		return this.modelRegistry;
	}

	getRegisteredCommands(): ResolvedCommand[] {
		this.commandDiagnostics = [];
		return this.resolveRegisteredCommands();
	}

	getCommandDiagnostics(): ResourceDiagnostic[] {
		return this.commandDiagnostics;
	}

	getCommand(name: string): ResolvedCommand | undefined {
		return this.resolveRegisteredCommands().find((command) => command.invocationName === name);
	}

	/**
	 * Request a graceful shutdown. Called by extension tools and event handlers.
	 * The actual shutdown behavior is provided by the mode via bindExtensions().
	 */
	shutdown(): void {
		this.shutdownHandler();
	}

	getActiveTools(): string[] {
		this.assertActive();
		return this.runtime.getActiveTools();
	}

	/**
	 * Create an ExtensionContext for use in event handlers and tool execution.
	 * Context values are resolved at call time, so changes via bindCore/bindUI are reflected.
	 */
	createContext(): ExtensionContext {
		const runner = this;
		const getModel = this.getModel;
		const getScopedModels = this.getScopedModels;
		return {
			get ui() {
				runner.assertActive();
				return runner.uiContext;
			},
			get mode() {
				runner.assertActive();
				return runner.mode;
			},
			get hasUI() {
				runner.assertActive();
				return runner.hasUI();
			},
			get cwd() {
				runner.assertActive();
				return runner.cwd;
			},
			get sessionManager() {
				runner.assertActive();
				return runner.readonlySessionManager;
			},
			get modelRegistry() {
				runner.assertActive();
				return runner.readonlyModelRegistry;
			},
			get model() {
				runner.assertActive();
				return getModel();
			},
			get scopedModels() {
				runner.assertActive();
				return getScopedModels();
			},
			get thinkingLevel() {
				runner.assertActive();
				return runner.runtime.getThinkingLevel();
			},
			isIdle: () => {
				runner.assertActive();
				return runner.isIdleFn();
			},
			isProjectTrusted: () => {
				runner.assertActive();
				return runner.isProjectTrustedFn();
			},
			get signal() {
				runner.assertActive();
				return runner.getSignalFn();
			},
			abort: () => {
				runner.assertActive();
				void runner.dispatchManagedAction(undefined, { type: "abort" }, () => runner.abortFn());
			},
			hasPendingMessages: () => {
				runner.assertActive();
				return runner.hasPendingMessagesFn();
			},
			shutdown: () => {
				runner.assertActive();
				void runner.dispatchManagedAction(undefined, { type: "shutdown" }, () => runner.shutdownHandler());
			},
			getContextUsage: () => {
				runner.assertActive();
				return runner.getContextUsageFn();
			},
			compact: (options) => {
				runner.assertActive();
				void runner.dispatchManagedAction(undefined, { type: "compact", options }, () => runner.compactFn(options));
			},
			getSystemPrompt: () => {
				runner.assertActive();
				return runner.getSystemPromptFn();
			},
		};
	}

	createCommandContext(): ExtensionCommandContext {
		// Use property descriptors instead of object spread so the guarded getters from
		// createContext() stay lazy. A spread would eagerly read them once and freeze the
		// old values into the returned object, bypassing stale-instance checks.
		const context = Object.defineProperties(
			{},
			Object.getOwnPropertyDescriptors(this.createContext()),
		) as ExtensionCommandContext;
		context.getSystemPromptOptions = () => {
			this.assertActive();
			return this.getSystemPromptOptionsFn();
		};
		context.waitForIdle = () => {
			this.assertActive();
			return this.waitForIdleFn();
		};
		context.newSession = (options) => {
			this.assertActive();
			return this.dispatchManagedAction(undefined, { type: "new_session", options }, () =>
				this.newSessionHandler(options),
			);
		};
		context.fork = (entryId, options) => {
			this.assertActive();
			return this.dispatchManagedAction(undefined, { type: "fork", entryId, options }, () =>
				this.forkHandler(entryId, options),
			);
		};
		context.navigateTree = (targetId, options) => {
			this.assertActive();
			return this.dispatchManagedAction(undefined, { type: "navigate_tree", targetId, options }, () =>
				this.navigateTreeHandler(targetId, options),
			);
		};
		context.switchSession = (sessionPath, options) => {
			this.assertActive();
			return this.dispatchManagedAction(undefined, { type: "switch_session", sessionPath, options }, () =>
				this.switchSessionHandler(sessionPath, options),
			);
		};
		context.reload = () => {
			this.assertActive();
			return this.dispatchManagedAction(undefined, { type: "reload" }, () => this.reloadHandler());
		};
		return context;
	}

	private isSessionBeforeEvent(event: RunnerEmitEvent): event is SessionBeforeEvent {
		return (
			event.type === "session_before_switch" ||
			event.type === "session_before_fork" ||
			event.type === "session_before_compact" ||
			event.type === "session_before_tree"
		);
	}

	async emit<TEvent extends RunnerEmitEvent>(event: TEvent): Promise<RunnerEmitResult<TEvent>> {
		const ctx = this.createContext();
		let result: SessionBeforeEventResult | undefined;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get(event.type);
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const handlerResult = await this.invokeExtensionHandler(ext.path, event.type, handlerIndex, () =>
						handler(event, ctx),
					);

					if (this.isSessionBeforeEvent(event) && handlerResult) {
						result = handlerResult as SessionBeforeEventResult;
						if (result.cancel) {
							return result as RunnerEmitResult<TEvent>;
						}
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: event.type,
						error: message,
						stack,
					});
				}
			}
		}

		return result as RunnerEmitResult<TEvent>;
	}

	async emitMessageEnd(event: MessageEndEvent): Promise<AgentMessage | undefined> {
		const ctx = this.createContext();
		let currentMessage = event.message;
		let modified = false;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("message_end");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const currentEvent: MessageEndEvent = { ...event, message: currentMessage };
					const handlerResult = (await this.invokeExtensionHandler(ext.path, "message_end", handlerIndex, () =>
						handler(currentEvent, ctx),
					)) as MessageEndEventResult | undefined;
					if (!handlerResult?.message) continue;

					if (handlerResult.message.role !== currentMessage.role) {
						this.emitError({
							extensionPath: ext.path,
							event: "message_end",
							error: "message_end handlers must return a message with the same role",
						});
						continue;
					}

					currentMessage = handlerResult.message;
					modified = true;
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "message_end",
						error: message,
						stack,
					});
				}
			}
		}

		return modified ? currentMessage : undefined;
	}

	async emitToolResult(event: ToolResultEvent): Promise<ToolResultEventResult | undefined> {
		const ctx = this.createContext();
		const currentEvent: ToolResultEvent = { ...event };
		let modified = false;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("tool_result");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const handlerResult = (await this.invokeExtensionHandler(ext.path, "tool_result", handlerIndex, () =>
						handler(currentEvent, ctx),
					)) as ToolResultEventResult | undefined;
					if (!handlerResult) continue;

					if (handlerResult.content !== undefined) {
						currentEvent.content = handlerResult.content;
						modified = true;
					}
					if (handlerResult.details !== undefined) {
						currentEvent.details = handlerResult.details;
						modified = true;
					}
					if (handlerResult.isError !== undefined) {
						currentEvent.isError = handlerResult.isError;
						modified = true;
					}
					if (handlerResult.usage !== undefined) {
						currentEvent.usage = handlerResult.usage;
						modified = true;
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "tool_result",
						error: message,
						stack,
					});
				}
			}
		}

		if (!modified) {
			return undefined;
		}

		return {
			content: currentEvent.content,
			details: currentEvent.details,
			isError: currentEvent.isError,
			usage: currentEvent.usage,
		};
	}

	async emitToolCall(event: ToolCallEvent): Promise<ToolCallEventResult | undefined> {
		const ctx = this.createContext();
		let result: ToolCallEventResult | undefined;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("tool_call");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				const handlerResult = await this.invokeExtensionHandler(ext.path, "tool_call", handlerIndex, () =>
					handler(event, ctx),
				);

				if (handlerResult) {
					result = handlerResult as ToolCallEventResult;
					if (result.block) {
						return result;
					}
				}
			}
		}

		return result;
	}

	async emitUserBash(event: UserBashEvent): Promise<UserBashEventResult | undefined> {
		const ctx = this.createContext();

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("user_bash");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const handlerResult = await this.invokeExtensionHandler(ext.path, "user_bash", handlerIndex, () =>
						handler(event, ctx),
					);
					if (handlerResult) {
						return handlerResult as UserBashEventResult;
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "user_bash",
						error: message,
						stack,
					});
				}
			}
		}

		return undefined;
	}

	async emitContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
		const ctx = this.createContext();
		let currentMessages = structuredClone(messages);

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("context");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const event: ContextEvent = { type: "context", messages: currentMessages };
					const handlerResult = await this.invokeExtensionHandler(ext.path, "context", handlerIndex, () =>
						handler(event, ctx),
					);

					if (handlerResult && (handlerResult as ContextEventResult).messages) {
						currentMessages = (handlerResult as ContextEventResult).messages!;
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "context",
						error: message,
						stack,
					});
				}
			}
		}

		return currentMessages;
	}

	async emitBeforeProviderRequest(payload: unknown): Promise<unknown> {
		const ctx = this.createContext();
		let currentPayload = payload;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("before_provider_request");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const event: BeforeProviderRequestEvent = {
						type: "before_provider_request",
						payload: currentPayload,
					};
					const handlerResult = await this.invokeExtensionHandler(
						ext.path,
						"before_provider_request",
						handlerIndex,
						() => handler(event, ctx),
					);
					if (handlerResult !== undefined) {
						currentPayload = handlerResult;
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "before_provider_request",
						error: message,
						stack,
					});
				}
			}
		}

		return currentPayload;
	}

	async emitBeforeProviderHeaders(headers: ProviderHeaders): Promise<ProviderHeaders> {
		const ctx = this.createContext();

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("before_provider_headers");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					// Handlers mutate `headers` in place; the return value is ignored.
					const event: BeforeProviderHeadersEvent = {
						type: "before_provider_headers",
						headers,
					};
					await this.invokeExtensionHandler(ext.path, "before_provider_headers", handlerIndex, () =>
						handler(event, ctx),
					);
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "before_provider_headers",
						error: message,
						stack,
					});
				}
			}
		}

		return headers;
	}

	async emitBeforeAgentStart(
		prompt: string,
		images: ImageContent[] | undefined,
		systemPrompt: string,
		systemPromptOptions: BuildSystemPromptOptions,
	): Promise<BeforeAgentStartCombinedResult | undefined> {
		let currentSystemPrompt = systemPrompt;
		const ctx = Object.defineProperties(
			{},
			Object.getOwnPropertyDescriptors(this.createContext()),
		) as ExtensionContext;
		ctx.getSystemPrompt = () => {
			this.assertActive();
			return currentSystemPrompt;
		};
		const messages: NonNullable<BeforeAgentStartEventResult["message"]>[] = [];
		let systemPromptModified = false;

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("before_agent_start");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const event: BeforeAgentStartEvent = {
						type: "before_agent_start",
						prompt,
						images,
						systemPrompt: currentSystemPrompt,
						systemPromptOptions,
					};
					const handlerResult = await this.invokeExtensionHandler(
						ext.path,
						"before_agent_start",
						handlerIndex,
						() => handler(event, ctx),
					);

					if (handlerResult) {
						const result = handlerResult as BeforeAgentStartEventResult;
						if (result.message) {
							messages.push(result.message);
						}
						if (result.systemPrompt !== undefined) {
							currentSystemPrompt = result.systemPrompt;
							systemPromptModified = true;
						}
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "before_agent_start",
						error: message,
						stack,
					});
				}
			}
		}

		if (messages.length > 0 || systemPromptModified) {
			return {
				messages: messages.length > 0 ? messages : undefined,
				systemPrompt: systemPromptModified ? currentSystemPrompt : undefined,
			};
		}

		return undefined;
	}

	async emitResourcesDiscover(
		cwd: string,
		reason: ResourcesDiscoverEvent["reason"],
	): Promise<{
		skillPaths: Array<{ path: string; extensionPath: string }>;
		promptPaths: Array<{ path: string; extensionPath: string }>;
		themePaths: Array<{ path: string; extensionPath: string }>;
	}> {
		const ctx = this.createContext();
		const skillPaths: Array<{ path: string; extensionPath: string }> = [];
		const promptPaths: Array<{ path: string; extensionPath: string }> = [];
		const themePaths: Array<{ path: string; extensionPath: string }> = [];

		for (const ext of this.extensions) {
			const handlers = ext.handlers.get("resources_discover");
			if (!handlers || handlers.length === 0) continue;

			for (const [handlerIndex, handler] of handlers.entries()) {
				try {
					const event: ResourcesDiscoverEvent = { type: "resources_discover", cwd, reason };
					const handlerResult = await this.invokeExtensionHandler(
						ext.path,
						"resources_discover",
						handlerIndex,
						() => handler(event, ctx),
					);
					const result = handlerResult as ResourcesDiscoverResult | undefined;

					if (result?.skillPaths?.length) {
						skillPaths.push(...result.skillPaths.map((path) => ({ path, extensionPath: ext.path })));
					}
					if (result?.promptPaths?.length) {
						promptPaths.push(...result.promptPaths.map((path) => ({ path, extensionPath: ext.path })));
					}
					if (result?.themePaths?.length) {
						themePaths.push(...result.themePaths.map((path) => ({ path, extensionPath: ext.path })));
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					const message = err instanceof Error ? err.message : String(err);
					const stack = err instanceof Error ? err.stack : undefined;
					this.emitError({
						extensionPath: ext.path,
						event: "resources_discover",
						error: message,
						stack,
					});
				}
			}
		}

		return { skillPaths, promptPaths, themePaths };
	}

	/** Emit input event. Transforms chain, "handled" short-circuits. */
	async emitInput(
		text: string,
		images: ImageContent[] | undefined,
		source: InputSource,
		streamingBehavior?: "steer" | "followUp",
	): Promise<InputEventResult> {
		const ctx = this.createContext();
		let currentText = text;
		let currentImages = images;

		for (const ext of this.extensions) {
			for (const [handlerIndex, handler] of (ext.handlers.get("input") ?? []).entries()) {
				try {
					const event: InputEvent = {
						type: "input",
						text: currentText,
						images: currentImages,
						source,
						streamingBehavior,
					};
					const result = (await this.invokeExtensionHandler(ext.path, "input", handlerIndex, () =>
						handler(event, ctx),
					)) as InputEventResult | undefined;
					if (result?.action === "handled") return result;
					if (result?.action === "transform") {
						currentText = result.text;
						currentImages = result.images ?? currentImages;
					}
				} catch (err) {
					this.rethrowManagedBoundary(err);
					this.emitError({
						extensionPath: ext.path,
						event: "input",
						error: err instanceof Error ? err.message : String(err),
						stack: err instanceof Error ? err.stack : undefined,
					});
				}
			}
		}
		return currentText !== text || currentImages !== images
			? { action: "transform", text: currentText, images: currentImages }
			: { action: "continue" };
	}
}
