import type {
	ImageContent,
	Message,
	Model,
	SimpleStreamOptions,
	TextContent,
	ThinkingBudgets,
	Transport,
} from "@earendil-works/pi-ai";
import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.ts";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	BeforeToolCallContext,
	BeforeToolCallResult,
	ManagedQueueAbortResult,
	ManagedQueueAdmissionResult,
	ManagedQueueItemInput,
	ManagedQueueItemPhase,
	ManagedQueueLane,
	ManagedQueueMaterializationDecision,
	ManagedQueueMaterializationHook,
	ManagedQueueMirrorItemSnapshot,
	ManagedQueuePublishResult,
	ManagedQueueRemovalResult,
	ManagedQueueStageResult,
	ManagedQueueTicket,
	PrepareNextTurnContext,
	QueueMode,
	ShouldStopAfterTurnContext,
	StreamFn,
	ToolExecutionMode,
} from "./types.ts";

export type { QueueMode } from "./types.ts";

function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
	);
}

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const DEFAULT_MODEL = {
	id: "unknown",
	name: "unknown",
	api: "unknown",
	provider: "unknown",
	baseUrl: "",
	reasoning: false,
	input: [],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 0,
	maxTokens: 0,
} satisfies Model<any>;

type MutableAgentState = Omit<AgentState, "isStreaming" | "streamingMessage" | "pendingToolCalls" | "errorMessage"> & {
	isStreaming: boolean;
	streamingMessage?: AgentMessage;
	pendingToolCalls: Set<string>;
	errorMessage?: string;
};

function createMutableAgentState(
	initialState?: Partial<Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">>,
): MutableAgentState {
	let tools = initialState?.tools?.slice() ?? [];
	let messages = initialState?.messages?.slice() ?? [];

	return {
		systemPrompt: initialState?.systemPrompt ?? "",
		model: initialState?.model ?? DEFAULT_MODEL,
		thinkingLevel: initialState?.thinkingLevel ?? "off",
		get tools() {
			return tools;
		},
		set tools(nextTools: AgentTool<any>[]) {
			tools = nextTools.slice();
		},
		get messages() {
			return messages;
		},
		set messages(nextMessages: AgentMessage[]) {
			messages = nextMessages.slice();
		},
		isStreaming: false,
		streamingMessage: undefined,
		pendingToolCalls: new Set<string>(),
		errorMessage: undefined,
	};
}

/** Options for constructing an {@link Agent}. */
export interface AgentOptions {
	initialState?: Partial<Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">>;
	convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	streamFn: StreamFn;
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
	shouldStopAfterTurn?: (context: ShouldStopAfterTurnContext, signal?: AbortSignal) => boolean | Promise<boolean>;
	prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	/** Enables stable managed queue admission and the awaited pre-visibility materialization barrier. */
	managedQueueMaterializationHook?: ManagedQueueMaterializationHook;
	sessionId?: string;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
	toolExecution?: ToolExecutionMode;
}

type LegacyPendingMessage = {
	kind: "legacy";
	message: AgentMessage;
};

type ManagedPendingMessage = {
	kind: "managed";
	itemId: string;
	lane: ManagedQueueLane;
	message: AgentMessage;
	mirrorRevision: number;
	phase: ManagedQueueItemPhase;
};

type PendingMessage = LegacyPendingMessage | ManagedPendingMessage;

class PendingMessageQueue {
	private items: PendingMessage[] = [];
	public mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	enqueue(message: AgentMessage): void {
		this.items.push({ kind: "legacy", message });
	}

	stage(item: ManagedPendingMessage): void {
		this.items.push(item);
	}

	hasDrainableItems(): boolean {
		const first = this.items[0];
		return first !== undefined && (first.kind === "legacy" || first.phase === "published");
	}

	select(): PendingMessage[] {
		if (!this.hasDrainableItems()) {
			return [];
		}

		let count = 1;
		if (this.mode === "all") {
			count = 0;
			for (const item of this.items) {
				if (item.kind === "managed" && item.phase !== "published") {
					break;
				}
				count++;
			}
		}

		const selected = this.items.slice(0, count);
		this.items = this.items.slice(count);
		for (const item of selected) {
			if (item.kind === "managed") {
				item.phase = "selected";
			}
		}
		return selected;
	}

	remove(target: ManagedPendingMessage): boolean {
		const index = this.items.indexOf(target);
		if (index === -1) {
			return false;
		}
		this.items.splice(index, 1);
		return true;
	}

	clear(): ManagedPendingMessage[] {
		const managed = this.items.filter((item): item is ManagedPendingMessage => item.kind === "managed");
		this.items = [];
		return managed;
	}
}

type ActiveRun = {
	promise: Promise<void>;
	resolve: () => void;
	abortController: AbortController;
};

/**
 * Stateful wrapper around the low-level agent loop.
 *
 * `Agent` owns the current transcript, emits lifecycle events, executes tools,
 * and exposes queueing APIs for steering and follow-up messages.
 */
export class Agent {
	private _state: MutableAgentState;
	private readonly listeners = new Set<(event: AgentEvent, signal: AbortSignal) => Promise<void> | void>();
	private readonly steeringQueue: PendingMessageQueue;
	private readonly followUpQueue: PendingMessageQueue;
	private readonly managedQueueItems = new Map<string, ManagedPendingMessage>();
	private readonly consumedManagedQueueItems = new Map<string, ManagedQueueTicket>();
	private managedQueueMirrorRevision = 0;
	private managedQueueFailure: Error | undefined;

	public convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	public transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	public streamFunction: StreamFn;
	public getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	public onPayload?: SimpleStreamOptions["onPayload"];
	public onResponse?: SimpleStreamOptions["onResponse"];
	public beforeToolCall?: (
		context: BeforeToolCallContext,
		signal?: AbortSignal,
	) => Promise<BeforeToolCallResult | undefined>;
	public afterToolCall?: (
		context: AfterToolCallContext,
		signal?: AbortSignal,
	) => Promise<AfterToolCallResult | undefined>;
	public shouldStopAfterTurn?: (
		context: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	) => boolean | Promise<boolean>;
	public prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public managedQueueMaterializationHook?: ManagedQueueMaterializationHook;
	private activeRun?: ActiveRun;
	private activeRunHasStarted = false;
	private activeRunTurnOpen = false;
	/** Session identifier forwarded to providers for cache-aware backends. */
	public sessionId?: string;
	/** Optional per-level thinking token budgets forwarded to the stream function. */
	public thinkingBudgets?: ThinkingBudgets;
	/** Preferred transport forwarded to the stream function. */
	public transport: Transport;
	/** Optional cap for provider-requested retry delays. */
	public maxRetryDelayMs?: number;
	/** Tool execution strategy for assistant messages that contain multiple tool calls. */
	public toolExecution: ToolExecutionMode;

	constructor(options: AgentOptions) {
		// Older compiled consumers may omit options or streamFn even though the current API requires them.
		const runtimeOptions: Partial<AgentOptions> = options ?? {};
		this._state = createMutableAgentState(runtimeOptions.initialState);
		this.convertToLlm = runtimeOptions.convertToLlm ?? defaultConvertToLlm;
		this.transformContext = runtimeOptions.transformContext;
		this.streamFunction = runtimeOptions.streamFn ?? getDefaultStreamFn();
		this.getApiKey = runtimeOptions.getApiKey;
		this.onPayload = runtimeOptions.onPayload;
		this.onResponse = runtimeOptions.onResponse;
		this.beforeToolCall = runtimeOptions.beforeToolCall;
		this.afterToolCall = runtimeOptions.afterToolCall;
		this.shouldStopAfterTurn = runtimeOptions.shouldStopAfterTurn;
		this.prepareNextTurn = runtimeOptions.prepareNextTurn;
		this.prepareNextTurnWithContext = runtimeOptions.prepareNextTurnWithContext;
		this.managedQueueMaterializationHook = runtimeOptions.managedQueueMaterializationHook;
		this.steeringQueue = new PendingMessageQueue(runtimeOptions.steeringMode ?? "one-at-a-time");
		this.followUpQueue = new PendingMessageQueue(runtimeOptions.followUpMode ?? "one-at-a-time");
		this.sessionId = runtimeOptions.sessionId;
		this.thinkingBudgets = runtimeOptions.thinkingBudgets;
		this.transport = runtimeOptions.transport ?? "auto";
		this.maxRetryDelayMs = runtimeOptions.maxRetryDelayMs;
		this.toolExecution = runtimeOptions.toolExecution ?? "parallel";
	}

	/**
	 * Subscribe to agent lifecycle events.
	 *
	 * Listener promises are awaited in subscription order and are included in
	 * the current run's settlement. Listeners also receive the active abort
	 * signal for the current run.
	 *
	 * `agent_end` is the final emitted event for a run, but the agent does not
	 * become idle until all awaited listeners for that event have settled.
	 */
	subscribe(listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	/**
	 * Current agent state.
	 *
	 * Assigning `state.tools` or `state.messages` copies the provided top-level array.
	 */
	get state(): AgentState {
		return this._state;
	}

	/** Controls how queued steering messages are drained. */
	set steeringMode(mode: QueueMode) {
		this.steeringQueue.mode = mode;
	}

	get steeringMode(): QueueMode {
		return this.steeringQueue.mode;
	}

	/** Controls how queued follow-up messages are drained. */
	set followUpMode(mode: QueueMode) {
		this.followUpQueue.mode = mode;
	}

	get followUpMode(): QueueMode {
		return this.followUpQueue.mode;
	}

	/** Queue a message to be injected after the current assistant turn finishes. */
	steer(message: AgentMessage): void {
		this.assertLegacyQueueAccess();
		this.steeringQueue.enqueue(message);
	}

	/** Queue a message to run only after the agent would otherwise stop. */
	followUp(message: AgentMessage): void {
		this.assertLegacyQueueAccess();
		this.followUpQueue.enqueue(message);
	}

	/** Provisionally stage a stable, non-drainable managed queue envelope. */
	stageManagedQueueItem(input: ManagedQueueItemInput): ManagedQueueStageResult {
		if (!this.managedQueueMaterializationHook) {
			throw new Error("Managed queue staging requires a managed queue materialization hook");
		}
		if (this.managedQueueFailure) {
			throw new Error(`Managed queue mirror is failed: ${this.managedQueueFailure.message}`);
		}
		if (input.itemId.trim().length === 0 || input.itemId !== input.itemId.trim()) {
			throw new Error("Managed queue item ID must be a non-empty canonical string");
		}

		const existing = this.managedQueueItems.get(input.itemId);
		if (existing) {
			return {
				type: "duplicate",
				ticket: this.ticketFor(existing),
				phase: existing.phase,
			};
		}
		if (this.consumedManagedQueueItems.has(input.itemId)) {
			throw new Error(`Managed queue item ${input.itemId} was already consumed`);
		}

		const item: ManagedPendingMessage = {
			kind: "managed",
			itemId: input.itemId,
			lane: input.lane,
			message: input.message,
			mirrorRevision: ++this.managedQueueMirrorRevision,
			phase: "staged",
		};
		this.managedQueueItems.set(item.itemId, item);
		this.queueFor(item.lane).stage(item);
		return { type: "staged", ticket: this.ticketFor(item) };
	}

	/** Mark a provisional managed item as durably admitted, without making it drainable. */
	admitManagedQueueItem(ticket: ManagedQueueTicket): ManagedQueueAdmissionResult {
		const item = this.resolveTicket(ticket);
		if (!item) {
			return this.isConsumedTicket(ticket) ? "already_admitted" : "stale";
		}
		if (item.phase === "staged") {
			item.phase = "admitted";
			return "admitted";
		}
		return "already_admitted";
	}

	/** Publish a durably admitted managed item so the next matching drain may select it. */
	publishManagedQueueItem(ticket: ManagedQueueTicket): ManagedQueuePublishResult {
		const item = this.resolveTicket(ticket);
		if (!item) {
			return this.isConsumedTicket(ticket) ? "already_published" : "stale";
		}
		if (item.phase === "staged") {
			throw new Error(`Managed queue item ${item.itemId} cannot publish before durable admission`);
		}
		if (item.phase === "admitted") {
			item.phase = "published";
			return "published";
		}
		return "already_published";
	}

	/** Abort a staged/admitted ticket. Published or selected items cannot be rolled back by admission abort. */
	abortManagedQueueItem(ticket: ManagedQueueTicket): ManagedQueueAbortResult {
		const item = this.resolveTicket(ticket);
		if (!item) {
			return this.isConsumedTicket(ticket) ? "already_consumed" : "stale";
		}
		if (item.phase === "published") {
			return "already_published";
		}
		if (item.phase === "selected" || item.phase === "failed") {
			return "already_selected";
		}
		this.removePendingManagedItem(item);
		return "removed";
	}

	/** Remove a managed item unless synchronous dequeue selection has already won. */
	removeManagedQueueItem(itemId: string): ManagedQueueRemovalResult {
		const item = this.managedQueueItems.get(itemId);
		if (!item) {
			return this.consumedManagedQueueItems.has(itemId) ? "already_consumed" : "not_found";
		}
		if (item.phase === "selected" || item.phase === "failed") {
			return "already_selected";
		}
		this.removePendingManagedItem(item);
		return "removed";
	}

	/** True while provisional, published, selected, or failed managed mirror evidence remains. */
	hasManagedQueueItems(): boolean {
		return this.managedQueueItems.size > 0;
	}

	/** True once an awaited materialization barrier has failed and closed this mirror generation. */
	hasManagedQueueFailure(): boolean {
		return this.managedQueueFailure !== undefined;
	}

	/** Snapshot stable managed queue evidence for close/reconciliation logic. */
	getManagedQueueMirrorSnapshot(): readonly ManagedQueueMirrorItemSnapshot[] {
		return [...this.managedQueueItems.values()].map((item) => ({
			itemId: item.itemId,
			lane: item.lane,
			mirrorRevision: item.mirrorRevision,
			phase: item.phase,
		}));
	}

	private assertLegacyQueueAccess(): void {
		if (this.managedQueueMaterializationHook) {
			throw new Error("Managed queue mode requires stable stage/admit/publish queue APIs");
		}
	}

	private queueFor(lane: ManagedQueueLane): PendingMessageQueue {
		return lane === "steer" ? this.steeringQueue : this.followUpQueue;
	}

	private ticketFor(item: ManagedPendingMessage): ManagedQueueTicket {
		return {
			itemId: item.itemId,
			lane: item.lane,
			mirrorRevision: item.mirrorRevision,
		};
	}

	private resolveTicket(ticket: ManagedQueueTicket): ManagedPendingMessage | undefined {
		const item = this.managedQueueItems.get(ticket.itemId);
		if (!item || item.lane !== ticket.lane || item.mirrorRevision !== ticket.mirrorRevision) {
			return undefined;
		}
		return item;
	}

	private isConsumedTicket(ticket: ManagedQueueTicket): boolean {
		const consumed = this.consumedManagedQueueItems.get(ticket.itemId);
		return consumed?.lane === ticket.lane && consumed.mirrorRevision === ticket.mirrorRevision;
	}

	private removePendingManagedItem(item: ManagedPendingMessage): void {
		if (!this.queueFor(item.lane).remove(item)) {
			throw new Error(`Managed queue item ${item.itemId} is not pending in its declared lane`);
		}
		this.managedQueueItems.delete(item.itemId);
	}

	private forgetClearedManagedItems(items: readonly ManagedPendingMessage[]): void {
		for (const item of items) {
			this.managedQueueItems.delete(item.itemId);
		}
	}

	/** Remove all queued steering messages. */
	clearSteeringQueue(): void {
		this.assertLegacyQueueAccess();
		this.forgetClearedManagedItems(this.steeringQueue.clear());
	}

	/** Remove all queued follow-up messages. */
	clearFollowUpQueue(): void {
		this.assertLegacyQueueAccess();
		this.forgetClearedManagedItems(this.followUpQueue.clear());
	}

	/** Remove all queued steering and follow-up messages. */
	clearAllQueues(): void {
		this.clearSteeringQueue();
		this.clearFollowUpQueue();
	}

	/** Returns true when either queue has a published item eligible for its next drain. */
	hasQueuedMessages(): boolean {
		return this.steeringQueue.hasDrainableItems() || this.followUpQueue.hasDrainableItems();
	}

	/** Active abort signal for the current run, if any. */
	get signal(): AbortSignal | undefined {
		return this.activeRun?.abortController.signal;
	}

	/** Abort the current run, if one is active. */
	abort(): void {
		this.activeRun?.abortController.abort();
	}

	/**
	 * Resolve when the current run and all awaited event listeners have finished.
	 *
	 * This resolves after `agent_end` listeners settle.
	 */
	waitForIdle(): Promise<void> {
		return this.activeRun?.promise ?? Promise.resolve();
	}

	/** Clear transcript state, runtime state, and queued messages. */
	reset(): void {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before resetting.");
		}

		this._state.messages = [];
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this._state.errorMessage = undefined;
		this.forgetClearedManagedItems(this.followUpQueue.clear());
		this.forgetClearedManagedItems(this.steeringQueue.clear());
		this.managedQueueItems.clear();
		this.consumedManagedQueueItems.clear();
		this.managedQueueFailure = undefined;
	}

	/** Start a new prompt from text, a single message, or a batch of messages. */
	async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
	async prompt(input: string, images?: ImageContent[]): Promise<void>;
	async prompt(input: string | AgentMessage | AgentMessage[], images?: ImageContent[]): Promise<void> {
		if (this.activeRun) {
			throw new Error(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
			);
		}
		const messages = this.normalizePromptInput(input, images);
		await this.runPromptMessages(messages);
	}

	/** Continue from the current transcript. The last message must be a user or tool-result message. */
	async continue(): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before continuing.");
		}

		const lastMessage = this._state.messages[this._state.messages.length - 1];
		if (!lastMessage) {
			throw new Error("No messages to continue from");
		}

		if (lastMessage.role === "assistant") {
			if (!this.steeringQueue.hasDrainableItems() && !this.followUpQueue.hasDrainableItems()) {
				throw new Error("Cannot continue from message role: assistant");
			}
			await this.runWithLifecycle(async (signal) => {
				const queuedSteering = await this.drainQueue(this.steeringQueue, signal);
				if (queuedSteering.length > 0) {
					await this.executePromptMessages(queuedSteering, signal, { skipInitialSteeringPoll: true });
					return;
				}

				const queuedFollowUps = await this.drainQueue(this.followUpQueue, signal);
				if (queuedFollowUps.length > 0) {
					await this.executePromptMessages(queuedFollowUps, signal);
				}
			});
			return;
		}

		await this.runContinuation();
	}

	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		if (Array.isArray(input)) {
			return input;
		}

		if (typeof input !== "string") {
			return [input];
		}

		const content: Array<TextContent | ImageContent> = [{ type: "text", text: input }];
		if (images && images.length > 0) {
			content.push(...images);
		}
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	private async runPromptMessages(
		messages: AgentMessage[],
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await this.runWithLifecycle(async (signal) => await this.executePromptMessages(messages, signal, options));
	}

	private async executePromptMessages(
		messages: AgentMessage[],
		signal: AbortSignal,
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await runAgentLoop(
			messages,
			this.createContextSnapshot(),
			this.createLoopConfig(signal, options),
			(event) => this.processEvents(event),
			signal,
			this.streamFunction,
		);
	}

	private async runContinuation(): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoopContinue(
				this.createContextSnapshot(),
				this.createLoopConfig(signal),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	private createContextSnapshot(): AgentContext {
		return {
			systemPrompt: this._state.systemPrompt,
			messages: this._state.messages.slice(),
			tools: this._state.tools.slice(),
		};
	}

	private createLoopConfig(signal: AbortSignal, options: { skipInitialSteeringPoll?: boolean } = {}): AgentLoopConfig {
		let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
		const shouldStopAfterTurn = this.shouldStopAfterTurn;
		return {
			model: this._state.model,
			reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel,
			sessionId: this.sessionId,
			onPayload: this.onPayload,
			onResponse: this.onResponse,
			transport: this.transport,
			thinkingBudgets: this.thinkingBudgets,
			maxRetryDelayMs: this.maxRetryDelayMs,
			toolExecution: this.toolExecution,
			beforeToolCall: this.beforeToolCall,
			afterToolCall: this.afterToolCall,
			shouldStopAfterTurn: shouldStopAfterTurn
				? async (context) => await shouldStopAfterTurn(context, this.signal)
				: undefined,
			prepareNextTurn:
				this.prepareNextTurnWithContext || this.prepareNextTurn
					? async (context) => {
							if (this.prepareNextTurnWithContext) {
								return await this.prepareNextTurnWithContext(context, this.signal);
							}
							return await this.prepareNextTurn?.(this.signal);
						}
					: undefined,
			convertToLlm: this.convertToLlm,
			transformContext: this.transformContext,
			getApiKey: this.getApiKey,
			getSteeringMessages: async () => {
				if (skipInitialSteeringPoll) {
					skipInitialSteeringPoll = false;
					return [];
				}
				return this.drainQueue(this.steeringQueue, signal);
			},
			getFollowUpMessages: async () => this.drainQueue(this.followUpQueue, signal),
		};
	}

	private async drainQueue(queue: PendingMessageQueue, signal: AbortSignal): Promise<AgentMessage[]> {
		const selected = queue.select();
		if (selected.length === 0) {
			return [];
		}

		const managed = selected.filter((item): item is ManagedPendingMessage => item.kind === "managed");
		if (managed.length === 0) {
			return selected.map((item) => item.message);
		}

		const hook = this.managedQueueMaterializationHook;
		if (!hook) {
			throw new Error("Managed queue item selected without a materialization hook");
		}

		let decisions: ReadonlyMap<string, ManagedQueueMaterializationDecision>;
		try {
			const result = await hook(
				managed.map((item) => ({ itemId: item.itemId, lane: item.lane, message: item.message })),
				signal,
			);
			decisions = this.validateMaterializationDecisions(managed, result);
		} catch (error) {
			const failure = error instanceof Error ? error : new Error(String(error));
			for (const item of managed) {
				item.phase = "failed";
			}
			this.managedQueueFailure = failure;
			throw failure;
		}

		const messages: AgentMessage[] = [];
		for (const item of selected) {
			if (item.kind === "legacy") {
				messages.push(item.message);
				continue;
			}

			const decision = decisions.get(item.itemId);
			if (!decision) {
				throw new Error(`Missing validated materialization decision for ${item.itemId}`);
			}
			this.managedQueueItems.delete(item.itemId);
			if (decision.type === "materialized") {
				this.consumedManagedQueueItems.set(item.itemId, this.ticketFor(item));
				messages.push(decision.message);
			}
		}
		return messages;
	}

	private validateMaterializationDecisions(
		selected: readonly ManagedPendingMessage[],
		decisions: readonly ManagedQueueMaterializationDecision[],
	): ReadonlyMap<string, ManagedQueueMaterializationDecision> {
		if (decisions.length !== selected.length) {
			throw new Error(
				`Managed queue materialization returned ${decisions.length} decisions for ${selected.length} selected items`,
			);
		}

		const selectedIds = new Set(selected.map((item) => item.itemId));
		const byItemId = new Map<string, ManagedQueueMaterializationDecision>();
		for (const decision of decisions) {
			if (!selectedIds.has(decision.itemId)) {
				throw new Error(`Managed queue materialization returned unknown item ${decision.itemId}`);
			}
			if (byItemId.has(decision.itemId)) {
				throw new Error(`Managed queue materialization returned duplicate item ${decision.itemId}`);
			}
			byItemId.set(decision.itemId, decision);
		}
		return byItemId;
	}

	private async runWithLifecycle(executor: (signal: AbortSignal) => Promise<void>): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing.");
		}

		const abortController = new AbortController();
		let resolvePromise = () => {};
		const promise = new Promise<void>((resolve) => {
			resolvePromise = resolve;
		});
		this.activeRun = { promise, resolve: resolvePromise, abortController };
		this.activeRunHasStarted = false;
		this.activeRunTurnOpen = false;

		this._state.isStreaming = true;
		this._state.streamingMessage = undefined;
		this._state.errorMessage = undefined;

		try {
			await executor(abortController.signal);
		} catch (error) {
			await this.handleRunFailure(error, abortController.signal.aborted);
		} finally {
			this.finishRun();
		}
	}

	private async handleRunFailure(error: unknown, aborted: boolean): Promise<void> {
		if (!this.activeRunHasStarted) {
			await this.processEvents({ type: "agent_start" });
		}
		if (!this.activeRunTurnOpen) {
			await this.processEvents({ type: "turn_start" });
		}
		const failureMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? "aborted" : "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		} satisfies AgentMessage;
		await this.processEvents({ type: "message_start", message: failureMessage });
		await this.processEvents({ type: "message_end", message: failureMessage });
		await this.processEvents({ type: "turn_end", message: failureMessage, toolResults: [] });
		await this.processEvents({ type: "agent_end", messages: [failureMessage] });
	}

	private finishRun(): void {
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this.activeRun?.resolve();
		this.activeRun = undefined;
		this.activeRunHasStarted = false;
		this.activeRunTurnOpen = false;
	}

	/**
	 * Reduce internal state for a loop event, then await listeners.
	 *
	 * `agent_end` only means no further loop events will be emitted. The run is
	 * considered idle later, after all awaited listeners for `agent_end` finish
	 * and `finishRun()` clears runtime-owned state.
	 */
	private async processEvents(event: AgentEvent): Promise<void> {
		switch (event.type) {
			case "agent_start":
				this.activeRunHasStarted = true;
				break;

			case "turn_start":
				this.activeRunTurnOpen = true;
				break;

			case "message_start":
				this._state.streamingMessage = event.message;
				break;

			case "message_update":
				this._state.streamingMessage = event.message;
				break;

			case "message_end":
				this._state.streamingMessage = undefined;
				this._state.messages.push(event.message);
				break;

			case "tool_execution_start": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.add(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "tool_execution_end": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.delete(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "turn_end":
				this.activeRunTurnOpen = false;
				if (event.message.role === "assistant" && event.message.errorMessage) {
					this._state.errorMessage = event.message.errorMessage;
				}
				break;

			case "agent_end":
				this._state.streamingMessage = undefined;
				break;
		}

		const signal = this.activeRun?.abortController.signal;
		if (!signal) {
			throw new Error("Agent listener invoked outside active run");
		}
		for (const listener of this.listeners) {
			await listener(event, signal);
		}
	}
}
