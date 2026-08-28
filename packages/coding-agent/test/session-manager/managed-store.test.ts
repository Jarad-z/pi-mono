import { describe, expect, it } from "vitest";
import {
	type ManagedSessionEntryAppendRequest,
	type ManagedSessionEntryReservationRequest,
	type ManagedSessionLeafCasRequest,
	type ManagedSessionStore,
	type ManagedSessionStoreSnapshot,
	type SessionEntry,
	type SessionHeader,
	SessionManager,
} from "../../src/core/session-manager.ts";
import { assistantMsg, userMsg } from "../utilities.ts";

function deferred(): { promise: Promise<void>; resolve: () => void; reject: (error: Error) => void } {
	let resolve = (): void => {};
	let reject = (_error: Error): void => {};
	const promise = new Promise<void>((settle, fail) => {
		resolve = settle;
		reject = fail;
	});
	return { promise, resolve, reject };
}

class TestManagedSessionStore implements ManagedSessionStore {
	readonly header: SessionHeader;
	readonly entries: SessionEntry[];
	readonly reserved = new Set<string>();
	leafId: string | null;
	nextId = 1;
	beforeAppend: (() => Promise<void>) | undefined;

	constructor(snapshot?: Partial<ManagedSessionStoreSnapshot>) {
		this.header = snapshot?.header ?? {
			type: "session",
			version: 3,
			id: "managed_session_1",
			timestamp: "2026-08-28T00:00:00.000Z",
			cwd: "D:/workspace",
		};
		this.entries = [...(snapshot?.entries ?? [])];
		this.leafId = snapshot?.leafId ?? null;
	}

	async loadSnapshot(): Promise<ManagedSessionStoreSnapshot> {
		return { header: this.header, entries: [...this.entries], leafId: this.leafId };
	}

	async reserveEntry(request: ManagedSessionEntryReservationRequest): Promise<{ entryId: string }> {
		expect(request.sessionId).toBe(this.header.id);
		const entryId = `host_entry_${this.nextId++}`;
		this.reserved.add(entryId);
		return { entryId };
	}

	async appendEntry(request: ManagedSessionEntryAppendRequest): Promise<SessionEntry> {
		await this.beforeAppend?.();
		if (request.sessionId !== this.header.id || request.expectedLeafId !== this.leafId) {
			throw new Error("store head CAS failed");
		}
		const existing = this.entries.find((entry) => entry.id === request.entry.id);
		if (existing) {
			if (JSON.stringify(existing) !== JSON.stringify(request.entry)) throw new Error("entry replay conflict");
			return existing;
		}
		if (!this.reserved.delete(request.entry.id)) {
			throw new Error(`entry ${request.entry.id} was not reserved`);
		}
		if (request.entry.parentId !== null && !this.entries.some((entry) => entry.id === request.entry.parentId)) {
			throw new Error("entry parent is missing");
		}
		this.entries.push(request.entry);
		this.leafId = request.entry.id;
		return request.entry;
	}

	async compareAndSetLeaf(request: ManagedSessionLeafCasRequest): Promise<void> {
		if (request.sessionId !== this.header.id || request.expectedLeafId !== this.leafId) {
			throw new Error("store head CAS failed");
		}
		if (request.nextLeafId !== null && !this.entries.some((entry) => entry.id === request.nextLeafId)) {
			throw new Error("next leaf is missing");
		}
		this.leafId = request.nextLeafId;
	}

	externallyReserve(entryId: string): void {
		this.reserved.add(entryId);
	}
}

describe("SessionManager managed store", () => {
	it("uses host-reserved identities and never opens a parallel JSONL store", async () => {
		const store = new TestManagedSessionStore();
		store.externallyReserve("reserved_input_entry");
		const session = await SessionManager.managed(store);

		expect(session.isManaged()).toBe(true);
		expect(session.isPersisted()).toBe(false);
		expect(session.getSessionFile()).toBeUndefined();
		await expect(session.appendManagedMessage(userMsg("hello"), "reserved_input_entry")).resolves.toBe(
			"reserved_input_entry",
		);
		await expect(session.appendManagedMessage(assistantMsg("answer"))).resolves.toBe("host_entry_1");

		expect(session.getBranch().map((entry) => [entry.id, entry.parentId])).toEqual([
			["reserved_input_entry", null],
			["host_entry_1", "reserved_input_entry"],
		]);
		expect(store.entries).toEqual(session.getEntries());
		expect(() => session.appendMessage(userMsg("legacy bypass"))).toThrow(
			"Managed SessionStore requires awaited synchronous append",
		);
		expect(store.entries).toHaveLength(2);
	});

	it("does not advance the projection until the durable append barrier commits", async () => {
		const store = new TestManagedSessionStore();
		const barrier = deferred();
		store.beforeAppend = () => barrier.promise;
		const session = await SessionManager.managed(store);

		const append = session.appendManagedMessage(userMsg("held"));
		await Promise.resolve();
		expect(session.getEntries()).toEqual([]);
		expect(session.getLeafId()).toBeNull();

		barrier.resolve();
		await append;
		expect(session.getLeafId()).toBe("host_entry_1");
		expect(store.leafId).toBe("host_entry_1");
	});

	it("leaves local state unchanged when the durable append rejects", async () => {
		const store = new TestManagedSessionStore();
		const barrier = deferred();
		store.beforeAppend = () => barrier.promise;
		const session = await SessionManager.managed(store);
		const append = session.appendManagedMessage(userMsg("failed"));
		await Promise.resolve();

		barrier.reject(new Error("sqlite unavailable"));
		await expect(append).rejects.toThrow("sqlite unavailable");
		expect(session.getEntries()).toEqual([]);
		expect(session.getLeafId()).toBeNull();
	});

	it("uses awaited branch-head CAS and supports atomic branch summary append", async () => {
		const root: SessionEntry = {
			type: "message",
			id: "entry_root",
			parentId: null,
			timestamp: "2026-08-28T00:00:01.000Z",
			message: userMsg("root"),
		};
		const child: SessionEntry = {
			type: "message",
			id: "entry_child",
			parentId: "entry_root",
			timestamp: "2026-08-28T00:00:02.000Z",
			message: assistantMsg("child"),
		};
		const store = new TestManagedSessionStore({ entries: [root, child], leafId: "entry_child" });
		const session = await SessionManager.managed(store);

		await session.compareAndSetManagedLeaf("entry_child", "entry_root");
		expect(session.getLeafId()).toBe("entry_root");
		await session.appendManagedBranchSummary("entry_root", "abandoned work");

		expect(session.getLeafEntry()).toMatchObject({
			type: "branch_summary",
			id: "host_entry_1",
			parentId: "entry_root",
			summary: "abandoned work",
		});
		await expect(session.compareAndSetManagedLeaf("entry_child", null)).rejects.toThrow(
			"local leaf changed before CAS",
		);
	});

	it("rejects corrupt snapshots before exposing a managed projection", async () => {
		const orphan: SessionEntry = {
			type: "message",
			id: "entry_orphan",
			parentId: "entry_missing",
			timestamp: "2026-08-28T00:00:01.000Z",
			message: userMsg("orphan"),
		};
		const store = new TestManagedSessionStore({ entries: [orphan], leafId: "entry_orphan" });

		await expect(SessionManager.managed(store)).rejects.toThrow("has missing parent entry_missing");
	});
});
