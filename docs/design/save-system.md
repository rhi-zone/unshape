# Save System — Never Lose Work

Motivation: creative tools lose hours of work (e.g. Krita: a ~15-minute autosave gap + non-atomic copy-over-original saves that fail at the copy step + a dismissible/temp-dir-dependent recovery dialog). The bar to beat/match: Reaper (gold standard), Krita (failure catalog), Clip Studio Paint (good auto-restore but opt-in), Paint.NET (no recovery — the floor).

## Substrate advantage

unshape is a deterministic node graph with op-granular history. An **append-only event log** therefore gives, nearly for free: continuous autosave (flush appended ops), exact crash recovery (replay from last checkpoint — deterministic ⇒ exact), and **persistent undo (the log *is* the undo history)**. The full project file is a periodic checkpoint/compaction of the log.

## Requirements

1. **Autosave/recovery ON by default, short cadence** (seconds, not Krita's 15 min — the gap *is* the lost work). With the event log this is just continuously flushing appended ops.
2. **Append-only journal as primary durability** — don't rely on periodically rewriting the whole file; append each op + fsync; project file = periodic checkpoint; recovery = replay from last checkpoint.
3. **Atomic writes** — write-temp → fsync(temp) → fsync(dir) → atomic rename; **never copy-over-original**. This is the one thing the substrate does NOT give for free, that reincarnate's persistence design omits, and that Krita fatally botches (copy-step failure → zero-byte / unwritable files). A crash must leave either the complete old file or the complete new file.
4. **Redundant fan-out** — tee writes to multiple independent backends (local + secondary); one failing write must not lose the others (cf. Reaper's timestamped backup in an additional directory; reincarnate's `tee()`).
5. **Persistent undo saved with the project** — free here (the log is the undo); cf. Reaper's `.RPP-UNDO`.
6. **Reversible deltas, not full snapshots** — op-as-value with an inverse (or recomputable via replay); cf. reincarnate's diff-moments (per-change `{old,new}`, live head kept whole).
7. **Recovery automatic and unmissable** — auto-restore the recovered session on launch (cf. CSP restore-on-reopen), recovered state clearly marked, pre-crash version still retrievable. Never a single dismissible yes/no that silently discards work (Krita), never opt-in (CSP) or absent (Paint.NET).
8. **No dependence on volatile/temp locations** — keep recovery data durable + project-adjacent, not OS temp (Krita's `%TEMP%`-clearing loss).
9. **Fault isolation for risky/live nodes** — a node crash must not take down the app or lose the unsaved graph (cf. Reaper's per-plugin process isolation); especially live-signal / external-code nodes.
10. **Safe under concurrent edits** — snapshot the log offset and keep editing; saving must never corrupt or block (Krita corrupts when painting during a save).

## Prior art (verified)

- **Reaper** — model to beat: persistent undo history with the project (`.RPP-UNDO`), recovery mode (open with FX offline), redundant timestamped backups to an additional directory, plugin process isolation.
- **reincarnate** (`docs/persistence.md`) — directly reusable internal design: `commit(state)` after each transition; save vs. history separated; reversible diff-moments; `tee()` fan-out to multiple backends; per-backend retention windows. *Gap:* does not specify atomic writes — add requirement #3.
- **Krita** — failure catalog: 15-min default gap, non-atomic temp-then-copy saves that fail at the copy, dismissible + `%TEMP%`-dependent recovery, autosave can silently stop.
- **CSP** — good auto-restore-on-reopen, but recovery is opt-in (a trap).
- **Paint.NET** — no autosave/recovery at all (below the floor).

## Bottom line

The deterministic-graph + op-granular append-only event log is the strongest possible foundation — continuous autosave, exact replay-recovery, and persistent undo are nearly free — **provided** atomic, fsync'd writes are added (the one piece the substrate and reincarnate's design don't guarantee).
