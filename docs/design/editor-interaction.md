# Editor Interaction — Agreed Principles (working)

This document records **only** what has been explicitly agreed, the open questions, and factual findings. It deliberately does **not** propose or bless a synthesized editor design.

An earlier version of this file proposed a "Reading Stack + Command Twin" synthesis produced by an automated design pass. That was rejected as performative chrome-optimization of the op-stack paradigm, and has been removed. See **Rejected** at the bottom.

## UX values (the bar — not a scoring rubric)

The stated values for "good UX" here. They are values to design *by*, **not** a scoring function to rank candidate designs against — reifying them into a rubric was a category error.

- Simple mental model.
- Not 10,000 things to learn.
- Surface the most *relevant* affordances naturally.
- The number of affordances must not overwhelm the user.
- Escape hatches. *(Held tentatively — not fully settled.)*

The middle two are a pair in tension that must hold *everywhere*: surface the relevant few **and** never overwhelm. That forces a genuine relevance signal, not just a hard cap.

## Modality is plural, not chosen

- **Nothing has one mode.** A thing is not locked to a single representation or way of editing. The error is the *exclusivity* (the "one"), not where the mode is attached — "mode" is **not** a property owned by the node, the slot, or the editor.
- **You don't pick one modality.** For a given action, multiple modalities co-exist; you use whichever your input *evokes*, rather than committing to one up front. *(Illustrative only, not canon: flick → radial; type → list/filter; browse → preview; a key → shortcut. The point is the plurality and the evocation, not this particular set.)*
- **Keyboard shortcuts are just another co-present modality** — the most *direct* one, for actions you've internalized. They should be surfaced *on* the visible modalities (their binding shown inline), so you learn a shortcut through the modality you're already using and it graduates to eyes-free.

*Precedent (factual): VS Code exposes one command set through several co-present modalities — command palette (type), menus, context menus, and direct keybindings — and shows each command's keybinding inline in the palette and menus, which is exactly the multi-modal + shortcut-graduation pattern. It is also a cautionary example for relevance: its palette is fuzzy-match + light recency over a flat list of hundreds, with no real contextual ranking.*

## Direct manipulation is one modality, and often insufficient

- It is **one** modality among many — not the thesis.
- It is often *insufficient*: many operations don't fit it — broadcasting (apply to many), form conversions, modifiers, instancing.
- It is **approximate**. That is fine for some use cases and not good enough for others.
- Therefore **precision is a real axis the modalities span**: approximate (drag/gesture) and exact (type / formula / numeric) are co-equal, and the use case decides which is needed. Direct manipulation is never the sole mode.

## Open questions (unresolved — no answer is blessed)

- **"Everything has every mode" vs. not overwhelming.** If nothing has one mode, modes must be *latent, not all shown at once* — but how the present subset is chosen is unresolved.
- **Default vs. correct.** Anything needs a *default* representation, but no representation is "correct." How the default is chosen — and how to keep it cheap enough to leave that it carries no claim — is open.
- **The carving.** A person's primitives are cross-cutting concepts (noise, field, warp, repeat, blend, oscillator, filter…); the codebase is partitioned by implementation domain. How — or whether — to organize the editor around the human carving is open, and "relevance" cannot be computed without answering it.
- **Editing through a definition.** Editing an *existing* formula, and direct-manipulating a value that is formula-defined — where the edit must run *back through* the definition (inverse / bidirectional). Open.
- **Stack vs. graph vs. nested.** When a composition should read and behave as a linear stack, a branched graph, or nested groups. Open.

## Findings (facts about the codebase, not decisions)

- There are ~a few hundred primitives, **partitioned by implementation domain** (mesh / audio / image / vector / field / …). The op registry imposes no categories; grouping is by crate.
- The **one** genuine cross-domain abstraction in code is `Field<I, O>` (~107 impls across 6 crates). `SpatialTransform` and the noise generator→field layering partially unify; most else does not.
- Human-recurring concepts are **reimplemented per domain**: filter/convolve/smooth, blend/mix/composite, warp/deform, oscillator/LFO each exist as separate per-domain types. Some **names collide** — two unrelated `Scatter`s, two unrelated `Warp`s.
- So the human-concept ↔ codebase mismatch is real, and in places the naming is genuinely broken.

## Rejected (do not reintroduce without explicit approval)

- The "Reading Stack + Command Twin" synthesis, and the broader five-candidate editor design pass it came from — rejected as redecorating the op-stack paradigm rather than designing for the medium.
- Direct manipulation as *the* thesis.
- Unconfirmed framings raised in discussion but not agreed: "representation follows definition," the storage-vs-display split, "the user never sees modes."
