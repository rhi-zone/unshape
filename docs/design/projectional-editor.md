# Projectional Editor — Agreed Foundation (working)

This records **only** what was explicitly agreed about the editor's foundation. The agreed *interaction* principles and the open questions live in [editor-interaction.md](./editor-interaction.md).

An earlier version of this file elaborated a full projectional design — a fixed list of projections, reactive-lens / optics mechanics, model-level typed holes, live-preview-on-everything, a staged implementation path, and an MVP spec. Most of that was un-blessed synthesis written in a confident voice, and has been removed (see **Removed**).

## Agreed

- **There is no single correct representation.** The editor is *projectional*: one underlying model, surfaced through multiple co-equal projections, with no projection privileged as "the" view. This is the root of the "nothing has one mode / don't pick one modality" principles recorded in [editor-interaction.md](./editor-interaction.md).
- **Platform / stack decision:** Rust-native, **egui + wgpu**. Port dusklight's projection *design* — composable reactive lenses, "no read/write asymmetry" — rather than building on dusklight's web stack. One expression language across the ecosystem (currently `wick` on crates.io; a rename was explored and paused).

## Prior art referenced (references, not decisions)

- **dusklight** — projection mechanics (reactive lenses over composed optics; data-as-program).
- **Plasticity** — fluid gestural manipulation (preview-then-commit factory, gizmos, numeric entry during drag).
- The ecosystem **affordance docs** (interaction-graph, affordance-opacity / -surfaces / -types).
- The bar to beat: After Effects, Blender, DaVinci Resolve, Houdini, Pure Data.

## Not yet designed

The concrete editor — its surfaces, how primitives are selected, how modes are presented — is **not** designed. See [editor-interaction.md](./editor-interaction.md) for the agreed principles and the open questions (the carving, default-vs-correct, latent-vs-shown, editing-through-a-definition, stack-vs-graph).

## Removed (un-blessed synthesis — do not reintroduce without explicit approval)

The prior version's fixed projection list, the ReactiveLens / optics mechanics as specified, model-level typed holes + program synthesis, live-preview-on-everything, the affordance-typing scheme, the staged implementation path, and the image-domain MVP slice. These were one automated synthesis, not agreed. Some may be worth revisiting — but only as proposals, not as recorded design.
