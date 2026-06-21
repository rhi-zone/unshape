# Interaction Trajectories (synthesized design material)

These are **synthesized exploration material, not a blessed spec.** They were produced by clean-context agents (no session history) asked to (1) brainstorm diverse realistic creative tasks across media and (2) write the *ideal* step-by-step interaction trajectory for each — what the user intends at each step, what the tool should surface (the relevant few), what they do, what it produces. The point of the exercise (per the interaction-graph model: there is no canonical surface; you validate the design against clean trajectories through "what can I do now → do it → what can I do now"). Their value: they independently converged on the agreed interaction principles (see editor-interaction.md "Validation"), and surfaced the recurring needs below.

## Recurring interaction needs (the distilled output)

1. **Promote a constant to a signal/variation — in place.** "Vary this per brick," "drive this by envelope/audio/noise," "follow with delay." The single most recurring move; one gesture, graph forms behind you and stays openable.
2. **Result-as-parameter / parameter-as-result**, late-bound through named first-class signals so a source can be swapped without rewiring destinations.
3. **Per-connection remap** — one source → many targets, each edge with its own range/curve/retiming.
4. **Drag and type interchangeable on every value** — approximate (scrub by eye/ear) and exact (typed, unit-aware, grid/tempo-snappable) are the same field; precision where it matters, approximate where it doesn't.
5. **"Vary/apply per X" and "shared token"** — per-instance variation and one-value-across-many are duals; plus local override that doesn't sever the link.
6. **Compare-variants is a first-class verb** — audition N versions of a value/seed/palette/patch side by side, live, pick one, at any granularity.
7. **Context-derived "what can I do now"** — the surfaced actions come from selection + named features of generated things (branch tips, brick cells, audio bands); a small relevant set, never a flat global menu, never hunting menus away from the work.
8. **Edit-earlier-without-redoing-later** — history/generation is relational and re-flows: change branch count and leaves/sway/lag follow; change rock count and distribution rules persist.
9. **Continuous audition while editing** — edit during playback/preview; scrub and everything updates; no bake/render/play cycle.
10. **Gesture and authored data are one representation** — recorded live performance lands as the same editable curve you'd author; quantize/smooth/loop operate on it; hand-edits and recorded points coexist.
11. **Invariants are owned by the tool, not the user** — tiling, loop-seamlessness, rig re-solve, UV preservation under topology change are requested-and-maintained, not babysat.

(The full task list and the detailed per-task trajectories live in the session logs; this doc records the distilled recurring needs, which are the load-bearing output.)

## Source corpus: concrete tasks and trajectories

This section records the clean-context (airgapped) synthesized corpus from which the 11 distilled needs above were derived. It is **generated material, recorded for traceability — not blessed design.** Nothing here is a decided conclusion or recommendation; it is the validation corpus the distillation was run against.

### The 33 concrete creative tasks

**3D / rigging**

1. Swaying low-poly tree with leaf lag.
2. Modular sci-fi corridor kit.
3. Slider-driven creature proportions with auto-resolve rig.
4. Procedural rope/cable with sag.
5. Faceted gemstone generator.
6. Face expression dials.
7. *(Awkward today)* Topology-changing mesh (5→8 spokes) without breaking UVs/rig.
8. *(Ambitious)* Relational shared-edge-loop constraint between two meshes.

**2D vector**

9. Icon set with global stroke/corner tokens.
10. Lip-sync 2D puppet from audio.
11. "Breathing" logo loop.
12. Path-following border brush with clean corners.
13. *(Awkward today)* Master-palette driving all shapes, audition 5 palettes.
14. Parametric hand-drawn line (wobble/taper).

**Image / texture**

15. Tiling brick with per-brick variation.
16. Weathered-metal (curvature edge-wear + scratches).
17. Blue-noise stipple of a photo.
18. Seamless terrain heightmap (ridged + billow + erosion, 16-bit).
19. *(Awkward today)* Infinite-resolution texture evaluated only where camera looks.
20. LUT/gradient-map color grade with scrub-compare.

**Audio**

21. Pitch-bending kick.
22. 30s evolving pad.
23. Generative ambient bed.
24. Surface/weight-parameterized footsteps.
25. *(Awkward today)* Grab output waveform on screen and back-solve params.

**Motion graphics**

26. Spring stagger-in title card.
27. Guaranteed-seamless particle loop.
28. Data-driven animating bar chart.

**Cross-media**

29. Audio-reactive displaced mesh + color (bass→displacement).
30. Paint-on-texture deforms mesh live.
31. *(Ambitious)* One "intensity" signal driving synth cutoff + particle rate + light temp, retimed per destination.
32. 500-rock scatter (noise-field rotation, painted density, seed reshuffle).
33. *(Ambitious)* Record live parameter-wiggling as an editable curve track.

### The 8 detailed trajectories

Each records: intent → what the tool surfaces → action → result, preserving the "avoid"/"go back"/"reach for next" notes.

#### A — Tiling brick (task 15)

Empty-texture context surfaces only tiling-relevant generators {cells/Voronoi, stripes, grid/bond, noise}, not the whole library → pick running-bond → bond/size/mortar surface as currently-relevant params → drag mortar (approx) OR type exact (same field, two precisions) → on the color param, the key affordance is "vary this per-cell" (promote constant→per-instance signal seeded by cell id), same affordance reused on roughness → "compare variants" shows 4 live thumbnails, pick one. **Avoid:** manually wiring Voronoi-id→ramp→mix; "vary per brick" is one gesture that builds that graph behind you and stays openable.

#### B — Pitch-bending kick (task 21)

Empty-audio context offers archetypes (kick/snare/pad/pluck = pre-wired openable patches) not raw oscillators → pick kick, trigger → pitch/decay/click surface → on pitch, affordance "drive with an envelope" keyed to trigger → bend editable as draggable curve handles (by ear) or typed start/end Hz+ms (exact) → tempo-sync toggle turns 120ms into 1/32 note → audition continuously while editing, no stop/render/play → "expose macro" one knob "weight" scaling pitch-depth+decay+click (one source, many destinations).

#### C — Swaying tree with leaf lag (task 1)

Empty-mesh organic context → {L-system/branch, lathe, extrude-path} → branching growth gives parametric trunk (topology generated, so changing branch count later isn't a remodel — **task 7 made routine**) → leaves instanced at named feature "branch tips," count derives from branch count (relational) → "drive by signal" on trunk bend with sway-appropriate sources {looping noise, sine, wind field}; loop-seamlessness is requestable (**task 27 free**) → on leaves a "follow with delay/springiness" affordance (secondary motion as a relationship, not keyframes), tuned by one "looseness" slider or exact stiffness/damping → seed scrub + compare-variants; **go back** and re-edit branch count and leaves/sway/lag re-flow.

#### D — One "intensity" → synth + particles + light (task 31)

Create a named signal as a first-class object (constant now, anything later) → on synth cutoff "drive by signal" lists named signals first; bind with an inline per-connection remap (200→8000 Hz log) → same signal drives particle emission (linear) and light temp (own/inverted curve), each shaped locally → on the source, "replace source" {hand-drawn curve, audio feature, LFO, live input}; swap constant→bass-band energy, all three react with no downstream rewiring (the named-signal indirection makes the late swap free).

#### E — Audio-reactive displaced mesh (task 29)

Drop audio; derived signals surface as ready-made features (level, bass/mid/treble, onset/beat) — no FFT graph to build → on mesh "displace by" {texture, signal, noise}, bind amplitude to bass, base noise gives spatial shape while bass gives amount; the mapping (mm/unit bass) wants exact, the frequency split is fine by ear → treble→hue via LUT → drag the playhead updates everything live, no re-bake.

#### F — Record live wiggling as editable curve (task 33)

On any param/signal an arm/record affordance → wiggle live against playback → gesture captured as a curve track, immediately editable points (gesture and authored data are the same representation, no import/convert) → {quantize, smooth, loop, trim}; quantize peaks to grid → record a second take, blend modes (add/replace/max); hand-edits and recorded points coexist → promote the track to a reusable signal (**reach for next: back to D**).

#### G — Global tokens across icon set (task 9)

When many shapes share a value, offer "promote literal to shared token" → change the token once, all 40 update → on one instance, "override token locally" without breaking the link (39 tracking, 1 flagged/re-attachable) → "compare variants" at the token-set level (5 palettes side by side; **task 13 generalized**).

#### H — Scatter 500 rocks (task 32)

On a surface "scatter instances" {what, density} → per-instance attrs offer "drive by field" with the instance's position as natural sampling input (yaw from noise field) → density accepts a painted mask OR a slope rule as readily as a number (mixed approx+precise on one property: paint by hand + "exclude where slope > 40°") → seed scrub + compare-variants; **go back** to change count/asset, distribution rules persist; **reach for next:** reuse the tree's wind field (signals shareable project-wide).
