#!/usr/bin/env bash
#
# Determinism guard — grep-based half.
#
# Nodes (`impl DynNode`) and ops (`#[derive(unshape_op::Op)]` / `impl DynOp`)
# MUST be deterministic. The function-call cases (SystemTime::now, Instant::now,
# rand::random / rand::rng / rand::thread_rng, env::var/vars) are enforced by
# clippy's `disallowed-methods` (see clippy.toml). This script enforces the
# cases clippy cannot see — macro invocations and language items — across the
# `src/` of the node/op crates:
#
#   - `thread_local!`   hidden per-thread mutable/ambient state
#   - `static mut`      hidden mutable global state
#   - `thread::spawn`   non-deterministic interleaving inside an op
#
# Scope: only the node/op crates listed in NODEOP_CRATES below, and only their
# `src/` trees (tests/, benches/, examples/, build.rs are out of scope — they
# are not node/op production code).
#
# Exemptions: a genuinely-deterministic use (e.g. an immutable const lookup
# table) may opt out by placing a marker comment on the SAME line or the line
# IMMEDIATELY ABOVE the offending line:
#
#     // determinism-guard: allow -- <reason>
#
# Keep exemptions rare and justified; the default answer is "don't".
#
# Run from the repo root:  tooling/check-determinism.sh
# Exits non-zero (and prints offenders) if any unexempted match is found.

set -euo pipefail

cd "$(dirname "$0")/.."

# Crates that define graph nodes / serializable ops. Keep in sync with the
# determinism invariant: a crate belongs here if it contains `impl DynNode`,
# `impl DynOp`, or `#[derive(unshape_op::Op)]`. Regenerate the candidate list:
#   grep -rl 'impl DynNode\|impl .*DynOp\|unshape_op::Op' crates/*/src \
#     | sed -E 's|/src/.*||' | sort -u
NODEOP_CRATES=(
  unshape-audio
  unshape-automata
  unshape-backend
  unshape-core
  unshape-crossdomain
  unshape-field
  unshape-fluid
  unshape-gpu
  unshape-history
  unshape-image
  unshape-lsystem
  unshape-mesh
  unshape-op
  unshape-particle
  unshape-physics
  unshape-pointcloud
  unshape-procgen
  unshape-rd
  unshape-rig
  unshape-scatter
  unshape-serde
  unshape-space-colonization
  unshape-spring
  unshape-vector
  unshape-voxel
)

# Patterns clippy cannot catch. Extended-regex, matched line-by-line.
PATTERN='thread_local!|static[[:space:]]+mut[[:space:]]|(std::)?thread::spawn'

EXEMPT='determinism-guard:[[:space:]]*allow'

violations=0

for crate in "${NODEOP_CRATES[@]}"; do
  src="crates/$crate/src"
  [ -d "$src" ] || continue

  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    file="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"

    # Same-line exemption?
    if printf '%s\n' "$hit" | grep -Eq "$EXEMPT"; then
      continue
    fi
    # Exemption on the immediately preceding line?
    if [ "$lineno" -gt 1 ]; then
      prev="$(sed -n "$((lineno - 1))p" "$file")"
      if printf '%s\n' "$prev" | grep -Eq "$EXEMPT"; then
        continue
      fi
    fi

    if [ "$violations" -eq 0 ]; then
      echo "Determinism guard: banned constructs found in node/op crates." >&2
      echo "Nodes/ops must be deterministic — no thread_local!, static mut, or thread::spawn." >&2
      echo "If a use is genuinely deterministic, annotate it with:" >&2
      echo "    // determinism-guard: allow -- <reason>" >&2
      echo >&2
    fi
    echo "  $hit" >&2
    violations=$((violations + 1))
  done < <(grep -rnE "$PATTERN" --include='*.rs' "$src" 2>/dev/null || true)
done

if [ "$violations" -gt 0 ]; then
  echo >&2
  echo "Determinism guard FAILED: $violations unexempted match(es)." >&2
  exit 1
fi

echo "Determinism guard: OK (no banned constructs in node/op crate src)."
