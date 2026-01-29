use crate::graph::{AudioGraph, Constant, NodeIndex};
use crate::primitive::DelayNode;

use super::engine::NodeType;

// Re-export AffineNode from graph module
pub use crate::graph::AffineNode;

/// Represents an affine operation for chain detection.
#[derive(Debug, Clone, Copy)]
enum AffineOp {
    /// Affine transform: output = input * gain + offset
    Affine { gain: f32, offset: f32 },
}

impl AffineOp {
    /// Convert to an AffineNode.
    fn to_affine(self) -> AffineNode {
        match self {
            AffineOp::Affine { gain, offset } => AffineNode::new(gain, offset),
        }
    }

    /// Compose two affine operations.
    fn then(self, other: Self) -> Self {
        let AffineOp::Affine {
            gain: g1,
            offset: o1,
        } = self;
        let AffineOp::Affine {
            gain: g2,
            offset: o2,
        } = other;
        // (x * g1 + o1) * g2 + o2 = x * (g1*g2) + (o1*g2 + o2)
        AffineOp::Affine {
            gain: g1 * g2,
            offset: o1 * g2 + o2,
        }
    }
}

/// Try to interpret a node as an affine operation.
fn node_as_affine(graph: &AudioGraph, idx: NodeIndex) -> Option<AffineOp> {
    let node_type = graph.node_type(idx)?;
    match node_type {
        NodeType::Affine => {
            // AffineNode stores gain at param 0, offset at param 1
            let gain = graph.node_param_value(idx, 0).unwrap_or(1.0);
            let offset = graph.node_param_value(idx, 1).unwrap_or(0.0);
            Some(AffineOp::Affine { gain, offset })
        }
        _ => None,
    }
}

/// Fuse chains of affine operations (Gain, Offset, PassThrough) into single AffineNode.
///
/// This optimization pass finds linear chains of affine nodes and replaces them
/// with a single fused node, reducing node count and eliminating intermediate storage.
///
/// # Example
///
/// ```text
/// Input -> Gain(0.5) -> Offset(1.0) -> Gain(2.0) -> Output
/// ```
/// Becomes:
/// ```text
/// Input -> AffineNode { gain: 1.0, offset: 2.0 } -> Output
/// // Because: ((x * 0.5) + 1.0) * 2.0 = x * 1.0 + 2.0
/// ```
pub fn fuse_affine_chains(graph: &mut AudioGraph) -> usize {
    let mut fused_count = 0;

    loop {
        // Find a chain to fuse
        let chain = find_affine_chain(graph);
        if chain.is_empty() || chain.len() < 2 {
            break;
        }

        // Compute the fused affine transform
        let mut combined = AffineNode::identity();
        for &idx in &chain {
            if let Some(op) = node_as_affine(graph, idx) {
                combined = combined.then(op.to_affine());
            }
        }

        // Skip if the result is identity (will be handled by identity elimination)
        if combined.is_identity() && chain.len() == 1 {
            break;
        }

        // Replace chain with fused node
        replace_affine_chain(graph, &chain, combined);
        fused_count += chain.len() - 1; // We reduced N nodes to 1
    }

    fused_count
}

/// Find a chain of affine nodes with single in/out connections.
fn find_affine_chain(graph: &AudioGraph) -> Vec<NodeIndex> {
    let node_count = graph.node_count();

    // Build adjacency info
    let mut in_degree = vec![0usize; node_count];
    let mut out_degree = vec![0usize; node_count];
    let mut successor = vec![None; node_count];
    let mut predecessor = vec![None; node_count];

    for wire in graph.audio_wires() {
        if wire.from < node_count && wire.to < node_count {
            out_degree[wire.from] += 1;
            in_degree[wire.to] += 1;
            successor[wire.from] = Some(wire.to);
            predecessor[wire.to] = Some(wire.from);
        }
    }

    // Find start of a chain (affine node with in_degree <= 1, followed by another affine)
    for start in 0..node_count {
        if node_as_affine(graph, start).is_none() {
            continue;
        }
        if out_degree[start] != 1 {
            continue;
        }

        let next = match successor[start] {
            Some(n) => n,
            None => continue,
        };

        if node_as_affine(graph, next).is_none() {
            continue;
        }

        // Found potential chain start, extend it
        let mut chain = vec![start];
        let mut current = next;

        while node_as_affine(graph, current).is_some()
            && in_degree[current] == 1
            && out_degree[current] <= 1
        {
            chain.push(current);
            if out_degree[current] == 0 {
                break;
            }
            current = match successor[current] {
                Some(n) => n,
                None => break,
            };
        }

        if chain.len() >= 2 {
            return chain;
        }
    }

    Vec::new()
}

/// Replace a chain of nodes with a single AffineNode.
fn replace_affine_chain(graph: &mut AudioGraph, chain: &[NodeIndex], affine: AffineNode) {
    if chain.is_empty() {
        return;
    }

    let first = chain[0];
    let last = chain[chain.len() - 1];

    // Add the new fused node
    let new_node = graph.add(affine);

    // Rewire inputs: anything that fed the first node should feed the new node
    let wires: Vec<_> = graph.audio_wires().to_vec();
    for wire in &wires {
        if wire.to == first && !chain.contains(&wire.from) {
            graph.connect(wire.from, new_node);
        }
    }

    // Rewire outputs: anything the last node fed should be fed by the new node
    for wire in &wires {
        if wire.from == last && !chain.contains(&wire.to) {
            graph.connect(new_node, wire.to);
        }
    }

    // Update input/output node references
    if graph.input_node() == Some(first) {
        graph.connect_input(new_node);
    }
    if graph.output_node() == Some(last) {
        graph.set_output(new_node);
    }

    // Remove chain nodes (in reverse order to preserve indices)
    let mut to_remove: Vec<NodeIndex> = chain.to_vec();
    to_remove.sort_by(|a, b| b.cmp(a));
    for idx in to_remove {
        graph.remove_node(idx);
    }
}

/// Remove identity nodes (Gain(1.0), Offset(0.0), PassThrough) from the graph.
///
/// These nodes don't change the signal and can be safely removed by rewiring
/// their inputs directly to their outputs.
pub fn eliminate_identities(graph: &mut AudioGraph) -> usize {
    let mut removed = 0;

    loop {
        let identity = find_identity_node(graph);
        if identity.is_none() {
            break;
        }
        let idx = identity.unwrap();

        // Rewire: connect predecessors directly to successors
        let wires: Vec<_> = graph.audio_wires().to_vec();
        let predecessors: Vec<NodeIndex> = wires
            .iter()
            .filter(|w| w.to == idx)
            .map(|w| w.from)
            .collect();
        let successors: Vec<NodeIndex> = wires
            .iter()
            .filter(|w| w.from == idx)
            .map(|w| w.to)
            .collect();

        // Connect each predecessor to each successor
        for &pred in &predecessors {
            for &succ in &successors {
                graph.connect(pred, succ);
            }
        }

        // Update input/output references
        if graph.input_node() == Some(idx) {
            if let Some(&succ) = successors.first() {
                graph.connect_input(succ);
            }
        }
        if graph.output_node() == Some(idx) {
            if let Some(&pred) = predecessors.first() {
                graph.set_output(pred);
            }
        }

        graph.remove_node(idx);
        removed += 1;
    }

    removed
}

/// Find a node that is an identity operation.
fn find_identity_node(graph: &AudioGraph) -> Option<NodeIndex> {
    for idx in 0..graph.node_count() {
        let node_type = graph.node_type(idx);
        let is_identity = match node_type {
            Some(NodeType::Affine) => {
                // Check if AffineNode is identity (gain ~= 1, offset ~= 0)
                let gain = graph.node_param_value(idx, 0).unwrap_or(1.0);
                let offset = graph.node_param_value(idx, 1).unwrap_or(0.0);
                (gain - 1.0).abs() < 1e-10 && offset.abs() < 1e-10
            }
            _ => false,
        };

        if is_identity {
            return Some(idx);
        }
    }
    None
}

/// Remove nodes that are not connected to the output.
///
/// A node is "live" if it's reachable from the output via audio wires,
/// or if it modulates a parameter of a live node.
///
/// # Example
///
/// ```text
/// Before:
///   A -> B -> Output
///   C -> D (disconnected)
///
/// After:
///   A -> B -> Output
///   (C and D removed)
/// ```
pub fn eliminate_dead_nodes(graph: &mut AudioGraph) -> usize {
    let output = match graph.output_node() {
        Some(out) => out,
        None => return 0, // No output, nothing to do
    };

    // Find all live nodes by walking backwards from output
    let mut live = vec![false; graph.node_count()];
    let mut worklist = vec![output];

    // Also mark input node as live if it exists
    if let Some(input) = graph.input_node() {
        worklist.push(input);
    }

    let audio_wires: Vec<_> = graph.audio_wires().to_vec();
    let param_wires: Vec<_> = graph.param_wires().to_vec();

    while let Some(idx) = worklist.pop() {
        if idx >= live.len() || live[idx] {
            continue;
        }
        live[idx] = true;

        // Add predecessors (nodes that feed into this one)
        for wire in &audio_wires {
            if wire.to == idx && !live[wire.from] {
                worklist.push(wire.from);
            }
        }

        // Add modulators (nodes that modulate this node's params)
        for wire in &param_wires {
            if wire.to == idx && !live[wire.from] {
                worklist.push(wire.from);
            }
        }
    }

    // Collect dead nodes (in reverse order to preserve indices during removal)
    let mut dead: Vec<NodeIndex> = (0..graph.node_count()).filter(|&i| !live[i]).collect();
    dead.sort_by(|a, b| b.cmp(a)); // Reverse order

    let removed = dead.len();
    for idx in dead {
        graph.remove_node(idx);
    }

    removed
}

/// Fold constant values through affine operations.
///
/// When a `Constant(a)` feeds into an `AffineNode { gain, offset }`,
/// the result is a known constant `a * gain + offset`, so we can replace
/// both nodes with a single `Constant`.
///
/// # Example
///
/// ```text
/// Before:
///   Constant(2.0) -> Gain(3.0) -> Offset(1.0) -> Output
///
/// After:
///   Constant(7.0) -> Output
///   // Because: 2.0 * 3.0 + 1.0 = 7.0
/// ```
pub fn fold_constants(graph: &mut AudioGraph) -> usize {
    let mut folded = 0;

    loop {
        let fold = find_constant_affine_pair(graph);
        if fold.is_none() {
            break;
        }
        let (const_idx, affine_idx, result_value) = fold.unwrap();

        // Replace the affine node with the folded constant
        let new_const = graph.add(crate::graph::Constant(result_value));

        // Rewire: anything the affine fed should now be fed by new constant
        let wires: Vec<_> = graph.audio_wires().to_vec();
        for wire in &wires {
            if wire.from == affine_idx {
                graph.connect(new_const, wire.to);
            }
        }

        // Update output if needed
        if graph.output_node() == Some(affine_idx) {
            graph.set_output(new_const);
        }

        // Remove both old nodes (affine first since it has higher index typically)
        let mut to_remove = vec![const_idx, affine_idx];
        to_remove.sort_by(|a, b| b.cmp(a)); // Reverse order
        for idx in to_remove {
            graph.remove_node(idx);
        }

        folded += 1;
    }

    folded
}

/// Find a Constant -> Affine pair that can be folded.
fn find_constant_affine_pair(graph: &AudioGraph) -> Option<(NodeIndex, NodeIndex, f32)> {
    let wires = graph.audio_wires();

    for wire in wires {
        // Check if source is a Constant
        let src_type = graph.node_type(wire.from);
        if src_type != Some(NodeType::Constant) {
            continue;
        }

        // Check if dest is an Affine
        let dst_type = graph.node_type(wire.to);
        if dst_type != Some(NodeType::Affine) {
            continue;
        }

        // Make sure the affine has only this one input (no other audio inputs)
        let input_count = wires.iter().filter(|w| w.to == wire.to).count();
        if input_count != 1 {
            continue;
        }

        // Make sure the constant has only this one output (not used elsewhere)
        let output_count = wires.iter().filter(|w| w.from == wire.from).count();
        if output_count != 1 {
            continue;
        }

        // Get the constant value and affine params
        let const_value = graph.node_param_value(wire.from, 0).unwrap_or(0.0);
        let gain = graph.node_param_value(wire.to, 0).unwrap_or(1.0);
        let offset = graph.node_param_value(wire.to, 1).unwrap_or(0.0);

        // Compute folded value
        let result = const_value * gain + offset;

        return Some((wire.from, wire.to, result));
    }

    None
}

/// Propagate constant values through chains of affine operations.
///
/// This is more aggressive than `fold_constants` - it tracks which nodes
/// have known constant outputs and propagates through longer chains.
///
/// # Example
///
/// ```text
/// Before:
///   Constant(1.0) -> Gain(2.0) -> Offset(3.0) -> Gain(4.0) -> Output
///
/// After:
///   Constant(20.0) -> Output
///   // Because: ((1.0 * 2.0) + 3.0) * 4.0 = 20.0
/// ```
pub fn propagate_constants(graph: &mut AudioGraph) -> usize {
    // This pass combines constant folding with affine chain fusion
    // Run both passes in a loop until no more changes
    let mut propagated = 0;

    loop {
        let folded = fold_constants(graph);
        let fused = fuse_affine_chains(graph);

        if folded == 0 && fused == 0 {
            break;
        }
        propagated += folded + fused;
    }

    propagated
}

/// Merge consecutive delay nodes with zero feedback.
///
/// When two `DelayNode` instances with `feedback == 0` are connected in series,
/// they can be merged into a single delay with combined time.
///
/// # Example
///
/// ```text
/// Before:
///   Input -> Delay(100 samples) -> Delay(50 samples) -> Output
///
/// After:
///   Input -> Delay(150 samples) -> Output
/// ```
///
/// # Limitations
///
/// - Only merges delays with zero feedback (feedback creates recurrence)
/// - Uses a conservative max buffer size (sum of both delay times + margin)
pub fn merge_delays(graph: &mut AudioGraph) -> usize {
    let mut merged = 0;

    loop {
        let pair = find_delay_pair(graph);
        if pair.is_none() {
            break;
        }
        let (first_idx, second_idx, combined_time) = pair.unwrap();

        // Create merged delay with buffer large enough for combined time
        let max_samples = (combined_time * 1.5) as usize + 100; // Add margin
        let mut new_delay = crate::primitive::DelayNode::new(max_samples);
        new_delay.set_time(combined_time);

        let new_node = graph.add(new_delay);

        // Rewire: inputs to first -> new node
        let wires: Vec<_> = graph.audio_wires().to_vec();
        for wire in &wires {
            if wire.to == first_idx && wire.from != second_idx {
                graph.connect(wire.from, new_node);
            }
        }

        // Rewire: outputs from second -> new node
        for wire in &wires {
            if wire.from == second_idx && wire.to != first_idx {
                graph.connect(new_node, wire.to);
            }
        }

        // Update input/output references
        if graph.input_node() == Some(first_idx) {
            graph.connect_input(new_node);
        }
        if graph.output_node() == Some(second_idx) {
            graph.set_output(new_node);
        }

        // Remove old nodes (in reverse order)
        let mut to_remove = vec![first_idx, second_idx];
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove {
            graph.remove_node(idx);
        }

        merged += 1;
    }

    merged
}

/// Find a pair of consecutive delays with zero feedback.
fn find_delay_pair(graph: &AudioGraph) -> Option<(NodeIndex, NodeIndex, f32)> {
    let wires = graph.audio_wires();

    for wire in wires {
        // Check if both nodes are delays
        if graph.node_type(wire.from) != Some(NodeType::Delay) {
            continue;
        }
        if graph.node_type(wire.to) != Some(NodeType::Delay) {
            continue;
        }

        // Check that first delay only outputs to second (single out-edge)
        let out_count = wires.iter().filter(|w| w.from == wire.from).count();
        if out_count != 1 {
            continue;
        }

        // Check that second delay only receives from first (single in-edge)
        let in_count = wires.iter().filter(|w| w.to == wire.to).count();
        if in_count != 1 {
            continue;
        }

        // Check feedback is zero for both
        // PARAM_FEEDBACK = 1 for DelayNode
        let feedback1 = graph.node_param_value(wire.from, 1).unwrap_or(0.0);
        let feedback2 = graph.node_param_value(wire.to, 1).unwrap_or(0.0);

        if feedback1.abs() > 1e-6 || feedback2.abs() > 1e-6 {
            continue;
        }

        // Get delay times (PARAM_TIME = 0)
        let time1 = graph.node_param_value(wire.from, 0).unwrap_or(0.0);
        let time2 = graph.node_param_value(wire.to, 0).unwrap_or(0.0);

        return Some((wire.from, wire.to, time1 + time2));
    }

    None
}

/// Run all graph optimization passes.
///
/// Returns the total number of nodes removed/fused.
pub fn run_optimization_passes(graph: &mut AudioGraph) -> usize {
    let mut total = 0;

    // Run passes until no more changes
    loop {
        let fused = fuse_affine_chains(graph);
        let folded = fold_constants(graph);
        let delays = merge_delays(graph);
        let identities = eliminate_identities(graph);
        let dead = eliminate_dead_nodes(graph);

        if fused == 0 && folded == 0 && delays == 0 && identities == 0 && dead == 0 {
            break;
        }
        total += fused + folded + delays + identities + dead;
    }

    total
}
