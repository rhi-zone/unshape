//! Pipeline serialization round-trip demo.
//!
//! Demonstrates building a mesh processing pipeline, serializing it to JSON,
//! deserializing it back, and executing it to get the same result.
//!
//! Run with: `cargo run --example pipeline_roundtrip --features dynop`

use unshape_mesh::{Cuboid, Mesh};
use unshape_op::{DynOp, OpRegistry, OpValue};

fn main() {
    println!("=== Pipeline Serialization Round-Trip ===\n");

    // Step 1: Create a mesh to process
    let cube = Cuboid::unit().apply();
    println!(
        "Original cube: {} vertices, {} triangles",
        cube.positions.len(),
        cube.indices.len() / 3
    );

    // Step 2: Build a pipeline of operations
    let smooth = unshape_mesh::Smooth::new(0.5, 3);
    let decimate = unshape_mesh::Decimate::target_ratio(0.5);
    let extrude = unshape_mesh::Extrude::new(0.2);

    println!("\nPipeline:");
    println!(
        "  1. Smooth (lambda: {}, iterations: {})",
        smooth.lambda, smooth.iterations
    );
    println!(
        "  2. Decimate (ratio: {:?})",
        decimate.target_ratio.unwrap_or(0.5)
    );
    println!("  3. Extrude (amount: {})", extrude.amount);

    // Step 3: Serialize each operation to JSON
    let ops_json: Vec<(String, serde_json::Value)> = vec![
        (smooth.type_name().to_string(), smooth.params()),
        (decimate.type_name().to_string(), decimate.params()),
        (extrude.type_name().to_string(), extrude.params()),
    ];

    println!("\nSerialized pipeline:");
    for (name, params) in &ops_json {
        println!("  {} -> {}", name, params);
    }

    // Step 4: Create registry and register mesh ops
    let mut registry = OpRegistry::new();
    unshape_mesh::register_ops(&mut registry);

    // Step 5: Deserialize and execute the pipeline
    println!("\nExecuting deserialized pipeline...");

    let mut current_mesh = cube.clone();
    for (type_name, params) in &ops_json {
        // Deserialize the operation
        let op = registry
            .deserialize(type_name, params.clone())
            .expect("Failed to deserialize op");

        // Execute it
        let input = OpValue::new(unshape_op::OpType::of::<Mesh>("Mesh"), current_mesh);
        let output = op.apply_dyn(input).expect("Failed to execute op");
        current_mesh = output.downcast::<Mesh>().expect("Wrong output type");

        println!(
            "  After {}: {} vertices, {} triangles",
            type_name.strip_prefix("resin::").unwrap_or(type_name),
            current_mesh.positions.len(),
            current_mesh.indices.len() / 3
        );
    }

    // Step 6: Compare with direct execution
    println!("\nDirect execution for comparison...");
    let direct_result = {
        let m1 = smooth.apply(&cube);
        let m2 = decimate.apply(&m1);
        extrude.apply(&m2)
    };
    println!(
        "  Direct result: {} vertices, {} triangles",
        direct_result.positions.len(),
        direct_result.indices.len() / 3
    );

    // Verify they match
    let match_status = if current_mesh.positions.len() == direct_result.positions.len()
        && current_mesh.indices.len() == direct_result.indices.len()
    {
        "Results match!"
    } else {
        "Results differ!"
    };
    println!("\n{}", match_status);

    // Bonus: Show the full pipeline as a single JSON document
    println!("\n=== Full Pipeline JSON ===\n");
    let pipeline_doc = serde_json::json!({
        "pipeline": ops_json.iter().map(|(name, params)| {
            serde_json::json!({
                "op": name,
                "params": params
            })
        }).collect::<Vec<_>>()
    });
    println!("{}", serde_json::to_string_pretty(&pipeline_doc).unwrap());
}
