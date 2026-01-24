//! L-system procedural generation demo.
//!
//! Generates a fractal plant using an L-system and prints the result.
//!
//! Run with: `cargo run --example procgen_lsystem`

use unshape_lsystem::{LSystem, Rule, Turtle2D};

fn main() {
    println!("=== L-System Fractal Plant ===\n");

    // Classic fractal plant L-system
    let lsystem = LSystem::new("X")
        .with_rule(Rule::simple('X', "F+[[X]-X]-F[-FX]+X"))
        .with_rule(Rule::simple('F', "FF"));

    // Generate the system (show progress for each iteration)
    let iterations = 4;
    for i in 1..=iterations {
        let result = lsystem.generate(i);
        println!("Iteration {}: {} characters", i, result.len());
    }

    // Generate final result
    let result = lsystem.generate(iterations);

    // Interpret as 2D turtle graphics
    let turtle = Turtle2D {
        angle: 25.0,
        step: 5.0,
        ..Default::default()
    };
    let segments = turtle.apply(&result);

    println!("\nGenerated {} line segments", segments.len());

    // Find bounding box
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for seg in &segments {
        min_x = min_x.min(seg.start.x).min(seg.end.x);
        max_x = max_x.max(seg.start.x).max(seg.end.x);
        min_y = min_y.min(seg.start.y).min(seg.end.y);
        max_y = max_y.max(seg.start.y).max(seg.end.y);
    }

    println!(
        "Bounding box: ({:.1}, {:.1}) to ({:.1}, {:.1})",
        min_x, min_y, max_x, max_y
    );
    println!("Size: {:.1} x {:.1}", max_x - min_x, max_y - min_y);

    // Print a simple ASCII visualization (top-down view)
    println!("\n=== ASCII Preview (60x30) ===\n");

    let width = 60;
    let height = 30;
    let mut canvas = vec![vec![' '; width]; height];

    let scale_x = (width - 1) as f32 / (max_x - min_x).max(0.001);
    let scale_y = (height - 1) as f32 / (max_y - min_y).max(0.001);

    for seg in &segments {
        // Plot start and end points
        let x1 = ((seg.start.x - min_x) * scale_x) as usize;
        let y1 = ((seg.start.y - min_y) * scale_y) as usize;
        let x2 = ((seg.end.x - min_x) * scale_x) as usize;
        let y2 = ((seg.end.y - min_y) * scale_y) as usize;

        // Simple line drawing
        let steps = ((x2 as i32 - x1 as i32)
            .abs()
            .max((y2 as i32 - y1 as i32).abs())) as usize
            + 1;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = (x1 as f32 + (x2 as f32 - x1 as f32) * t) as usize;
            let y = (y1 as f32 + (y2 as f32 - y1 as f32) * t) as usize;
            if x < width && y < height {
                canvas[height - 1 - y][x] = '*';
            }
        }
    }

    for row in &canvas {
        println!("{}", row.iter().collect::<String>());
    }
}
