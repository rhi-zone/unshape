//! Benchmarks for field expression evaluation.
//!
//! Compares JIT-compiled vs interpreted evaluation.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use unshape_expr_field::FieldExpr;
use std::collections::HashMap;

const GRID_SIZE: usize = 100; // 100x100 = 10,000 evaluations

fn create_complex_expr() -> FieldExpr {
    // perlin(x * 4, y * 4) * 0.5 + 0.5
    FieldExpr::Add(
        Box::new(FieldExpr::Mul(
            Box::new(FieldExpr::Perlin2 {
                x: Box::new(FieldExpr::Mul(
                    Box::new(FieldExpr::X),
                    Box::new(FieldExpr::Constant(4.0)),
                )),
                y: Box::new(FieldExpr::Mul(
                    Box::new(FieldExpr::Y),
                    Box::new(FieldExpr::Constant(4.0)),
                )),
            }),
            Box::new(FieldExpr::Constant(0.5)),
        )),
        Box::new(FieldExpr::Constant(0.5)),
    )
}

fn create_trig_expr() -> FieldExpr {
    // sin(x * 3.14159) * cos(y * 3.14159)
    FieldExpr::Mul(
        Box::new(FieldExpr::Sin(Box::new(FieldExpr::Mul(
            Box::new(FieldExpr::X),
            Box::new(FieldExpr::Constant(std::f32::consts::PI)),
        )))),
        Box::new(FieldExpr::Cos(Box::new(FieldExpr::Mul(
            Box::new(FieldExpr::Y),
            Box::new(FieldExpr::Constant(std::f32::consts::PI)),
        )))),
    )
}

fn create_sdf_expr() -> FieldExpr {
    // sdf_circle(x, y, 0.5)
    FieldExpr::SdfCircle {
        x: Box::new(FieldExpr::X),
        y: Box::new(FieldExpr::Y),
        radius: 0.5,
    }
}

fn bench_interpreted(c: &mut Criterion) {
    let expr = create_complex_expr();
    let bindings: HashMap<String, f32> = HashMap::new();

    c.bench_function("interpreted_perlin_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32;
                    let y = j as f32 / GRID_SIZE as f32;
                    sum += expr.eval(x, y, 0.0, 0.0, &bindings);
                }
            }
            black_box(sum)
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_jit(c: &mut Criterion) {
    use unshape_expr_field::jit_impl::FieldExprCompiler;

    let expr = create_complex_expr();
    let mut compiler = FieldExprCompiler::new().unwrap();
    let compiled = compiler.compile(&expr).unwrap();

    c.bench_function("jit_perlin_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32;
                    let y = j as f32 / GRID_SIZE as f32;
                    sum += compiled.eval(x, y, 0.0, 0.0);
                }
            }
            black_box(sum)
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_jit_trig(c: &mut Criterion) {
    use unshape_expr_field::jit_impl::FieldExprCompiler;

    let expr = create_trig_expr();
    let bindings: HashMap<String, f32> = HashMap::new();
    let mut compiler = FieldExprCompiler::new().unwrap();
    let compiled = compiler.compile(&expr).unwrap();

    c.bench_function("interpreted_trig_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32;
                    let y = j as f32 / GRID_SIZE as f32;
                    sum += expr.eval(x, y, 0.0, 0.0, &bindings);
                }
            }
            black_box(sum)
        })
    });

    c.bench_function("jit_trig_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32;
                    let y = j as f32 / GRID_SIZE as f32;
                    sum += compiled.eval(x, y, 0.0, 0.0);
                }
            }
            black_box(sum)
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_jit_sdf(c: &mut Criterion) {
    use unshape_expr_field::jit_impl::FieldExprCompiler;

    let expr = create_sdf_expr();
    let bindings: HashMap<String, f32> = HashMap::new();
    let mut compiler = FieldExprCompiler::new().unwrap();
    let compiled = compiler.compile(&expr).unwrap();

    c.bench_function("interpreted_sdf_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32 - 0.5;
                    let y = j as f32 / GRID_SIZE as f32 - 0.5;
                    sum += expr.eval(x, y, 0.0, 0.0, &bindings);
                }
            }
            black_box(sum)
        })
    });

    c.bench_function("jit_sdf_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..GRID_SIZE {
                for j in 0..GRID_SIZE {
                    let x = i as f32 / GRID_SIZE as f32 - 0.5;
                    let y = j as f32 / GRID_SIZE as f32 - 0.5;
                    sum += compiled.eval(x, y, 0.0, 0.0);
                }
            }
            black_box(sum)
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_compile_time(c: &mut Criterion) {
    use unshape_expr_field::jit_impl::FieldExprCompiler;

    let expr = create_complex_expr();

    c.bench_function("compile_perlin_expr", |b| {
        b.iter(|| {
            let mut compiler = FieldExprCompiler::new().unwrap();
            black_box(compiler.compile(&expr).unwrap())
        })
    });
}

#[cfg(feature = "cranelift")]
criterion_group!(
    benches,
    bench_interpreted,
    bench_jit,
    bench_jit_trig,
    bench_jit_sdf,
    bench_compile_time
);

#[cfg(not(feature = "cranelift"))]
criterion_group!(benches, bench_interpreted);

criterion_main!(benches);
