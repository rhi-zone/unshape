//! Basic field implementations: Constant, Coordinates, and FnField.

use std::marker::PhantomData;

use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

/// A constant field that always returns the same value.
#[derive(Debug, Clone, Copy)]
pub struct Constant<O> {
    /// The constant value to return.
    pub value: O,
}

impl<O> Constant<O> {
    /// Create a new constant field.
    pub fn new(value: O) -> Self {
        Self { value }
    }
}

impl<I, O: Clone> Field<I, O> for Constant<O> {
    fn sample(&self, _input: I, _ctx: &EvalContext) -> O {
        self.value.clone()
    }
}

/// A field that returns the input coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct Coordinates;

impl Field<Vec2, Vec2> for Coordinates {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        input
    }
}

impl Field<Vec3, Vec3> for Coordinates {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> Vec3 {
        input
    }
}

impl Field<f32, f32> for Coordinates {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        input
    }
}

// ============================================================================
// Function adapter
// ============================================================================

/// Function adapter - wraps a closure as a field.
pub struct FnField<I, O, F> {
    f: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F: Fn(I, &EvalContext) -> O> FnField<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Fn(I, &EvalContext) -> O> Field<I, O> for FnField<I, O, F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> O {
        (self.f)(input, ctx)
    }
}

/// Creates a field from a closure.
pub fn from_fn<I, O, F: Fn(I, &EvalContext) -> O>(f: F) -> FnField<I, O, F> {
    FnField::new(f)
}
