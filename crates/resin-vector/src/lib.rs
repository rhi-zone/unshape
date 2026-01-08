//! 2D vector graphics for resin.
//!
//! Provides path primitives and operations for 2D vector art.

mod path;

pub use path::{
    Path, PathBuilder, PathCommand,
    // Primitives
    circle, ellipse, line, polygon, polyline,
    rect, rect_centered, regular_polygon, rounded_rect, star,
};
