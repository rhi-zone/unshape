//! 2D vector graphics for resin.
//!
//! Provides path primitives, vector networks, and operations for 2D vector art.

mod boolean;
mod network;
mod path;
mod stroke;
pub mod svg;

pub use boolean::{path_intersect, path_subtract, path_union, path_xor};
pub use network::{
    Edge, EdgeHandle, EdgeId, EdgeType, HandleStyle, Node, NodeId, Region, VectorNetwork,
};
pub use path::{
    Path,
    PathBuilder,
    PathCommand,
    // Primitives
    circle,
    ellipse,
    line,
    polygon,
    polyline,
    rect,
    rect_centered,
    regular_polygon,
    rounded_rect,
    star,
};
pub use stroke::{
    CapStyle, DashPattern, JoinStyle, StrokeConfig, dash_path, offset_path, path_length,
    point_at_length, stroke_to_path, tangent_at_length,
};
