//! SVG export for 2D vector graphics.
//!
//! Provides functionality to export paths and shapes to SVG format.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_vector::{Path, PathBuilder, circle};
//! use rhizome_resin_vector::svg::{SvgDocument, SvgStyle};
//!
//! let path = circle(glam::Vec2::new(50.0, 50.0), 40.0);
//! let mut doc = SvgDocument::new(100.0, 100.0);
//! doc.add_path(&path, SvgStyle::stroke("#ff0000", 2.0));
//! let svg_string = doc.to_string();
//! ```

use crate::{Path, PathCommand};
use glam::Vec2;
use std::fmt::Write;

/// Style for SVG elements.
#[derive(Debug, Clone)]
pub struct SvgStyle {
    /// Fill color (None for no fill).
    pub fill: Option<String>,
    /// Stroke color (None for no stroke).
    pub stroke: Option<String>,
    /// Stroke width.
    pub stroke_width: f32,
    /// Stroke line cap style.
    pub stroke_linecap: StrokeLinecap,
    /// Stroke line join style.
    pub stroke_linejoin: StrokeLinejoin,
    /// Fill opacity (0.0 to 1.0).
    pub fill_opacity: f32,
    /// Stroke opacity (0.0 to 1.0).
    pub stroke_opacity: f32,
    /// Stroke dash array.
    pub stroke_dasharray: Option<Vec<f32>>,
    /// Stroke dash offset.
    pub stroke_dashoffset: f32,
}

/// Stroke line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeLinecap {
    /// Flat cap at the end of the line.
    #[default]
    Butt,
    /// Rounded cap.
    Round,
    /// Square cap extending past the end.
    Square,
}

/// Stroke line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeLinejoin {
    /// Sharp corner.
    #[default]
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled corner.
    Bevel,
}

impl Default for SvgStyle {
    fn default() -> Self {
        Self {
            fill: Some("black".to_string()),
            stroke: None,
            stroke_width: 1.0,
            stroke_linecap: StrokeLinecap::default(),
            stroke_linejoin: StrokeLinejoin::default(),
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            stroke_dasharray: None,
            stroke_dashoffset: 0.0,
        }
    }
}

impl SvgStyle {
    /// Creates a style with only fill.
    pub fn fill(color: impl Into<String>) -> Self {
        Self {
            fill: Some(color.into()),
            stroke: None,
            ..Default::default()
        }
    }

    /// Creates a style with only stroke.
    pub fn stroke(color: impl Into<String>, width: f32) -> Self {
        Self {
            fill: None,
            stroke: Some(color.into()),
            stroke_width: width,
            ..Default::default()
        }
    }

    /// Creates a style with both fill and stroke.
    pub fn fill_stroke(
        fill_color: impl Into<String>,
        stroke_color: impl Into<String>,
        stroke_width: f32,
    ) -> Self {
        Self {
            fill: Some(fill_color.into()),
            stroke: Some(stroke_color.into()),
            stroke_width,
            ..Default::default()
        }
    }

    /// Converts the style to SVG attribute string.
    fn to_attributes(&self) -> String {
        let mut attrs = String::new();

        match &self.fill {
            Some(color) => write!(&mut attrs, "fill=\"{}\" ", color).unwrap(),
            None => attrs.push_str("fill=\"none\" "),
        }

        if self.fill_opacity < 1.0 {
            write!(&mut attrs, "fill-opacity=\"{:.3}\" ", self.fill_opacity).unwrap();
        }

        if let Some(color) = &self.stroke {
            write!(&mut attrs, "stroke=\"{}\" ", color).unwrap();
            write!(&mut attrs, "stroke-width=\"{:.3}\" ", self.stroke_width).unwrap();

            if self.stroke_opacity < 1.0 {
                write!(&mut attrs, "stroke-opacity=\"{:.3}\" ", self.stroke_opacity).unwrap();
            }

            match self.stroke_linecap {
                StrokeLinecap::Round => attrs.push_str("stroke-linecap=\"round\" "),
                StrokeLinecap::Square => attrs.push_str("stroke-linecap=\"square\" "),
                StrokeLinecap::Butt => {}
            }

            match self.stroke_linejoin {
                StrokeLinejoin::Round => attrs.push_str("stroke-linejoin=\"round\" "),
                StrokeLinejoin::Bevel => attrs.push_str("stroke-linejoin=\"bevel\" "),
                StrokeLinejoin::Miter => {}
            }

            if let Some(dasharray) = &self.stroke_dasharray {
                let dashes: Vec<String> = dasharray.iter().map(|d| format!("{:.3}", d)).collect();
                write!(&mut attrs, "stroke-dasharray=\"{}\" ", dashes.join(",")).unwrap();
            }

            if self.stroke_dashoffset != 0.0 {
                write!(
                    &mut attrs,
                    "stroke-dashoffset=\"{:.3}\" ",
                    self.stroke_dashoffset
                )
                .unwrap();
            }
        }

        attrs.trim_end().to_string()
    }
}

/// An SVG document containing paths and shapes.
#[derive(Debug, Clone)]
pub struct SvgDocument {
    /// Document width.
    pub width: f32,
    /// Document height.
    pub height: f32,
    /// ViewBox (min_x, min_y, width, height).
    pub viewbox: Option<ViewBox>,
    /// Elements in the document.
    elements: Vec<SvgElement>,
}

/// SVG viewBox definition.
#[derive(Debug, Clone, Copy)]
pub struct ViewBox {
    /// Minimum X coordinate.
    pub min_x: f32,
    /// Minimum Y coordinate.
    pub min_y: f32,
    /// Width of the viewBox.
    pub width: f32,
    /// Height of the viewBox.
    pub height: f32,
}

/// An element in an SVG document.
#[derive(Debug, Clone)]
enum SvgElement {
    Path {
        data: String,
        style: SvgStyle,
    },
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        style: SvgStyle,
    },
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        rx: f32,
        ry: f32,
        style: SvgStyle,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        style: SvgStyle,
    },
    Group {
        elements: Vec<SvgElement>,
        transform: Option<String>,
    },
}

impl SvgDocument {
    /// Creates a new SVG document with the given dimensions.
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            viewbox: None,
            elements: Vec::new(),
        }
    }

    /// Creates a new SVG document with a viewBox.
    pub fn with_viewbox(width: f32, height: f32, viewbox: ViewBox) -> Self {
        Self {
            width,
            height,
            viewbox: Some(viewbox),
            elements: Vec::new(),
        }
    }

    /// Adds a path to the document.
    pub fn add_path(&mut self, path: &Path, style: SvgStyle) {
        let data = path_to_svg_data(path);
        self.elements.push(SvgElement::Path { data, style });
    }

    /// Adds a circle to the document.
    pub fn add_circle(&mut self, center: Vec2, radius: f32, style: SvgStyle) {
        self.elements.push(SvgElement::Circle {
            cx: center.x,
            cy: center.y,
            r: radius,
            style,
        });
    }

    /// Adds a rectangle to the document.
    pub fn add_rect(&mut self, pos: Vec2, size: Vec2, style: SvgStyle) {
        self.elements.push(SvgElement::Rect {
            x: pos.x,
            y: pos.y,
            width: size.x,
            height: size.y,
            rx: 0.0,
            ry: 0.0,
            style,
        });
    }

    /// Adds a rounded rectangle to the document.
    pub fn add_rounded_rect(&mut self, pos: Vec2, size: Vec2, radius: f32, style: SvgStyle) {
        self.elements.push(SvgElement::Rect {
            x: pos.x,
            y: pos.y,
            width: size.x,
            height: size.y,
            rx: radius,
            ry: radius,
            style,
        });
    }

    /// Adds a line to the document.
    pub fn add_line(&mut self, from: Vec2, to: Vec2, style: SvgStyle) {
        self.elements.push(SvgElement::Line {
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            style,
        });
    }

    /// Converts the document to an SVG string.
    pub fn to_svg_string(&self) -> String {
        let mut svg = String::new();

        // XML declaration
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // SVG element
        write!(
            &mut svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.3}\" height=\"{:.3}\"",
            self.width, self.height
        )
        .unwrap();

        if let Some(vb) = &self.viewbox {
            write!(
                &mut svg,
                " viewBox=\"{:.3} {:.3} {:.3} {:.3}\"",
                vb.min_x, vb.min_y, vb.width, vb.height
            )
            .unwrap();
        }

        svg.push_str(">\n");

        // Elements
        for element in &self.elements {
            write_element(&mut svg, element, 1);
        }

        svg.push_str("</svg>\n");
        svg
    }
}

impl std::fmt::Display for SvgDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_svg_string())
    }
}

/// Writes an SVG element to a string with indentation.
fn write_element(svg: &mut String, element: &SvgElement, indent: usize) {
    let indent_str = "  ".repeat(indent);

    match element {
        SvgElement::Path { data, style } => {
            write!(
                svg,
                "{}<path d=\"{}\" {}/>\n",
                indent_str,
                data,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Circle { cx, cy, r, style } => {
            write!(
                svg,
                "{}<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"{:.3}\" {}/>\n",
                indent_str,
                cx,
                cy,
                r,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Rect {
            x,
            y,
            width,
            height,
            rx,
            ry,
            style,
        } => {
            write!(
                svg,
                "{}<rect x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\"",
                indent_str, x, y, width, height
            )
            .unwrap();
            if *rx > 0.0 {
                write!(svg, " rx=\"{:.3}\"", rx).unwrap();
            }
            if *ry > 0.0 {
                write!(svg, " ry=\"{:.3}\"", ry).unwrap();
            }
            write!(svg, " {}/>\n", style.to_attributes()).unwrap();
        }
        SvgElement::Line {
            x1,
            y1,
            x2,
            y2,
            style,
        } => {
            write!(
                svg,
                "{}<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\" {}/>\n",
                indent_str,
                x1,
                y1,
                x2,
                y2,
                style.to_attributes()
            )
            .unwrap();
        }
        SvgElement::Group {
            elements,
            transform,
        } => {
            if let Some(t) = transform {
                write!(svg, "{}<g transform=\"{}\">\n", indent_str, t).unwrap();
            } else {
                write!(svg, "{}<g>\n", indent_str).unwrap();
            }
            for child in elements {
                write_element(svg, child, indent + 1);
            }
            write!(svg, "{}</g>\n", indent_str).unwrap();
        }
    }
}

/// Converts a Path to SVG path data string.
pub fn path_to_svg_data(path: &Path) -> String {
    let mut data = String::new();

    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) => {
                write!(&mut data, "M{:.3},{:.3} ", p.x, p.y).unwrap();
            }
            PathCommand::LineTo(p) => {
                write!(&mut data, "L{:.3},{:.3} ", p.x, p.y).unwrap();
            }
            PathCommand::QuadTo { control, to } => {
                write!(
                    &mut data,
                    "Q{:.3},{:.3} {:.3},{:.3} ",
                    control.x, control.y, to.x, to.y
                )
                .unwrap();
            }
            PathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                write!(
                    &mut data,
                    "C{:.3},{:.3} {:.3},{:.3} {:.3},{:.3} ",
                    control1.x, control1.y, control2.x, control2.y, to.x, to.y
                )
                .unwrap();
            }
            PathCommand::Close => {
                data.push_str("Z ");
            }
        }
    }

    data.trim_end().to_string()
}

/// Converts a Path to a complete SVG string (simple helper).
pub fn path_to_svg(path: &Path, width: f32, height: f32, style: SvgStyle) -> String {
    let mut doc = SvgDocument::new(width, height);
    doc.add_path(path, style);
    doc.to_svg_string()
}

// ============================================================================
// SVG Import / Parsing
// ============================================================================

/// Error type for SVG parsing.
#[derive(Debug, thiserror::Error)]
pub enum SvgParseError {
    /// Invalid path data.
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    /// Invalid number format.
    #[error("Invalid number: {0}")]
    InvalidNumber(String),
    /// Unexpected end of input.
    #[error("Unexpected end of path data")]
    UnexpectedEnd,
    /// Unknown command.
    #[error("Unknown command: {0}")]
    UnknownCommand(char),
}

/// Result type for SVG parsing.
pub type SvgParseResult<T> = Result<T, SvgParseError>;

/// Parses SVG path data (d attribute) into a Path.
///
/// Supports the following commands:
/// - M/m: moveto
/// - L/l: lineto
/// - H/h: horizontal lineto
/// - V/v: vertical lineto
/// - C/c: cubic bezier
/// - Q/q: quadratic bezier
/// - S/s: smooth cubic bezier
/// - T/t: smooth quadratic bezier
/// - A/a: arc (approximated with cubic beziers)
/// - Z/z: closepath
///
/// # Example
///
/// ```
/// use rhizome_resin_vector::svg::parse_path_data;
///
/// let path = parse_path_data("M0,0 L100,100 Z").unwrap();
/// assert!(!path.is_empty());
/// ```
pub fn parse_path_data(data: &str) -> SvgParseResult<Path> {
    let mut builder = crate::PathBuilder::new();
    let mut current = Vec2::ZERO;
    let mut start = Vec2::ZERO;
    let mut last_control: Option<Vec2> = None;
    let mut last_command: Option<char> = None;

    let mut chars = data.chars().peekable();

    while let Some(&c) = chars.peek() {
        // Skip whitespace and commas
        if c.is_whitespace() || c == ',' {
            chars.next();
            continue;
        }

        // Determine command
        let command = if c.is_alphabetic() {
            chars.next();
            c
        } else if let Some(lc) = last_command {
            // Implicit command (repeat last command)
            match lc {
                'M' => 'L',
                'm' => 'l',
                _ => lc,
            }
        } else {
            return Err(SvgParseError::InvalidPath("Expected command".to_string()));
        };

        let is_relative = command.is_lowercase();
        let base = if is_relative { current } else { Vec2::ZERO };

        match command.to_ascii_uppercase() {
            'M' => {
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;
                current = base + Vec2::new(x, y);
                start = current;
                builder = builder.move_to(current);
                last_control = None;
            }
            'L' => {
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;
                current = base + Vec2::new(x, y);
                builder = builder.line_to(current);
                last_control = None;
            }
            'H' => {
                let x = parse_number(&mut chars)?;
                current.x = if is_relative { current.x + x } else { x };
                builder = builder.line_to(current);
                last_control = None;
            }
            'V' => {
                let y = parse_number(&mut chars)?;
                current.y = if is_relative { current.y + y } else { y };
                builder = builder.line_to(current);
                last_control = None;
            }
            'C' => {
                let x1 = parse_number(&mut chars)?;
                let y1 = parse_number(&mut chars)?;
                let x2 = parse_number(&mut chars)?;
                let y2 = parse_number(&mut chars)?;
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;

                let cp1 = base + Vec2::new(x1, y1);
                let cp2 = base + Vec2::new(x2, y2);
                current = base + Vec2::new(x, y);

                builder = builder.cubic_to(cp1, cp2, current);
                last_control = Some(cp2);
            }
            'S' => {
                // Smooth cubic - reflect last control point
                let cp1 = match last_control {
                    Some(lc) => current * 2.0 - lc,
                    None => current,
                };

                let x2 = parse_number(&mut chars)?;
                let y2 = parse_number(&mut chars)?;
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;

                let cp2 = base + Vec2::new(x2, y2);
                current = base + Vec2::new(x, y);

                builder = builder.cubic_to(cp1, cp2, current);
                last_control = Some(cp2);
            }
            'Q' => {
                let x1 = parse_number(&mut chars)?;
                let y1 = parse_number(&mut chars)?;
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;

                let cp = base + Vec2::new(x1, y1);
                current = base + Vec2::new(x, y);

                builder = builder.quad_to(cp, current);
                last_control = Some(cp);
            }
            'T' => {
                // Smooth quadratic - reflect last control point
                let cp = match last_control {
                    Some(lc) => current * 2.0 - lc,
                    None => current,
                };

                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;
                current = base + Vec2::new(x, y);

                builder = builder.quad_to(cp, current);
                last_control = Some(cp);
            }
            'A' => {
                // Arc command - convert to cubic bezier approximation
                let rx = parse_number(&mut chars)?;
                let ry = parse_number(&mut chars)?;
                let x_rotation = parse_number(&mut chars)?;
                let large_arc = parse_number(&mut chars)? != 0.0;
                let sweep = parse_number(&mut chars)? != 0.0;
                let x = parse_number(&mut chars)?;
                let y = parse_number(&mut chars)?;

                let end = base + Vec2::new(x, y);

                // Convert arc to cubic beziers
                builder =
                    arc_to_cubics(builder, current, end, rx, ry, x_rotation, large_arc, sweep);
                current = end;
                last_control = None;
            }
            'Z' => {
                builder = builder.close();
                current = start;
                last_control = None;
                // Z takes no parameters, so it shouldn't become the implicit command
                // (otherwise "z-5" would infinite loop trying to repeat Z)
                last_command = None;
                continue;
            }
            _ => {
                return Err(SvgParseError::UnknownCommand(command));
            }
        }

        last_command = Some(command);
    }

    Ok(builder.build())
}

/// Parses a number from the character stream.
fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> SvgParseResult<f32> {
    // Skip whitespace and commas
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() || c == ',' {
            chars.next();
        } else {
            break;
        }
    }

    let mut s = String::new();

    // Handle sign
    if let Some(&c) = chars.peek() {
        if c == '-' || c == '+' {
            s.push(chars.next().unwrap());
        }
    }

    // Handle digits and decimal point
    let mut has_dot = false;
    let mut has_exp = false;

    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            s.push(chars.next().unwrap());
        } else if c == '.' && !has_dot && !has_exp {
            has_dot = true;
            s.push(chars.next().unwrap());
        } else if (c == 'e' || c == 'E') && !has_exp {
            has_exp = true;
            s.push(chars.next().unwrap());
            // Handle exponent sign
            if let Some(&c2) = chars.peek() {
                if c2 == '-' || c2 == '+' {
                    s.push(chars.next().unwrap());
                }
            }
        } else {
            break;
        }
    }

    if s.is_empty() || s == "-" || s == "+" {
        return Err(SvgParseError::UnexpectedEnd);
    }

    s.parse::<f32>()
        .map_err(|_| SvgParseError::InvalidNumber(s))
}

/// Converts an arc to cubic bezier curves.
fn arc_to_cubics(
    mut builder: crate::PathBuilder,
    start: Vec2,
    end: Vec2,
    rx: f32,
    ry: f32,
    x_rotation: f32,
    large_arc: bool,
    sweep: bool,
) -> crate::PathBuilder {
    // Handle degenerate cases
    if (start - end).length() < 1e-6 {
        return builder;
    }

    if rx.abs() < 1e-6 || ry.abs() < 1e-6 {
        return builder.line_to(end);
    }

    let mut rx = rx.abs();
    let mut ry = ry.abs();

    // Convert to center parameterization
    let phi = x_rotation.to_radians();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Transform to unit circle space
    let dx = (start.x - end.x) / 2.0;
    let dy = (start.y - end.y) / 2.0;
    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    // Correct out-of-range radii
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let sqrt_lambda = lambda.sqrt();
        rx *= sqrt_lambda;
        ry *= sqrt_lambda;
    }

    // Compute center point
    let rxsq = rx * rx;
    let rysq = ry * ry;
    let x1psq = x1p * x1p;
    let y1psq = y1p * y1p;

    let radicand =
        ((rxsq * rysq) - (rxsq * y1psq) - (rysq * x1psq)) / ((rxsq * y1psq) + (rysq * x1psq));
    let radicand = radicand.max(0.0);
    let coef = if large_arc != sweep { 1.0 } else { -1.0 } * radicand.sqrt();

    let cxp = coef * rx * y1p / ry;
    let cyp = -coef * ry * x1p / rx;

    // Transform back from unit circle space
    let cx = cos_phi * cxp - sin_phi * cyp + (start.x + end.x) / 2.0;
    let cy = sin_phi * cxp + cos_phi * cyp + (start.y + end.y) / 2.0;

    // Compute angles
    fn angle(ux: f32, uy: f32, vx: f32, vy: f32) -> f32 {
        let n = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        if n < 1e-6 {
            return 0.0;
        }
        let c = (ux * vx + uy * vy) / n;
        let c = c.clamp(-1.0, 1.0);
        let sign = if ux * vy - uy * vx < 0.0 { -1.0 } else { 1.0 };
        sign * c.acos()
    }

    let theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry);
    let mut dtheta = angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    );

    if !sweep && dtheta > 0.0 {
        dtheta -= std::f32::consts::TAU;
    } else if sweep && dtheta < 0.0 {
        dtheta += std::f32::consts::TAU;
    }

    // Split into segments of at most 90 degrees
    let n_segs = (dtheta.abs() / (std::f32::consts::FRAC_PI_2)).ceil() as i32;
    let n_segs = n_segs.max(1);
    let d_theta = dtheta / n_segs as f32;

    for i in 0..n_segs {
        let theta = theta1 + d_theta * i as f32;
        let theta_end = theta + d_theta;

        // Compute bezier control points for this arc segment
        let t = (d_theta / 4.0).tan();
        let alpha = (d_theta.sin()) * ((4.0 + 3.0 * t * t).sqrt() - 1.0) / 3.0;

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let cos_te = theta_end.cos();
        let sin_te = theta_end.sin();

        let p1x = rx * cos_t;
        let p1y = ry * sin_t;
        let p2x = rx * cos_te;
        let p2y = ry * sin_te;

        let cp1x = p1x - alpha * rx * sin_t;
        let cp1y = p1y + alpha * ry * cos_t;
        let cp2x = p2x + alpha * rx * sin_te;
        let cp2y = p2y - alpha * ry * cos_te;

        // Transform back from unit circle space
        let transform = |x: f32, y: f32| -> Vec2 {
            Vec2::new(
                cos_phi * x - sin_phi * y + cx,
                sin_phi * x + cos_phi * y + cy,
            )
        };

        let cp1 = transform(cp1x, cp1y);
        let cp2 = transform(cp2x, cp2y);
        let p2 = transform(p2x, p2y);

        builder = builder.cubic_to(cp1, cp2, p2);
    }

    builder
}

/// Parses multiple path data strings and returns a vector of Paths.
pub fn parse_paths_data(data: &[&str]) -> SvgParseResult<Vec<Path>> {
    data.iter().map(|d| parse_path_data(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circle, line, rect};

    #[test]
    fn test_path_to_svg_data() {
        let path = line(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let data = path_to_svg_data(&path);
        assert!(data.contains("M0.000,0.000"));
        assert!(data.contains("L100.000,100.000"));
    }

    #[test]
    fn test_svg_document_basic() {
        let mut doc = SvgDocument::new(200.0, 200.0);
        doc.add_circle(Vec2::new(100.0, 100.0), 50.0, SvgStyle::fill("red"));

        let svg = doc.to_svg_string();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("circle"));
        assert!(svg.contains("fill=\"red\""));
    }

    #[test]
    fn test_svg_style_stroke() {
        let style = SvgStyle::stroke("#0000ff", 2.0);
        let attrs = style.to_attributes();
        assert!(attrs.contains("stroke=\"#0000ff\""));
        assert!(attrs.contains("stroke-width=\"2.000\""));
        assert!(attrs.contains("fill=\"none\""));
    }

    #[test]
    fn test_svg_style_fill_stroke() {
        let style = SvgStyle::fill_stroke("white", "black", 1.0);
        let attrs = style.to_attributes();
        assert!(attrs.contains("fill=\"white\""));
        assert!(attrs.contains("stroke=\"black\""));
    }

    #[test]
    fn test_svg_path() {
        let path = circle(Vec2::new(50.0, 50.0), 40.0);
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_path(&path, SvgStyle::stroke("blue", 2.0));

        let svg = doc.to_svg_string();
        assert!(svg.contains("<path"));
        assert!(svg.contains("d=\""));
    }

    #[test]
    fn test_svg_rect() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_rect(
            Vec2::new(10.0, 10.0),
            Vec2::new(80.0, 80.0),
            SvgStyle::fill("green"),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("<rect"));
        assert!(svg.contains("x=\"10.000\""));
        assert!(svg.contains("width=\"80.000\""));
    }

    #[test]
    fn test_svg_rounded_rect() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_rounded_rect(
            Vec2::new(10.0, 10.0),
            Vec2::new(80.0, 80.0),
            5.0,
            SvgStyle::fill("blue"),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("rx=\"5.000\""));
    }

    #[test]
    fn test_svg_line() {
        let mut doc = SvgDocument::new(100.0, 100.0);
        doc.add_line(
            Vec2::ZERO,
            Vec2::new(100.0, 100.0),
            SvgStyle::stroke("red", 1.0),
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("<line"));
        assert!(svg.contains("x1=\"0.000\""));
        assert!(svg.contains("x2=\"100.000\""));
    }

    #[test]
    fn test_svg_dashed_stroke() {
        let mut style = SvgStyle::stroke("black", 1.0);
        style.stroke_dasharray = Some(vec![5.0, 3.0]);
        let attrs = style.to_attributes();
        assert!(attrs.contains("stroke-dasharray=\"5.000,3.000\""));
    }

    #[test]
    fn test_svg_viewbox() {
        let doc = SvgDocument::with_viewbox(
            100.0,
            100.0,
            ViewBox {
                min_x: -50.0,
                min_y: -50.0,
                width: 100.0,
                height: 100.0,
            },
        );

        let svg = doc.to_svg_string();
        assert!(svg.contains("viewBox=\"-50.000 -50.000 100.000 100.000\""));
    }

    #[test]
    fn test_path_to_svg_helper() {
        let path = rect(Vec2::new(10.0, 10.0), Vec2::new(80.0, 80.0));
        let svg = path_to_svg(&path, 100.0, 100.0, SvgStyle::fill("red"));

        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
        assert!(svg.contains("</svg>"));
    }

    // ========== SVG Import Tests ==========

    #[test]
    fn test_parse_simple_line() {
        let path = parse_path_data("M0,0 L100,100").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_closed_path() {
        let path = parse_path_data("M0,0 L100,0 L100,100 L0,100 Z").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_relative_commands() {
        let path = parse_path_data("m10,10 l50,0 l0,50 l-50,0 z").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_horizontal_vertical() {
        let path = parse_path_data("M0,0 H100 V100 H0 V0 Z").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_cubic_bezier() {
        let path = parse_path_data("M0,0 C25,50 75,50 100,0").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_smooth_cubic() {
        let path = parse_path_data("M0,0 C25,50 75,50 100,0 S175,-50 200,0").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_quadratic_bezier() {
        let path = parse_path_data("M0,0 Q50,50 100,0").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_smooth_quadratic() {
        let path = parse_path_data("M0,0 Q50,50 100,0 T200,0").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_arc() {
        let path = parse_path_data("M0,50 A50,50 0 1,1 100,50").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_implicit_lineto() {
        // After M, subsequent coordinate pairs are treated as L
        let path = parse_path_data("M0,0 10,10 20,20").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_whitespace_variants() {
        // Various whitespace separators
        let path1 = parse_path_data("M0,0L100,100").unwrap();
        let path2 = parse_path_data("M0 0 L100 100").unwrap();
        let path3 = parse_path_data("M 0 , 0 L 100 , 100").unwrap();

        assert!(!path1.is_empty());
        assert!(!path2.is_empty());
        assert!(!path3.is_empty());
    }

    #[test]
    fn test_parse_negative_numbers() {
        let path = parse_path_data("M-10,-10 L-50,-50").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_scientific_notation() {
        let path = parse_path_data("M1e2,1e2 L2e2,2e2").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_error_unknown_command() {
        let result = parse_path_data("M0,0 X100,100");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_missing_coordinates() {
        let result = parse_path_data("M0,0 L100");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_multiple_paths() {
        let paths = parse_paths_data(&["M0,0 L100,100", "M50,50 L150,150"]).unwrap();

        assert_eq!(paths.len(), 2);
    }
}
