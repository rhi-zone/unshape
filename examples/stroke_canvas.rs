//! Minimal proof-of-concept: pressure-sensitive stroke capture on an egui canvas.
//!
//! Draws with mouse or stylus. While the pointer is down, raw `(position,
//! pressure)` samples are collected into a `PressureStroke`. On release the
//! pressure channel is smoothed and `PressureStrokeRender` (from
//! `unshape-vector`) converts the samples into a filled vector outline
//! (`Path`), which is painted with egui's `Painter`. Supports multiple
//! strokes: draw, release, draw again.
//!
//! Pressure is read from egui's `Event::Touch { force, .. }`, which
//! tablet/touch-capable backends populate; plain mouse input has no pressure
//! channel and falls back to 1.0.
//!
//! Run with: `cargo run --example stroke_canvas`

use eframe::egui;
use glam::Vec2;
use unshape_vector::{Path, PathCommand, PressureStroke, PressureStrokeRender};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "unshape stroke canvas (POC)",
        options,
        Box::new(|_cc| Ok(Box::new(StrokeCanvasApp::default()))),
    )
}

struct StrokeCanvasApp {
    /// Finished strokes, already converted to filled outline paths.
    strokes: Vec<Path>,
    /// Raw pressure samples for the stroke currently being drawn.
    current: PressureStroke,
    /// Whether the pointer is currently down and drawing.
    drawing: bool,
    /// Most recently reported pressure (from a touch/pen event), reused for
    /// plain `PointerMoved` samples that carry no pressure of their own.
    last_pressure: f32,
    /// Pressure-to-width configuration applied to every stroke.
    config: PressureStrokeRender,
}

impl Default for StrokeCanvasApp {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            current: PressureStroke::new(),
            drawing: false,
            last_pressure: 1.0,
            config: PressureStrokeRender::new(1.0, 10.0),
        }
    }
}

impl eframe::App for StrokeCanvasApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Stroke canvas (POC)");
                if ui.button("Clear").clicked() {
                    self.strokes.clear();
                }
                ui.weak(format!("{} stroke(s)", self.strokes.len()));
            });
            ui.label(
                "Draw with mouse or stylus. Pressure is used where the input backend reports it \
                 (e.g. a graphics tablet); mouse input draws at full pressure.",
            );
            ui.separator();

            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_gray(24));

            self.handle_input(ui, rect, &response);
            self.paint(&painter, rect);
        });
    }
}

impl StrokeCanvasApp {
    /// Consumes this frame's raw input events, updating the in-progress
    /// stroke and finalizing it on release.
    fn handle_input(&mut self, ui: &egui::Ui, rect: egui::Rect, response: &egui::Response) {
        if response.drag_started() {
            self.current = PressureStroke::new();
            self.drawing = true;
            if let Some(pos) = response.interact_pointer_pos() {
                self.push_point(pos, rect, self.last_pressure);
            }
        }

        // `Event::Touch` carries pressure (`force`) alongside the same
        // `PointerMoved`/`PointerButton` events plain pointers emit; walk raw
        // events for both so tablets and mice both drive the stroke.
        let events = ui.ctx().input(|i| i.events.clone());
        for event in &events {
            match event {
                egui::Event::Touch { pos, force, .. } => {
                    if let Some(force) = force {
                        self.last_pressure = *force;
                    }
                    if self.drawing {
                        self.push_point(*pos, rect, self.last_pressure);
                    }
                }
                egui::Event::PointerMoved(pos) if self.drawing => {
                    self.push_point(*pos, rect, self.last_pressure);
                }
                _ => {}
            }
        }

        if response.drag_stopped() {
            self.finish_stroke();
        }
    }

    /// Appends a sample to the in-progress stroke, in canvas-local coordinates.
    fn push_point(&mut self, pos: egui::Pos2, rect: egui::Rect, pressure: f32) {
        if !rect.contains(pos) {
            return;
        }
        let local = pos - rect.min;
        let point = Vec2::new(local.x, local.y);
        // Skip near-duplicate samples so a still pen doesn't flood the stroke
        // with thousands of coincident points.
        if let Some(last) = self.current.points.last()
            && (last.position - point).length() < 0.75
        {
            return;
        }
        self.current.add_point(point, pressure);
    }

    /// Converts the in-progress raw samples into a clean filled outline and
    /// files it away as a finished stroke.
    fn finish_stroke(&mut self) {
        if !self.drawing {
            return;
        }
        self.drawing = false;
        let raw = std::mem::replace(&mut self.current, PressureStroke::new());
        if raw.len() < 2 {
            return;
        }
        // Smooth the recorded pressure signal, then let unshape-vector's
        // stroke simulation turn the pressure samples into a filled outline.
        let smoothed = raw.smooth_pressure(2);
        let path = self.config.apply(&smoothed);
        if !path.is_empty() {
            self.strokes.push(path);
        }
    }

    fn paint(&self, painter: &egui::Painter, rect: egui::Rect) {
        for path in &self.strokes {
            paint_outline(painter, rect, path, egui::Color32::from_rgb(225, 225, 255));
        }

        // Live preview of the stroke currently being drawn.
        if self.drawing && self.current.len() >= 2 {
            let preview = self.config.apply(&self.current);
            paint_outline(
                painter,
                rect,
                &preview,
                egui::Color32::from_rgb(255, 200, 120),
            );
        }
    }
}

/// Converts a filled vector outline (produced by `PressureStrokeRender`) into
/// egui screen-space points and paints it.
///
/// `PressureStrokeRender::apply` always emits a closed polygon built only
/// from `MoveTo`/`LineTo`/`Close` commands, so no curve flattening is needed
/// here. Pressure-sensitive stroke outlines are generally NON-convex
/// (variable width plus joins/caps can fold back on themselves), so
/// `epaint::PathShape::convex_polygon`'s fan triangulation (fixed at vertex
/// 0) would produce visible fill artifacts. Instead, the polygon interior is
/// triangulated with ear-clipping and painted as an `egui::Shape::Mesh`; the
/// boundary is stroked separately with an unfilled `PathShape`.
fn paint_outline(painter: &egui::Painter, rect: egui::Rect, path: &Path, color: egui::Color32) {
    let mut points = Vec::with_capacity(path.commands().len());
    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) | PathCommand::LineTo(p) => {
                points.push(rect.min + egui::vec2(p.x, p.y));
            }
            PathCommand::QuadTo { .. } | PathCommand::CubicTo { .. } | PathCommand::Close => {}
        }
    }
    if points.len() < 3 {
        return;
    }

    let mut mesh = egui::Mesh::default();
    mesh.vertices
        .extend(points.iter().map(|&p| egui::epaint::Vertex {
            pos: p,
            uv: egui::epaint::WHITE_UV,
            color,
        }));
    for [a, b, c] in triangulate_polygon(&points) {
        mesh.add_triangle(a, b, c);
    }
    painter.add(egui::Shape::mesh(mesh));

    painter.add(egui::Shape::Path(egui::epaint::PathShape::closed_line(
        points,
        egui::Stroke::new(1.0, color.gamma_multiply(0.7)),
    )));
}

/// Triangulates a simple (non-self-intersecting) polygon via ear clipping.
///
/// Works for both convex and non-convex polygons, in either winding order.
/// Returns triangles as index triples into `points`. If the polygon is
/// degenerate (e.g. has coincident/collinear points that prevent finding a
/// valid ear), triangulation stops early and simply omits the remaining
/// interior — better to under-fill than to loop forever.
fn triangulate_polygon(points: &[egui::Pos2]) -> Vec<[u32; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // Winding order of the polygon, used to tell convex corners from reflex
    // ones consistently.
    let signed_area: f32 = (0..n)
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum::<f32>()
        * 0.5;
    let ccw = signed_area >= 0.0;

    let is_convex_corner = |a: egui::Pos2, b: egui::Pos2, c: egui::Pos2| -> bool {
        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if ccw { cross > 0.0 } else { cross < 0.0 }
    };
    let point_in_triangle = |p: egui::Pos2, a: egui::Pos2, b: egui::Pos2, c: egui::Pos2| -> bool {
        let sign = |p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2| -> f32 {
            (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        };
        let d1 = sign(p, a, b);
        let d2 = sign(p, b, c);
        let d3 = sign(p, c, a);
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(has_neg && has_pos)
    };

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::with_capacity(n.saturating_sub(2));

    while remaining.len() > 3 {
        let m = remaining.len();
        let mut ear_index = None;
        for i in 0..m {
            let prev = remaining[(i + m - 1) % m];
            let curr = remaining[i];
            let next = remaining[(i + 1) % m];
            let (a, b, c) = (points[prev], points[curr], points[next]);
            if !is_convex_corner(a, b, c) {
                continue;
            }
            let no_points_inside = remaining
                .iter()
                .filter(|&&idx| idx != prev && idx != curr && idx != next)
                .all(|&idx| !point_in_triangle(points[idx], a, b, c));
            if no_points_inside {
                ear_index = Some((i, prev, curr, next));
                break;
            }
        }
        let Some((i, prev, curr, next)) = ear_index else {
            // Degenerate polygon (e.g. collinear/duplicate points prevent
            // finding a valid ear); stop rather than looping forever.
            break;
        };
        triangles.push([prev as u32, curr as u32, next as u32]);
        remaining.remove(i);
    }
    if remaining.len() == 3 {
        triangles.push([
            remaining[0] as u32,
            remaining[1] as u32,
            remaining[2] as u32,
        ]);
    }

    triangles
}
