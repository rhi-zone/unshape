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
/// here. Note: egui's polygon fill (`epaint::PathShape`) uses fan
/// triangulation and is only exact for convex shapes — acceptable for this
/// POC, but a sharply back-tracking stroke may show minor fill artifacts at
/// its joins.
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
    painter.add(egui::Shape::Path(egui::epaint::PathShape::convex_polygon(
        points,
        color,
        egui::Stroke::new(1.0, color.gamma_multiply(0.7)),
    )));
}
