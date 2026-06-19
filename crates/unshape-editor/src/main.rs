//! Projectional editor MVP binary.
//!
//! One live texture, an editable op stack on the left, the live preview in the
//! center, and an editable formula projection of the selected modifier.

use eframe::egui;
use unshape_editor::render::Renderer;
use unshape_editor::{Document, ImageMod, PassKind};
use wick_core::Expr;

const TEX_SIZE: u32 = 256;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "unshape projectional editor",
        options,
        Box::new(|_cc| Ok(Box::new(EditorApp::new()))),
    )
}

struct EditorApp {
    doc: Document,
    selected: Option<usize>,
    renderer: Option<Renderer>,
    /// Last error from GPU init / render, shown in the UI instead of panicking.
    error: Option<String>,
    /// Set whenever the document mutates; gates re-running the GPU pipeline.
    dirty: bool,
    /// Retained CPU-side egui texture of the latest render.
    preview: Option<egui::TextureHandle>,
}

impl EditorApp {
    fn new() -> Self {
        let (renderer, error) = match Renderer::new(TEX_SIZE, TEX_SIZE) {
            Ok(r) => (Some(r), None),
            Err(e) => (None, Some(e)),
        };
        Self {
            doc: Document::default(),
            selected: Some(0),
            renderer,
            error,
            dirty: true,
            preview: None,
        }
    }

    /// Re-runs the GPU pipeline and uploads the result into the egui texture.
    fn rerender(&mut self, ctx: &egui::Context) {
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        match renderer.render(&self.doc) {
            Ok(img) => {
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [img.width as usize, img.height as usize],
                    &img.rgba,
                );
                let handle = ctx.load_texture("preview", color, egui::TextureOptions::NEAREST);
                self.preview = Some(handle);
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }
}

impl eframe::App for EditorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.dirty {
            let ctx = ui.ctx().clone();
            self.rerender(&ctx);
            self.dirty = false;
        }

        egui::Panel::left("op_stack")
            .resizable(true)
            .default_size(320.0)
            .show_inside(ui, |ui| {
                ui.heading("Op-stack (editable)");
                ui.label("Ordered modifiers applied to the source texture.");
                ui.separator();

                self.op_stack_ui(ui);

                ui.separator();
                ui.heading("Formula (editable)");
                ui.label("The same modifier as the dew formula the GPU runs. Editing it drives the GPU directly.");
                self.formula_ui(ui);
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Live preview");
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, format!("Render error: {err}"));
            }
            if let Some(tex) = &self.preview {
                let size = tex.size_vec2();
                ui.image((tex.id(), size));
            } else if self.error.is_none() {
                ui.label("Rendering…");
            }
        });
    }
}

impl EditorApp {
    fn op_stack_ui(&mut self, ui: &mut egui::Ui) {
        // Row list with select + delete.
        let mut delete: Option<usize> = None;
        let len = self.doc.mods.len();
        for i in 0..len {
            ui.horizontal(|ui| {
                let selected = self.selected == Some(i);
                if ui
                    .selectable_label(selected, format!("{}: {}", i, self.doc.mods[i].label()))
                    .clicked()
                {
                    self.selected = Some(i);
                }
                if ui.small_button("up").clicked() && i > 0 {
                    self.doc.mods.swap(i, i - 1);
                    self.selected = Some(i - 1);
                    self.dirty = true;
                }
                if ui.small_button("down").clicked() && i + 1 < len {
                    self.doc.mods.swap(i, i + 1);
                    self.selected = Some(i + 1);
                    self.dirty = true;
                }
                if ui.small_button("x").clicked() {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = delete {
            self.doc.mods.remove(i);
            if self.doc.mods.is_empty() {
                self.selected = None;
            } else {
                self.selected = Some(self.selected.unwrap_or(0).min(self.doc.mods.len() - 1));
            }
            self.dirty = true;
        }

        ui.separator();
        ui.label("Add modifier:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("UV Scale").clicked() {
                self.push(ImageMod::UvScale { x: 1.0, y: 1.0 });
            }
            if ui.button("UV Rotate").clicked() {
                self.push(ImageMod::UvRotate { angle: 0.0 });
            }
            if ui.button("Brightness").clicked() {
                self.push(ImageMod::ColorBrightness { factor: 1.0 });
            }
            if ui.button("Hue Shift").clicked() {
                self.push(ImageMod::ColorHueShift { amount: 0.0 });
            }
            if ui.button("Gamma").clicked() {
                self.push(ImageMod::ColorGamma { gamma: 1.0 });
            }
            if ui.button("+ formula (uv)").clicked() {
                self.push(ImageMod::Expr {
                    kind: PassKind::Uv,
                    source: "uv".to_string(),
                });
            }
            if ui.button("+ formula (color)").clicked() {
                self.push(ImageMod::Expr {
                    kind: PassKind::Color,
                    source: "rgba".to_string(),
                });
            }
        });

        ui.separator();
        ui.label("Parameters (selected modifier):");
        self.params_ui(ui);
    }

    fn push(&mut self, m: ImageMod) {
        self.doc.mods.push(m);
        self.selected = Some(self.doc.mods.len() - 1);
        self.dirty = true;
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) {
        let Some(i) = self.selected else {
            ui.weak("No modifier selected.");
            return;
        };
        let Some(m) = self.doc.mods.get_mut(i) else {
            return;
        };

        let mut changed = false;
        match m {
            ImageMod::UvScale { x, y } => {
                changed |= ui
                    .add(egui::Slider::new(x, 0.1..=4.0).text("scale x"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(y, 0.1..=4.0).text("scale y"))
                    .changed();
            }
            ImageMod::UvRotate { angle } => {
                changed |= ui
                    .add(
                        egui::Slider::new(angle, -std::f32::consts::PI..=std::f32::consts::PI)
                            .text("angle (rad)"),
                    )
                    .changed();
            }
            ImageMod::ColorBrightness { factor } => {
                changed |= ui
                    .add(egui::Slider::new(factor, 0.0..=3.0).text("factor"))
                    .changed();
            }
            ImageMod::ColorHueShift { amount } => {
                changed |= ui
                    .add(
                        egui::Slider::new(amount, -std::f32::consts::PI..=std::f32::consts::PI)
                            .text("hue (rad)"),
                    )
                    .changed();
            }
            ImageMod::ColorGamma { gamma } => {
                changed |= ui
                    .add(egui::Slider::new(gamma, 0.1..=4.0).text("gamma"))
                    .changed();
            }
            ImageMod::Expr { .. } => {
                ui.weak("Free-form expression node — edit it in the Formula panel below.");
            }
        }
        if changed {
            self.dirty = true;
        }
    }

    fn formula_ui(&mut self, ui: &mut egui::Ui) {
        let Some(i) = self.selected else {
            ui.weak("No modifier selected.");
            return;
        };
        let Some(m) = self.doc.mods.get(i) else {
            return;
        };
        let kind = m.kind();
        let signature = match kind {
            PassKind::Uv => "uv -> uv'   (bound: uv: vec2)",
            PassKind::Color => "rgba -> rgba'   (bound: rgba: vec4)",
        };
        ui.weak(signature);

        // The formula is editable for every node. For a typed node the field is
        // seeded from its derived formula; the first edit PROMOTES it to a
        // free-form `Expr` node so editing always takes effect.
        let mut text = m.to_formula();
        let response = ui.add(
            egui::TextEdit::multiline(&mut text)
                .desired_rows(4)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );

        if response.changed() {
            // Promote typed nodes to an Expr node carrying the edited text; for an
            // Expr node, write the edit straight to its source.
            self.doc.mods[i] = ImageMod::Expr {
                kind,
                source: text.clone(),
            };
            self.dirty = true;
        }

        // Inline parse feedback. A parse error never crashes or blanks the
        // preview: the offending stage is skipped (pass-through) in the renderer.
        match Expr::parse(&text) {
            Ok(_) => {
                ui.weak("parsed ok");
            }
            Err(e) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0xff, 0x99, 0x00),
                    format!("parse error: {e} (this stage is skipped until fixed)"),
                );
            }
        }
    }
}
