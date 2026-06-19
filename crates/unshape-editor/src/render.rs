//! GPU render pipeline for the editor.
//!
//! Owns a private [`GpuContext`] (NOT shared with eframe's backend) and folds the
//! op stack into a final RGBA8 image read back to the CPU. egui only ever shows a
//! CPU image, so there is no wgpu device sharing and no version coupling.

use unshape_gpu::{
    GpuContext, GpuTexture, map_pixels_ast_gpu, map_pixels_gpu, remap_uv_ast_gpu, remap_uv_gpu,
};
use wick_core::Expr;

use crate::{Document, ImageMod, ModExpr, PassKind};

/// A CPU RGBA8 image: the read-back result of running the pipeline.
pub struct RenderedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixels (`width * height * 4` bytes).
    pub rgba: Vec<u8>,
}

/// Owns the GPU context and the source texture, and runs the pipeline.
pub struct Renderer {
    ctx: GpuContext,
    width: u32,
    height: u32,
    /// Source pixels, kept on the CPU so the source texture can be rebuilt with
    /// the correct (`TEXTURE_BINDING`-capable) usage between runs.
    source_rgba: Vec<u8>,
}

impl Renderer {
    /// Creates a renderer with a procedurally generated source gradient.
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let ctx = GpuContext::new().map_err(|e| format!("GPU init failed: {e}"))?;
        let source_rgba = gradient_rgba(width, height);
        Ok(Self {
            ctx,
            width,
            height,
            source_rgba,
        })
    }

    /// Runs the full op stack and reads the result back to the CPU.
    ///
    /// Each stage's output is read back and re-uploaded as the next stage's input.
    /// The shipping image-expr passes create their output with storage-only usage,
    /// so the read-back/re-upload is what makes a `TEXTURE_BINDING` input available
    /// to the following pass.
    pub fn render(&self, doc: &Document) -> Result<RenderedImage, String> {
        let mut current: Vec<u8> = self.source_rgba.clone();

        for m in &doc.mods {
            // Free-form expression nodes that fail to parse are skipped (a
            // pass-through), so a half-typed formula never breaks the preview.
            let ast = match m {
                ImageMod::Expr { source, .. } => match Expr::parse(source) {
                    Ok(expr) => Some(expr),
                    Err(_) => continue,
                },
                _ => None,
            };

            let input = GpuTexture::from_rgba8(&self.ctx, self.width, self.height, &current)
                .map_err(|e| format!("source upload failed: {e}"))?;

            let output = match (m.to_expr(), ast.as_ref()) {
                // Typed modifiers: apply via the typed GPU path.
                (Some(ModExpr::Uv(expr)), _) => remap_uv_gpu(&self.ctx, &input, &expr)
                    .map_err(|e| format!("remap_uv_gpu failed: {e}"))?,
                (Some(ModExpr::Color(expr)), _) => map_pixels_gpu(&self.ctx, &input, &expr)
                    .map_err(|e| format!("map_pixels_gpu failed: {e}"))?,
                // Free-form expression nodes: apply the parsed AST directly.
                (None, Some(expr)) => match m.kind() {
                    PassKind::Uv => remap_uv_ast_gpu(&self.ctx, &input, expr.ast())
                        .map_err(|e| format!("remap_uv_ast_gpu failed: {e}"))?,
                    PassKind::Color => map_pixels_ast_gpu(&self.ctx, &input, expr.ast())
                        .map_err(|e| format!("map_pixels_ast_gpu failed: {e}"))?,
                },
                // Unreachable: a node is either typed (Some expr, no ast) or an
                // Expr node (None, Some ast); the parse-failure case `continue`d.
                (None, None) => continue,
            };

            current = output.read_to_rgba8(&self.ctx);
        }

        // With no modifiers, `current` is just the source gradient.
        Ok(RenderedImage {
            width: self.width,
            height: self.height,
            rgba: current,
        })
    }
}

/// Builds an RGBA8 gradient: red ramps over x, green over y, blue constant.
fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width.max(1) as f32) * 255.0) as u8;
            let g = ((y as f32 / height.max(1) as f32) * 255.0) as u8;
            rgba.extend_from_slice(&[r, g, 128, 255]);
        }
    }
    rgba
}
