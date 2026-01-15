//! Proc macros for static audio graph compilation.
//!
//! Provides compile-time code generation for audio effect graphs,
//! eliminating runtime overhead from dynamic dispatch and graph traversal.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// A node definition: `name: Type::constructor(args)`
struct NodeDef {
    name: Ident,
    ty: Type,
    init: Expr,
}

impl Parse for NodeDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;

        // Parse the full expression (Type::method(args))
        let init: Expr = input.parse()?;

        // Extract the type from the expression
        let ty = extract_type_from_expr(&init)?;

        Ok(NodeDef { name, ty, init })
    }
}

fn extract_type_from_expr(expr: &Expr) -> syn::Result<Type> {
    match expr {
        Expr::Call(call) => {
            if let Expr::Path(path) = &*call.func {
                // Get path without the last segment (method name)
                let mut segments = path.path.segments.clone();
                if segments.len() > 1 {
                    segments.pop();
                    // Remove trailing punctuation
                    let last = segments.pop().unwrap();
                    segments.push_value(last.into_value());
                }
                Ok(Type::Path(syn::TypePath {
                    qself: None,
                    path: syn::Path {
                        leading_colon: path.path.leading_colon,
                        segments,
                    },
                }))
            } else {
                Err(syn::Error::new_spanned(expr, "expected Type::method(args)"))
            }
        }
        Expr::MethodCall(mc) => extract_type_from_expr(&mc.receiver),
        Expr::Path(path) => Ok(Type::Path(syn::TypePath {
            qself: None,
            path: path.path.clone(),
        })),
        _ => Err(syn::Error::new_spanned(
            expr,
            "expected Type::method(args) or Type",
        )),
    }
}

/// Audio wire: `source -> dest`
struct AudioWire {
    source: Ident,
    dest: Ident,
}

impl Parse for AudioWire {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let source: Ident = input.parse()?;
        input.parse::<Token![->]>()?;
        let dest: Ident = input.parse()?;
        Ok(AudioWire { source, dest })
    }
}

/// Modulation wire: `source -> dest.param(base: val, scale: val)`
struct ModWire {
    source: Ident,
    dest: Ident,
    param: Ident,
    base: Expr,
    scale: Expr,
}

impl Parse for ModWire {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let source: Ident = input.parse()?;
        input.parse::<Token![->]>()?;
        let dest: Ident = input.parse()?;
        input.parse::<Token![.]>()?;
        let param: Ident = input.parse()?;

        // Parse (base: val, scale: val)
        let content;
        syn::parenthesized!(content in input);

        // base: val
        let _base_label: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        let base: Expr = content.parse()?;
        content.parse::<Token![,]>()?;

        // scale: val
        let _scale_label: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        let scale: Expr = content.parse()?;

        Ok(ModWire {
            source,
            dest,
            param,
            base,
            scale,
        })
    }
}

/// The full graph_effect! input
struct GraphEffectInput {
    name: Ident,
    nodes: Vec<NodeDef>,
    audio_wires: Vec<AudioWire>,
    mod_wires: Vec<ModWire>,
    output: Ident,
    input_node: Option<Ident>,
}

impl Parse for GraphEffectInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut nodes = Vec::new();
        let mut audio_wires = Vec::new();
        let mut mod_wires = Vec::new();
        let mut output = None;
        let mut input_node = None;

        while !input.is_empty() {
            let label: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match label.to_string().as_str() {
                "name" => {
                    name = Some(input.parse()?);
                }
                "nodes" => {
                    let content;
                    syn::braced!(content in input);
                    while !content.is_empty() {
                        nodes.push(content.parse()?);
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                        }
                    }
                }
                "audio" => {
                    let content;
                    syn::bracketed!(content in input);
                    let wires: Punctuated<AudioWire, Token![,]> =
                        Punctuated::parse_terminated(&content)?;
                    audio_wires = wires.into_iter().collect();
                }
                "modulation" => {
                    let content;
                    syn::bracketed!(content in input);
                    let wires: Punctuated<ModWire, Token![,]> =
                        Punctuated::parse_terminated(&content)?;
                    mod_wires = wires.into_iter().collect();
                }
                "output" => {
                    output = Some(input.parse()?);
                }
                "input" => {
                    input_node = Some(input.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        label.span(),
                        format!("unknown field: {}", other),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(GraphEffectInput {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing 'name' field"))?,
            nodes,
            audio_wires,
            mod_wires,
            output: output
                .ok_or_else(|| syn::Error::new(input.span(), "missing 'output' field"))?,
            input_node,
        })
    }
}

/// Generate a static audio effect struct from a graph definition.
///
/// # Example
///
/// ```ignore
/// graph_effect! {
///     name: StaticTremolo,
///     nodes: {
///         lfo: PhaseOsc::new(),
///         gain: GainNode::new(1.0),
///     },
///     audio: [input -> gain],
///     modulation: [lfo -> gain.gain(base: 0.5, scale: 0.5)],
///     output: gain,
///     input: input,
/// }
/// ```
#[proc_macro]
pub fn graph_effect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as GraphEffectInput);

    let name = &input.name;

    // Generate struct fields
    let field_defs: Vec<TokenStream2> = input
        .nodes
        .iter()
        .map(|node| {
            let field_name = &node.name;
            let field_type = &node.ty;
            quote! { #field_name: #field_type }
        })
        .collect();

    // Generate field initializers
    let field_inits: Vec<TokenStream2> = input
        .nodes
        .iter()
        .map(|node| {
            let field_name = &node.name;
            let init_expr = &node.init;
            quote! { #field_name: #init_expr }
        })
        .collect();

    // Generate process logic
    // First, determine processing order from audio wires
    let process_steps = generate_process_steps(&input);

    let output_node = &input.output;

    // Generate the struct and impl
    let expanded = quote! {
        pub struct #name {
            #(#field_defs),*
        }

        impl #name {
            pub fn new() -> Self {
                Self {
                    #(#field_inits),*
                }
            }
        }

        impl AudioNode for #name {
            fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
                #process_steps
                self.#output_node.process(signal, ctx)
            }

            fn reset(&mut self) {
                // Reset all nodes
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_process_steps(input: &GraphEffectInput) -> TokenStream2 {
    let mut steps = Vec::new();

    // Initialize signal from input
    steps.push(quote! { let mut signal = input; });

    // Process modulation first (LFOs, envelopes)
    for mod_wire in &input.mod_wires {
        let source = &mod_wire.source;
        let dest = &mod_wire.dest;
        let param = &mod_wire.param;
        let base = &mod_wire.base;
        let scale = &mod_wire.scale;

        let param_setter = format_ident!("set_{}", param);

        steps.push(quote! {
            let mod_val = self.#source.process(0.0, ctx);
            let param_val = #base + mod_val * #scale;
            self.#dest.#param_setter(param_val);
        });
    }

    // Process audio chain
    for wire in &input.audio_wires {
        let source = &wire.source;

        // Check if source is "input" (special case)
        if source == "input" {
            // signal already set from input
            continue;
        }

        steps.push(quote! {
            signal = self.#source.process(signal, ctx);
        });
    }

    quote! { #(#steps)* }
}
