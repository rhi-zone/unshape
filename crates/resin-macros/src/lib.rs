//! Derive macros for resin.
//!
//! Provides `#[derive(DynNode)]` for implementing the DynNode trait.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Ident, Type, parse_macro_input};

/// Information about an input or output field.
struct FieldInfo {
    name: Ident,
    ty: Type,
    port_name: String,
}

/// Determines the ValueType for a Rust type.
fn value_type_for(ty: &Type, crate_path: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let ty_str = quote!(#ty).to_string().replace(' ', "");
    match ty_str.as_str() {
        "f32" => quote!(#crate_path::ValueType::F32),
        "f64" => quote!(#crate_path::ValueType::F64),
        "i32" => quote!(#crate_path::ValueType::I32),
        "bool" => quote!(#crate_path::ValueType::Bool),
        "Vec2" | "glam::Vec2" => quote!(#crate_path::ValueType::Vec2),
        "Vec3" | "glam::Vec3" => quote!(#crate_path::ValueType::Vec3),
        "Vec4" | "glam::Vec4" => quote!(#crate_path::ValueType::Vec4),
        _ => quote!(#crate_path::ValueType::F32), // fallback
    }
}

/// Generates the conversion from Value to the field type.
fn value_extract_for(
    ty: &Type,
    idx: usize,
    _crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let ty_str = quote!(#ty).to_string().replace(' ', "");
    match ty_str.as_str() {
        "f32" => quote!(inputs[#idx].as_f32().unwrap_or_default()),
        "f64" => quote!(inputs[#idx].as_f64().unwrap_or_default()),
        "i32" => quote!(inputs[#idx].as_i32().unwrap_or_default()),
        "bool" => quote!(inputs[#idx].as_bool().unwrap_or_default()),
        "Vec2" | "glam::Vec2" => quote!(inputs[#idx].as_vec2().unwrap_or_default()),
        "Vec3" | "glam::Vec3" => quote!(inputs[#idx].as_vec3().unwrap_or_default()),
        "Vec4" | "glam::Vec4" => quote!(inputs[#idx].as_vec4().unwrap_or_default()),
        _ => quote!(inputs[#idx].as_f32().unwrap_or_default()),
    }
}

/// Generates the conversion from a field to Value.
fn value_wrap_for(
    ty: &Type,
    expr: proc_macro2::TokenStream,
    crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let ty_str = quote!(#ty).to_string().replace(' ', "");
    match ty_str.as_str() {
        "f32" => quote!(#crate_path::Value::F32(#expr)),
        "f64" => quote!(#crate_path::Value::F64(#expr)),
        "i32" => quote!(#crate_path::Value::I32(#expr)),
        "bool" => quote!(#crate_path::Value::Bool(#expr)),
        "Vec2" | "glam::Vec2" => quote!(#crate_path::Value::Vec2(#expr)),
        "Vec3" | "glam::Vec3" => quote!(#crate_path::Value::Vec3(#expr)),
        "Vec4" | "glam::Vec4" => quote!(#crate_path::Value::Vec4(#expr)),
        _ => quote!(#crate_path::Value::F32(#expr as f32)),
    }
}

/// Derive macro for implementing DynNode.
///
/// # Usage
///
/// ```ignore
/// use rhizome_resin_macros::DynNode;
/// use rhizome_resin_core::Value;
///
/// #[derive(DynNode, Default)]
/// struct AddNode {
///     #[input]
///     a: f32,
///     #[input]
///     b: f32,
///     #[output]
///     result: f32,
/// }
///
/// impl AddNode {
///     fn compute(&mut self) {
///         self.result = self.a + self.b;
///     }
/// }
/// ```
///
/// Use `#[node(crate = "crate")]` when deriving inside resin-core itself.
///
/// The macro generates a DynNode implementation that:
/// - Returns the struct name as type_name
/// - Creates PortDescriptors for `#[input]` and `#[output]` fields
/// - In execute(), extracts inputs, calls compute(), and returns outputs
#[proc_macro_derive(DynNode, attributes(input, output, node))]
pub fn derive_dyn_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let type_name_str = name.to_string();

    // Parse #[node(crate = "...")] attribute
    let crate_path: proc_macro2::TokenStream = input
        .attrs
        .iter()
        .find(|a| a.path().is_ident("node"))
        .and_then(|attr| {
            attr.parse_args::<syn::MetaNameValue>()
                .ok()
                .and_then(|meta| {
                    if meta.path.is_ident("crate")
                        && let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(s),
                            ..
                        }) = meta.value
                    {
                        let path = s.value();
                        return Some(if path == "crate" {
                            quote!(crate)
                        } else {
                            let ident = syn::Ident::new(&path, proc_macro2::Span::call_site());
                            quote!(#ident)
                        });
                    }
                    None
                })
        })
        .unwrap_or_else(|| quote!(::resin_core));

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("DynNode can only be derived for structs with named fields"),
        },
        _ => panic!("DynNode can only be derived for structs"),
    };

    let mut inputs: Vec<FieldInfo> = Vec::new();
    let mut outputs: Vec<FieldInfo> = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap().clone();
        let field_ty = field.ty.clone();
        let port_name = field_name.to_string();

        let is_input = field.attrs.iter().any(|a| a.path().is_ident("input"));
        let is_output = field.attrs.iter().any(|a| a.path().is_ident("output"));

        if is_input {
            inputs.push(FieldInfo {
                name: field_name.clone(),
                ty: field_ty.clone(),
                port_name: port_name.clone(),
            });
        }
        if is_output {
            outputs.push(FieldInfo {
                name: field_name,
                ty: field_ty,
                port_name,
            });
        }
    }

    // Generate input descriptors
    let input_descriptors = inputs.iter().map(|f| {
        let port_name = &f.port_name;
        let value_type = value_type_for(&f.ty, &crate_path);
        quote! {
            #crate_path::PortDescriptor::new(#port_name, #value_type)
        }
    });

    // Generate output descriptors
    let output_descriptors = outputs.iter().map(|f| {
        let port_name = &f.port_name;
        let value_type = value_type_for(&f.ty, &crate_path);
        quote! {
            #crate_path::PortDescriptor::new(#port_name, #value_type)
        }
    });

    // Generate input extraction
    let input_extractions = inputs.iter().enumerate().map(|(i, f)| {
        let field_name = &f.name;
        let extraction = value_extract_for(&f.ty, i, &crate_path);
        quote! {
            node.#field_name = #extraction;
        }
    });

    // Generate output collection
    let output_collections = outputs.iter().map(|f| {
        let field_name = &f.name;
        let wrap = value_wrap_for(&f.ty, quote!(node.#field_name), &crate_path);
        quote! {
            #wrap
        }
    });

    let expanded = quote! {
        impl #crate_path::DynNode for #name {
            fn type_name(&self) -> &'static str {
                #type_name_str
            }

            fn inputs(&self) -> Vec<#crate_path::PortDescriptor> {
                vec![#(#input_descriptors),*]
            }

            fn outputs(&self) -> Vec<#crate_path::PortDescriptor> {
                vec![#(#output_descriptors),*]
            }

            fn execute(
                &self,
                inputs: &[#crate_path::Value],
                _ctx: &#crate_path::EvalContext,
            ) -> Result<Vec<#crate_path::Value>, #crate_path::GraphError> {
                let mut node = self.clone();
                #(#input_extractions)*
                node.compute();
                Ok(vec![#(#output_collections),*])
            }

            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }
    };

    TokenStream::from(expanded)
}
