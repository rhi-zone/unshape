//! Proc macros for resin-op.
//!
//! Provides `#[derive(Op)]` to auto-generate `DynOp` implementations.
//!
//! # Example
//!
//! ```ignore
//! use resin_op_macros::Op;
//!
//! #[derive(Clone, Serialize, Deserialize, Op)]
//! #[op(input = Mesh, output = Mesh)]
//! pub struct Subdivide {
//!     pub levels: u32,
//! }
//!
//! impl Subdivide {
//!     pub fn apply(&self, mesh: &Mesh) -> Mesh {
//!         // implementation
//!     }
//! }
//! ```
//!
//! The macro generates:
//! - `DynOp::type_name()` from struct name (or `#[op(name = "...")]`)
//! - `DynOp::input_type()` / `output_type()` from `#[op(input, output)]`
//! - `DynOp::apply_dyn()` that calls `self.apply()` with type conversion
//! - `DynOp::params()` using serde

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, LitStr, Token, Type, TypeTuple, parse_macro_input};

/// Derives `DynOp` for an operation struct.
///
/// # Attributes
///
/// - `#[op(input = Type)]` - Input type (required). Can be any type.
/// - `#[op(output = Type)]` - Output type (required). Can be any type.
/// - `#[op(name = "...")]` - Custom type name (optional, defaults to resin::StructName)
///
/// # Requirements
///
/// The struct must:
/// - Implement `Clone` and `serde::{Serialize, Deserialize}`
/// - Have an `apply(&self, input: &InputType) -> OutputType` method
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, Serialize, Deserialize, Op)]
/// #[op(input = Mesh, output = Mesh)]
/// pub struct Subdivide { pub levels: u32 }
///
/// impl Subdivide {
///     pub fn apply(&self, mesh: &Mesh) -> Mesh { ... }
/// }
/// ```
#[proc_macro_derive(Op, attributes(op))]
pub fn derive_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Parse #[op(...)] attributes
    let mut input_type: Option<Type> = None;
    let mut output_type: Option<Type> = None;
    let mut custom_name: Option<String> = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("op") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("input") {
                meta.input.parse::<Token![=]>()?;
                input_type = Some(meta.input.parse::<Type>()?);
                Ok(())
            } else if meta.path.is_ident("output") {
                meta.input.parse::<Token![=]>()?;
                output_type = Some(meta.input.parse::<Type>()?);
                Ok(())
            } else if meta.path.is_ident("name") {
                meta.input.parse::<Token![=]>()?;
                let lit: LitStr = meta.input.parse()?;
                custom_name = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("unknown op attribute"))
            }
        })
        .expect("failed to parse op attribute");
    }

    let input_type = input_type.expect("#[op(input = Type)] is required");
    let output_type = output_type.expect("#[op(output = Type)] is required");

    // Generate type name
    let type_name_str = custom_name.unwrap_or_else(|| format!("resin::{}", struct_name));

    // Type names for display (use the token representation)
    let input_type_name = input_type.to_token_stream().to_string();
    let output_type_name = output_type.to_token_stream().to_string();

    // Check if input type is unit type ()
    let is_unit_input = match &input_type {
        Type::Tuple(TypeTuple { elems, .. }) => elems.is_empty(),
        _ => false,
    };

    // Generate the apply_dyn body based on whether input is unit type
    let apply_dyn_body = if is_unit_input {
        quote! {
            let _: () = input.downcast()?;
            let result = self.apply();
            ::std::result::Result::Ok(::unshape_op::OpValue::new(
                ::unshape_op::OpType::of::<#output_type>(#output_type_name),
                result
            ))
        }
    } else {
        quote! {
            let value: #input_type = input.downcast()?;
            let result = self.apply(&value);
            ::std::result::Result::Ok(::unshape_op::OpValue::new(
                ::unshape_op::OpType::of::<#output_type>(#output_type_name),
                result
            ))
        }
    };

    // Generate the DynOp impl
    let expanded = quote! {
        impl #impl_generics ::unshape_op::DynOp for #struct_name #ty_generics #where_clause {
            fn type_name(&self) -> &'static str {
                #type_name_str
            }

            fn input_type(&self) -> ::unshape_op::OpType {
                ::unshape_op::OpType::of::<#input_type>(#input_type_name)
            }

            fn output_type(&self) -> ::unshape_op::OpType {
                ::unshape_op::OpType::of::<#output_type>(#output_type_name)
            }

            fn apply_dyn(&self, input: ::unshape_op::OpValue) -> ::std::result::Result<::unshape_op::OpValue, ::unshape_op::OpError> {
                #apply_dyn_body
            }

            fn params(&self) -> ::serde_json::Value {
                ::serde_json::to_value(self).unwrap_or_default()
            }
        }
    };

    TokenStream::from(expanded)
}
