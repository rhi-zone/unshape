//! Extracts ops-as-values structs from unshape crates.
//!
//! Run from repo root:
//!   `cargo run -p extract-ops`              - JSON output (full)
//!   `cargo run -p extract-ops -- --md`      - Markdown output (full)
//!   `cargo run -p extract-ops -- --names`   - Names only (compact)
//!
//! Redirect to file:
//!   `cargo run -p extract-ops > ops-reference.json`
//!   `cargo run -p extract-ops -- --md > docs/ops-reference.md`
//!   `cargo run -p extract-ops -- --names > docs/ops-index.md`

use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use syn::{Attribute, Fields, Item, ItemImpl, ItemStruct, Lit, Meta, Type};
use walkdir::WalkDir;

#[derive(Debug, Serialize)]
struct OpInfo {
    name: String,
    doc: Option<String>,
    fields: Vec<FieldInfo>,
    input_type: Option<String>,
    output_type: Option<String>,
    file: String,
    line: usize,
}

#[derive(Debug, Serialize)]
struct FieldInfo {
    name: String,
    ty: String,
    doc: Option<String>,
}

#[derive(Debug, Serialize)]
struct CrateOps {
    crate_name: String,
    ops: Vec<OpInfo>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let markdown_mode = args.iter().any(|a| a == "--md" || a == "--markdown");
    let names_only = args.iter().any(|a| a == "--names");

    let crates_dir = Path::new("crates");
    if !crates_dir.exists() {
        eprintln!("Run from repo root (crates/ directory not found)");
        std::process::exit(1);
    }

    let mut all_crates: BTreeMap<String, Vec<OpInfo>> = BTreeMap::new();

    for entry in WalkDir::new(crates_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file = match syn::parse_file(&content) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Extract crate name from path
        let crate_name = path
            .components()
            .nth(1)
            .and_then(|c| c.as_os_str().to_str())
            .unwrap_or("unknown")
            .to_string();

        let relative_path = path.to_string_lossy().to_string();

        // Find all structs with apply methods
        let structs_with_apply = find_ops(&file, &relative_path);

        if !structs_with_apply.is_empty() {
            all_crates
                .entry(crate_name)
                .or_default()
                .extend(structs_with_apply);
        }
    }

    // Convert to output format
    let output: Vec<CrateOps> = all_crates
        .into_iter()
        .map(|(crate_name, mut ops)| {
            ops.sort_by(|a, b| a.name.cmp(&b.name));
            CrateOps { crate_name, ops }
        })
        .collect();

    if names_only {
        print_names_only(&output);
    } else if markdown_mode {
        print_markdown(&output);
    } else {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    }
}

fn print_names_only(crates: &[CrateOps]) {
    println!("# Ops Index");
    println!();
    let total_ops: usize = crates.iter().map(|c| c.ops.len()).sum();
    println!("{} ops across {} crates.", total_ops, crates.len());
    println!();
    println!("Regenerate: `cargo run -p extract-ops -- --names > docs/ops-index.md`");
    println!();

    for c in crates {
        println!("## {}", c.crate_name);
        println!();
        for op in &c.ops {
            println!("- `{}`", op.name);
        }
        println!();
    }
}

fn print_markdown(crates: &[CrateOps]) {
    println!("# Ops Reference");
    println!();
    println!("Auto-generated list of all ops-as-values structs in unshape.");
    println!();
    println!("Regenerate with: `cargo run -p extract-ops -- --md > docs/ops-reference.md`");
    println!();

    // Table of contents
    println!("## Contents");
    println!();
    let total_ops: usize = crates.iter().map(|c| c.ops.len()).sum();
    println!("**{} ops across {} crates**", total_ops, crates.len());
    println!();
    for c in crates {
        println!(
            "- [{}](#{}): {} ops",
            c.crate_name,
            c.crate_name.replace('-', "_"),
            c.ops.len()
        );
    }
    println!();

    // Each crate
    for c in crates {
        println!("---");
        println!();
        println!("## {}", c.crate_name);
        println!();

        for op in &c.ops {
            println!("### `{}`", op.name);
            println!();

            if let Some(doc) = &op.doc {
                // Take first paragraph as summary
                let summary = doc.split("\n\n").next().unwrap_or(doc);
                println!("{}", summary);
                println!();
            }

            // Signature
            let input = op.input_type.as_deref().unwrap_or("()");
            let output = op.output_type.as_deref().unwrap_or("()");
            println!("`apply({}) -> {}`", input, output);
            println!();

            // Fields table
            if !op.fields.is_empty() {
                println!("| Field | Type | Description |");
                println!("|-------|------|-------------|");
                for f in &op.fields {
                    let desc = f
                        .doc
                        .as_ref()
                        .map(|d| d.lines().next().unwrap_or(""))
                        .unwrap_or("");
                    println!("| `{}` | `{}` | {} |", f.name, f.ty, desc);
                }
                println!();
            }

            // Source link
            println!(
                "*Source: [{}:{}]({}#L{})*",
                op.file, op.line, op.file, op.line
            );
            println!();
        }
    }
}

fn find_ops(file: &syn::File, file_path: &str) -> Vec<OpInfo> {
    let mut ops = Vec::new();

    // Collect all struct definitions
    let mut structs: BTreeMap<String, (ItemStruct, usize)> = BTreeMap::new();
    for item in &file.items {
        if let Item::Struct(s) = item {
            let line = estimate_line(file_path, &s.ident.to_string());
            structs.insert(s.ident.to_string(), (s.clone(), line));
        }
    }

    // Find impl blocks with apply methods
    for item in &file.items {
        if let Item::Impl(impl_block) = item {
            if let Some((struct_name, input_type, output_type)) = has_apply_method(impl_block) {
                if let Some((struct_def, line)) = structs.get(&struct_name) {
                    let doc = extract_doc_comment(&struct_def.attrs);
                    let fields = extract_fields(&struct_def.fields);

                    ops.push(OpInfo {
                        name: struct_name,
                        doc,
                        fields,
                        input_type,
                        output_type,
                        file: file_path.to_string(),
                        line: *line,
                    });
                }
            }
        }
    }

    ops
}

fn has_apply_method(impl_block: &ItemImpl) -> Option<(String, Option<String>, Option<String>)> {
    // Get the struct name from the impl
    let struct_name = match &*impl_block.self_ty {
        Type::Path(p) => p.path.segments.last()?.ident.to_string(),
        _ => return None,
    };

    // Skip trait impls
    if impl_block.trait_.is_some() {
        return None;
    }

    // Look for apply method
    for item in &impl_block.items {
        if let syn::ImplItem::Fn(method) = item {
            if method.sig.ident == "apply" {
                // Extract input type from first non-self parameter
                let input_type = method
                    .sig
                    .inputs
                    .iter()
                    .filter_map(|arg| {
                        if let syn::FnArg::Typed(pat) = arg {
                            Some(type_to_string(&pat.ty))
                        } else {
                            None
                        }
                    })
                    .next();

                // Extract output type
                let output_type = match &method.sig.output {
                    syn::ReturnType::Type(_, ty) => Some(type_to_string(ty)),
                    syn::ReturnType::Default => None,
                };

                return Some((struct_name, input_type, output_type));
            }
        }
    }

    None
}

fn extract_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(s) = &expr_lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

fn extract_fields(fields: &Fields) -> Vec<FieldInfo> {
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .map(|f| FieldInfo {
                name: f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default(),
                ty: type_to_string(&f.ty),
                doc: extract_doc_comment(&f.attrs),
            })
            .collect(),
        Fields::Unnamed(unnamed) => unnamed
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| FieldInfo {
                name: format!("{}", i),
                ty: type_to_string(&f.ty),
                doc: extract_doc_comment(&f.attrs),
            })
            .collect(),
        Fields::Unit => vec![],
    }
}

fn type_to_string(ty: &Type) -> String {
    quote::quote!(#ty).to_string().replace(" ", "")
}

fn estimate_line(file_path: &str, struct_name: &str) -> usize {
    // Simple heuristic - read file and find the struct definition
    if let Ok(content) = fs::read_to_string(file_path) {
        for (i, line) in content.lines().enumerate() {
            if line.contains(&format!("struct {}", struct_name))
                || line.contains(&format!("struct {}<", struct_name))
            {
                return i + 1;
            }
        }
    }
    0
}
