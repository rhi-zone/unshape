# Plugin Architecture

How third-party code extends unshape with custom operations.

## Principle

**Unshape defines the contract. Host handles loading.**

Unshape is a library, not a framework. Different hosts have different needs:
- Game engine: no user plugins, just built-in ops
- DAW: existing Lua scripting system
- Standalone tool: WASM sandbox for security
- Internal tool: native plugins, trusted code

Forcing a plugin system on everyone would conflict with our modular philosophy.

## Layers

```
┌─────────────────────────────────────────────────┐
│  Host Application (DAW, modeler, game, etc.)    │
├─────────────────────────────────────────────────┤
│  Optional: rhi-unshape-wasm-plugins / rhi-unshape-lua / ... │  ← adapters
├─────────────────────────────────────────────────┤
│  rhi-unshape-core: traits + serialization contract    │  ← unshape provides
└─────────────────────────────────────────────────┘
```

## What Unshape Provides

### 1. Op Traits

```rust
/// Core trait for mesh operations
pub trait MeshOp: Send + Sync {
    fn apply(&self, mesh: &Mesh) -> Mesh;
}

/// Similar for other domains
pub trait AudioOp: Send + Sync { ... }
pub trait TextureOp: Send + Sync { ... }
```

### 2. Serialization Contract

Ops must be serializable to/from a common format (JSON, MessagePack, etc.).

**The problem**: `Box<dyn MeshOp>` loses concrete type info. Need to recover type name for serialization.

**Solution**: Trait includes type name + uses `erased_serde`:

```rust
pub trait MeshOp: erased_serde::Serialize + Send + Sync {
    fn apply(&self, mesh: &Mesh) -> Mesh;

    /// Unique type identifier, e.g. "unshape::mesh::Subdivide"
    fn type_name(&self) -> &'static str;
}

// Concrete ops use derive + const
#[derive(Serialize, Deserialize)]
pub struct Subdivide {
    pub levels: u32,
}

impl MeshOp for Subdivide {
    fn apply(&self, mesh: &Mesh) -> Mesh { ... }
    fn type_name(&self) -> &'static str { "unshape::mesh::Subdivide" }
}
```

**Serializing a graph node**:

```rust
fn serialize_node(op: &dyn MeshOp) -> Result<Value> {
    Ok(json!({
        "type": op.type_name(),
        "params": erased_serde::serialize(op, serde_json::value::Serializer)?
    }))
}
```

**Deserializing a graph node**:

```rust
fn deserialize_node(registry: &OpRegistry, value: &Value) -> Result<Box<dyn MeshOp>> {
    let type_name = value["type"].as_str()?;
    let params = &value["params"];
    registry.deserialize(type_name, params.clone())
}
```

### 3. Registry Interface

```rust
/// Type alias for deserialize functions
type DeserializeFn<Op> = Box<dyn Fn(Value) -> Result<Box<Op>> + Send + Sync>;

pub struct OpRegistry<Op: ?Sized> {
    deserializers: HashMap<String, DeserializeFn<Op>>,
}

impl<Op: ?Sized + 'static> OpRegistry<Op> {
    pub fn new() -> Self {
        Self { deserializers: HashMap::new() }
    }

    /// Register a concrete op type
    pub fn register<T>(&mut self)
    where
        T: Op + DeserializeOwned + 'static,
    {
        self.deserializers.insert(
            T::TYPE_NAME.to_string(),  // or use a trait method
            Box::new(|v| Ok(Box::new(serde_json::from_value::<T>(v)?))),
        );
    }

    /// Deserialize an op by type name
    pub fn deserialize(&self, type_name: &str, value: Value) -> Result<Box<Op>> {
        let deserialize = self.deserializers
            .get(type_name)
            .ok_or_else(|| Error::UnknownOp(type_name.to_string()))?;
        deserialize(value)
    }

    /// Check if a type is registered
    pub fn contains(&self, type_name: &str) -> bool {
        self.deserializers.contains_key(type_name)
    }
}
```

## What Host Provides

Plugin loading mechanism. Examples:

### Static Linking (Cargo crates)

Simplest case. User adds plugin crate to `Cargo.toml`, rebuilds app.

```rust
// In host's startup
registry.register::<my_plugin::CustomBevel>("myplugin::CustomBevel");
```

Works with `typetag` for automatic registration if all code is statically linked.

### WASM Plugins (optional adapter)

```rust
// rhi-unshape-wasm-plugins crate
pub struct WasmPluginHost {
    engine: wasmtime::Engine,
    registry: OpRegistry<dyn MeshOp>,
}

impl WasmPluginHost {
    pub fn load_plugin(&mut self, wasm_bytes: &[u8]) -> Result<()> {
        // Instantiate WASM module
        // Call its register() export
        // Wrap WASM functions as trait impls
    }
}
```

### Lua Scripting (optional adapter)

```rust
// rhizome-rhi-unshape-lua-plugins crate
pub struct LuaPluginHost {
    lua: mlua::Lua,
    registry: OpRegistry<dyn MeshOp>,
}

impl LuaPluginHost {
    pub fn load_script(&mut self, source: &str) -> Result<()> {
        // Execute Lua script
        // Script calls register_op(name, apply_fn)
        // Wrap Lua functions as trait impls
    }
}
```

### Native Plugins (C ABI)

For hosts that need maximum performance and trust their plugins:

```rust
// Plugin exposes C ABI
#[no_mangle]
pub extern "C" fn register_ops(registry: *mut OpRegistry) { ... }

// Host loads with dlopen/LoadLibrary
```

## Graph Serialization

Graphs reference ops by type name:

```json
{
  "nodes": [
    { "id": 1, "op": "unshape::mesh::Cube", "params": { "size": [1, 1, 1] } },
    { "id": 2, "op": "unshape::mesh::Subdivide", "params": { "levels": 2 } },
    { "id": 3, "op": "myplugin::CustomBevel", "params": { "amount": 0.1 } }
  ],
  "edges": [[1, 2], [2, 3]]
}
```

Deserialization uses registry to resolve type names to implementations.

If a type name isn't registered, deserialization fails with clear error ("unknown op: myplugin::CustomBevel").

## Summary

| Component | Responsibility |
|-----------|----------------|
| rhi-unshape-core | Op traits, serialization format, registry interface |
| unshape-*-plugins | Optional adapters for common plugin models |
| Host application | Plugin discovery, loading, sandboxing |

This keeps unshape focused and lets hosts make appropriate choices for their context.
