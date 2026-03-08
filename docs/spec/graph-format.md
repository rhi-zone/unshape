# Graph Serialization Format

Reference for the `SerialGraph` JSON format produced and consumed by `unshape-serde`.

**Sync requirement:** Update this document when `SerialGraph`, `SerialNode`, or `SerialWire` fields change.

---

## Top-Level Structure

`SerialGraph` serializes to a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer (u32) | Format version. Currently `1`. Old graphs without this field deserialize as version `0` via `#[serde(default)]`. |
| `nodes` | array of `SerialNode` | All nodes in the graph, in insertion order |
| `wires` | array of `SerialWire` | All connections between node ports |
| `next_id` | integer (u32) | Next node ID that will be assigned on deserialization |

`next_id` is required for round-trip fidelity: deserializing and re-serializing a graph must preserve ID assignment so that subsequently added nodes do not collide with existing ones.

## Node Format

Each element of `nodes` is a `SerialNode`:

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer (u32) | Node ID, unique within the graph |
| `type_name` | string | Fully qualified type name used for registry lookup |
| `params` | JSON object | Node parameters, embedded directly as a JSON object |

`params` is a plain JSON object — no double-encoding. The structure of the object is defined by each node type.

**Example node:**

```json
{
  "id": 0,
  "type_name": "test::Const",
  "params": { "value": 2.0 }
}
```

### Type Names

Type names are arbitrary strings chosen by node authors. The convention used in tests is `"namespace::TypeName"` (e.g., `"test::Const"`, `"test::Add"`). There is no enforced naming scheme beyond being a non-empty string that matches what was registered in the `NodeRegistry`.

When using `NodeRegistry::register<N>()` (without an explicit name), the type name is `std::any::type_name::<N>()`, which is the Rust fully-qualified path. When using `register_with_name`, the caller controls the string.

## Wire Format

Each element of `wires` is a `SerialWire` with two string fields:

| Field | Type | Description |
|-------|------|-------------|
| `from` | string | Source endpoint: `"nodeId:portName"` |
| `to` | string | Destination endpoint: `"nodeId:portName"` |

Port names come from `DynNode::output_port_names()` and `DynNode::input_port_names()`, which by default derive from the `name` field of each `PortDescriptor` returned by `outputs()` and `inputs()`.

**Example wire** (connecting the `value` output of node 0 to the `b` input of node 2):

```json
{ "from": "0:value", "to": "2:b" }
```

The `"nodeId:portName"` format is more readable than numeric indices and is refactoring-safe as long as port names don't change. Names are resolved back to numeric indices during deserialization by scanning the node's port descriptor list.

## Complete Example

A three-node graph: two constant nodes feeding an add node.

```
Const(2.0) --[value]--> Add(a) --[result]
Const(3.0) --[value]--> Add(b)
```

```json
{
  "version": 1,
  "nodes": [
    {
      "id": 0,
      "type_name": "test::Const",
      "params": { "value": 2.0 }
    },
    {
      "id": 1,
      "type_name": "test::Const",
      "params": { "value": 3.0 }
    },
    {
      "id": 2,
      "type_name": "test::Add",
      "params": {}
    }
  ],
  "wires": [
    { "from": "0:value", "to": "2:a" },
    { "from": "1:value", "to": "2:b" }
  ],
  "next_id": 3
}
```

## Node Registry

`NodeRegistry` maps type name strings to factory functions that reconstruct `Box<dyn DynNode>` from a `serde_json::Value`.

Three registration methods are available:

| Method | When to use |
|--------|-------------|
| `register<N>()` | `N` implements `DeserializeOwned`; uses `std::any::type_name::<N>()` as the key |
| `register_with_name<N>(name)` | `N` implements `DeserializeOwned`; caller controls the key string |
| `register_factory(name, fn)` | Custom deserialization logic (e.g., type has extra construction steps) |

During deserialization, `registry.deserialize(type_name, params_value)` is called for each node. If the type name is not registered, deserialization returns `SerdeError::UnknownNodeType`.

### Making a Node Serializable

A node that implements `DynNode` must also implement `SerializableNode` (from `unshape-serde`) to participate in serialization:

```rust
impl SerializableNode for MyNode {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}
```

The returned value is stored as the `params` field. For symmetric round-trips, it must be deserializable by whatever factory was registered for this node's type name.

## Format Variants

### JSON

`JsonFormat` serializes `SerialGraph` to UTF-8 JSON bytes using `serde_json`. Params are embedded as native JSON objects — no double-encoding.

| Mode | Construction | Notes |
|------|--------------|-------|
| Compact | `JsonFormat::new()` | No whitespace; smallest file size |
| Pretty | `JsonFormat::pretty()` | Indented; git-diffable |
| File extension | `.json` | |

Both modes produce identical `SerialGraph` values on deserialization.

### Bincode

`BincodeFormat` serializes `SerialGraph` to binary using `bincode` with the standard configuration (`bincode::config::standard()`).

| Property | Value |
|----------|-------|
| Encoding | Bincode standard config |
| File extension | `.bin` |
| Human-readable | No |
| Size | Smaller than JSON for graphs with many nodes |

Internally, the bincode path converts `params` (a `serde_json::Value`) to a JSON string before encoding, because bincode's serde bridge does not support the `any` data model required by `serde_json::Value`. This conversion happens transparently in `BincodeFormat::serialize` / `deserialize` — the `SerialGraph` API is the same in both paths.

### GraphFormat Trait

Both formats implement `GraphFormat`:

```rust
pub trait GraphFormat: Send + Sync {
    fn serialize(&self, graph: &SerialGraph) -> Result<Vec<u8>, SerdeError>;
    fn deserialize(&self, bytes: &[u8]) -> Result<SerialGraph, SerdeError>;
    fn name(&self) -> &'static str;
    fn extension(&self) -> &'static str;
}
```

Custom formats can implement this trait and pass them to `serialize_graph` / `deserialize_graph`.

## High-Level API

```rust
// Serialize a Graph to bytes
let bytes = serialize_graph(&graph, extract_params_fn, &format)?;

// Deserialize bytes back to a Graph
let graph = deserialize_graph(&bytes, &registry, &format)?;
```

`extract_params_fn` is a `Fn(&dyn DynNode) -> Option<serde_json::Value>`. The typical implementation downcasts the node to its concrete type (via `SerializableNode::params()`) and returns its JSON representation. Returning `None` causes `SerdeError::NotSerializable`.

For the intermediate step:

```rust
// Graph → SerialGraph
let serial = graph_to_serial(&graph, extract_params_fn)?;

// SerialGraph → Graph
let graph = serial_to_graph(serial, &registry)?;
```

## Error Cases

| Error | Cause |
|-------|-------|
| `SerdeError::UnknownNodeType(name)` | Type name not in registry during deserialization |
| `SerdeError::NotSerializable(name)` | `extract_params_fn` returned `None` for this node |
| `SerdeError::Json(e)` | JSON parse/encode failure |
| `SerdeError::Bincode(e)` | Bincode decode failure |
| `SerdeError::BincodeEncode(e)` | Bincode encode failure |
| `SerdeError::Graph(e)` | Graph reconstruction failed (e.g., invalid wire referencing missing node) |
| `SerdeError::InvalidWireFormat(msg)` | Wire endpoint string is not valid `"nodeId:portName"`, or the named port does not exist on the node |

## Versioning

`SerialGraph` includes a `version: u32` field (currently `1`). Graphs serialized without this field (version 0 / pre-stable format) deserialize with `version = 0` via `#[serde(default)]`. There is no automatic migration — consumers can inspect `version` and handle differences explicitly.

## What Is Not Serialized

- **`Value::Opaque`** — opaque values (meshes, images, GPU buffers) are skipped by serde (`#[serde(skip)]`). Nodes that produce or consume opaque values cannot be serialized through this format directly; they must use domain-specific serialization for their data.
- **Port descriptors** — port names and types are not stored; they are reconstructed by calling `inputs()` and `outputs()` on the deserialized node.
- **Evaluation cache** — `LazyEvaluator` cache state is not serialized.
- **Topological order cache** — recomputed on first execution after deserialization.
