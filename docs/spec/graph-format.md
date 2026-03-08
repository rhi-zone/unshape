# Graph Serialization Format

Reference for the `SerialGraph` JSON format produced and consumed by `unshape-serde`.

**Sync requirement:** Update this document when `SerialGraph`, `SerialNode`, or `Wire` fields change.

---

## Top-Level Structure

`SerialGraph` serializes to a JSON object with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | array of `SerialNode` | All nodes in the graph, in insertion order |
| `wires` | array of `Wire` | All connections between node ports |
| `next_id` | integer (u32) | Next node ID that will be assigned on deserialization |

`next_id` is required for round-trip fidelity: deserializing and re-serializing a graph must preserve ID assignment so that subsequently added nodes do not collide with existing ones.

## Node Format

Each element of `nodes` is a `SerialNode`:

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer (u32) | Node ID, unique within the graph |
| `type_name` | string | Fully qualified type name used for registry lookup |
| `params_json` | string | Node parameters, JSON-encoded as a **string** |

`params_json` is a JSON string whose value is itself a JSON object. This double-encoding is intentional: it allows bincode to serialize the params field without requiring bincode to understand arbitrary JSON values. The inner JSON structure is defined by each node type.

**Example node:**

```json
{
  "id": 0,
  "type_name": "test::Const",
  "params_json": "{\"value\":2.0}"
}
```

### Type Names

Type names are arbitrary strings chosen by node authors. The convention used in tests is `"namespace::TypeName"` (e.g., `"test::Const"`, `"test::Add"`). There is no enforced naming scheme beyond being a non-empty string that matches what was registered in the `NodeRegistry`.

When using `NodeRegistry::register<N>()` (without an explicit name), the type name is `std::any::type_name::<N>()`, which is the Rust fully-qualified path. When using `register_with_name`, the caller controls the string.

## Wire Format

Each element of `wires` is a `Wire`:

| Field | Type | Description |
|-------|------|-------------|
| `from_node` | integer (u32) | Source node ID |
| `from_port` | integer (usize) | Output port index on the source node |
| `to_node` | integer (u32) | Destination node ID |
| `to_port` | integer (usize) | Input port index on the destination node |

Ports are zero-indexed and correspond to the order returned by the node's `outputs()` and `inputs()` methods respectively.

**Example wire** (connecting output port 0 of node 0 to input port 1 of node 2):

```json
{
  "from_node": 0,
  "from_port": 0,
  "to_node": 2,
  "to_port": 1
}
```

## Complete Example

A three-node graph: two constant nodes feeding an add node.

```
Const(2.0) --[port 0]--> Add --[output]
Const(3.0) --[port 1]--> Add
```

```json
{
  "nodes": [
    {
      "id": 0,
      "type_name": "test::Const",
      "params_json": "{\"value\":2.0}"
    },
    {
      "id": 1,
      "type_name": "test::Const",
      "params_json": "{\"value\":3.0}"
    },
    {
      "id": 2,
      "type_name": "test::Add",
      "params_json": "{}"
    }
  ],
  "wires": [
    { "from_node": 0, "from_port": 0, "to_node": 2, "to_port": 0 },
    { "from_node": 1, "from_port": 0, "to_node": 2, "to_port": 1 }
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

During deserialization, `registry.deserialize(type_name, params_json_value)` is called for each node. If the type name is not registered, deserialization returns `SerdeError::UnknownNodeType`.

### Making a Node Serializable

A node that implements `DynNode` must also implement `SerializableNode` (from `unshape-serde`) to participate in serialization:

```rust
impl SerializableNode for MyNode {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}
```

The returned value is what gets stored in `params_json`. For symmetric round-trips, it must be deserializable by whatever factory was registered for this node's type name.

## Format Variants

### JSON

`JsonFormat` serializes `SerialGraph` to UTF-8 JSON bytes using `serde_json`.

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

`params_json` is stored as a length-prefixed string in bincode. This is why params are stored as a JSON string rather than a structured value — bincode cannot encode `serde_json::Value` directly.

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

## Versioning

There is no version field in `SerialGraph`. The format has no built-in versioning or schema migration mechanism. If the format changes in a breaking way, consumers must handle this externally (e.g., by file extension convention or wrapper envelope).

## What Is Not Serialized

- **`Value::Opaque`** — opaque values (meshes, images, GPU buffers) are skipped by serde (`#[serde(skip)]`). Nodes that produce or consume opaque values cannot be serialized through this format directly; they must use domain-specific serialization for their data.
- **Port descriptors** — port names and types are not stored; they are reconstructed by calling `inputs()` and `outputs()` on the deserialized node.
- **Evaluation cache** — `LazyEvaluator` cache state is not serialized.
- **Topological order cache** — recomputed on first execution after deserialization.
