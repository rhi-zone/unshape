//! [`LiveSourceNode`] — wraps a [`LiveSource`] as a zero-input graph node.

use std::any::Any;
use std::sync::Mutex;

use unshape_core::{DynNode, EvalContext, GraphError, PortDescriptor, Value, ValueType};

use crate::source::{LiveCache, LiveSource};

/// A source node whose value arrives on its own schedule instead of being
/// computed on pull, per `docs/design/domain-subsumption.md`'s `LiveSource`
/// section: "a live source is a node whose output changes on its own schedule
/// rather than [being recomputed from wired inputs]."
///
/// Shaped like [`unshape_core::nodes::GraphInput`] (zero inputs, one `"value"`
/// output, optional default) but reads from a [`LiveSource`] instead of
/// `EvalContext`'s named-input map. On every [`execute`](DynNode::execute) the
/// wrapped source is polled; if it has new data the cached value is updated,
/// otherwise the previous cached value (or `default`) is returned — this is
/// the push-to-pull bridge, kept here rather than in `unshape-core` because
/// it's a policy over this node's own staleness, not a core eval-loop change.
pub struct LiveSourceNode<S: LiveSource> {
    /// Host-facing name, used only for error messages (unlike `GraphInput`,
    /// this node's value doesn't come from `EvalContext`'s named-input map).
    pub name: String,
    /// The `Value` type of the `"value"` output port.
    pub value_type: ValueType,
    /// Fallback used when the source has never produced a value.
    pub default: Option<Value>,
    cache: Mutex<LiveCache<S>>,
}

impl<S: LiveSource> LiveSourceNode<S> {
    /// Wraps `source` as a node that reports its output as `value_type`.
    pub fn new(name: impl Into<String>, value_type: ValueType, source: S) -> Self {
        Self {
            name: name.into(),
            value_type,
            default: None,
            cache: Mutex::new(LiveCache::new(source)),
        }
    }

    /// Sets the fallback value used before the source has produced anything.
    pub fn with_default(mut self, default: impl Into<Value>) -> Self {
        self.default = Some(default.into());
        self
    }
}

impl<S> DynNode for LiveSourceNode<S>
where
    S: LiveSource + 'static,
    S::Output: Clone + Into<Value>,
{
    fn type_name(&self) -> &'static str {
        "live::LiveSourceNode"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("value", self.value_type)]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let mut cache = self.cache.lock().expect("live source cache poisoned");
        cache.refresh();

        let value = cache
            .latest()
            .cloned()
            .map(Into::into)
            .or_else(|| self.default.clone())
            .ok_or_else(|| GraphError::MissingInput {
                name: self.name.clone(),
            })?;

        if value.value_type() != self.value_type {
            return Err(GraphError::InputTypeMismatch {
                name: self.name.clone(),
                expected: self.value_type,
                got: value.value_type(),
            });
        }

        Ok(vec![value])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StepSource {
        steps: Vec<f32>,
    }

    impl LiveSource for StepSource {
        type Output = f32;

        fn poll(&mut self) -> Option<f32> {
            if self.steps.is_empty() {
                None
            } else {
                Some(self.steps.remove(0))
            }
        }

        fn has_pending(&self) -> bool {
            !self.steps.is_empty()
        }
    }

    #[test]
    fn node_reports_missing_input_before_any_data() {
        let node = LiveSourceNode::new("x", ValueType::F32, StepSource { steps: vec![] });
        let err = node.execute(&[], &EvalContext::new()).unwrap_err();
        assert!(matches!(err, GraphError::MissingInput { name } if name == "x"));
    }

    #[test]
    fn node_uses_default_before_any_data() {
        let node = LiveSourceNode::new("x", ValueType::F32, StepSource { steps: vec![] })
            .with_default(0.0f32);
        let out = node.execute(&[], &EvalContext::new()).unwrap();
        assert_eq!(out, vec![Value::F32(0.0)]);
    }

    #[test]
    fn node_advances_and_holds_between_pushes() {
        let node = LiveSourceNode::new(
            "x",
            ValueType::F32,
            StepSource {
                steps: vec![1.0, 2.0],
            },
        );
        let out = node.execute(&[], &EvalContext::new()).unwrap();
        assert_eq!(out, vec![Value::F32(1.0)]);
        let out = node.execute(&[], &EvalContext::new()).unwrap();
        assert_eq!(out, vec![Value::F32(2.0)]);
        // Source is drained; the node holds the last observed value.
        let out = node.execute(&[], &EvalContext::new()).unwrap();
        assert_eq!(out, vec![Value::F32(2.0)]);
    }

    #[test]
    fn node_ports() {
        let node = LiveSourceNode::new("x", ValueType::F32, StepSource { steps: vec![] });
        assert!(node.inputs().is_empty());
        let outputs = node.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "value");
        assert_eq!(outputs[0].value_type, ValueType::F32);
    }
}
