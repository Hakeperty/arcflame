#![allow(dead_code)]

/// Represents a range of layers assigned to this node
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LayerRange {
    pub first_layer: i32,
    pub num_layers: i32,
    pub has_lm_head: bool,
}

impl LayerRange {
    pub fn new(first_layer: i32, num_layers: i32, has_lm_head: bool) -> Self {
        Self { first_layer, num_layers, has_lm_head }
    }

    pub fn last_layer(&self) -> i32 {
        self.first_layer + self.num_layers - 1
    }
}

/// Parse tensor names to determine which layer they belong to
/// GGUF tensor names follow the pattern: "blk.N.xxxx" where N is layer number
#[allow(dead_code)]
pub fn layer_from_tensor_name(name: &str) -> Option<i32> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        rest[..dot_pos].parse::<i32>().ok()
    } else {
        // Non-layer tensors: token embd, output weight, output norm
        None
    }
}

/// Check if a tensor belongs to this node's layer range
#[allow(dead_code)]
pub fn tensor_belongs_to_node(tensor_name: &str, range: &LayerRange) -> bool {
    if let Some(layer) = layer_from_tensor_name(tensor_name) {
        layer >= range.first_layer && layer <= range.last_layer()
    } else {
        // Non-layer tensors go to whichever node has the LM head
        tensor_name.contains("output") && range.has_lm_head
    }
}
