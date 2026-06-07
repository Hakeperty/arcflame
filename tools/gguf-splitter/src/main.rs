//! GGUF Splitter — Splits a GGUF model file into per-layer shards
//! for distributed inference across ArcFlare nodes.

use clap::Parser;
use std::collections::HashMap;
use std::io::{Seek, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gguf-splitter")]
struct Args {
    /// Path to the source GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Number of layers per shard (default: auto-calculate from model metadata)
    #[arg(short = 'n', long)]
    layers_per_shard: Option<u32>,

    /// Output directory for shards
    #[arg(short, long, default_value = "./shards")]
    output_dir: PathBuf,

    /// JSON file with node capabilities for optimal splitting
    #[arg(short, long)]
    node_config: Option<PathBuf>,

    /// List all tensor names in the model and exit
    #[arg(long)]
    list_tensors: bool,

    /// Actually perform the split and write shard GGUF files
    #[arg(long)]
    split: bool,

    /// Overwrite existing shard files without asking
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.model.exists() {
        anyhow::bail!("Model file not found: {}", args.model.display());
    }

    let file = std::fs::File::open(&args.model)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let data = &mmap[..];
    let file_size = data.len();

    if data.len() < 24 {
        anyhow::bail!("File too small for valid GGUF");
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != 0x46554747 {
        anyhow::bail!("Not a valid GGUF file (magic: {:#x})", magic);
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

    let mut metadata_end = 24usize;
    for _ in 0..metadata_kv_count {
        let key_len = read_u64(data, &mut metadata_end)?;
        metadata_end += key_len as usize;
        let value_type = read_u32(data, &mut metadata_end)?;
        skip_value(data, &mut metadata_end, value_type)?;
    }

    // The first tensor info immediately follows the metadata section.
    // Search a small window for the correct start — some GGUF writers
    // store metadata in a way that our skip_value may overshoot or
    // the format includes alignment padding.
    let tensor_info_start = {
        let mut best = metadata_end;
        for candidate in (best.saturating_sub(32)..best + 32).step_by(4) {
            if candidate + 8 > data.len() { continue; }
            let nl = u64::from_le_bytes(data[candidate..candidate+8].try_into().unwrap());
            if nl > 0 && nl < 256 && candidate + 8 + nl as usize <= data.len() {
                let nm = String::from_utf8_lossy(&data[candidate+8..candidate+8+nl as usize]);
                // Valid tensor names contain '.' or known suffixes
                if nm.contains('.') || nm.contains("weight") || nm.contains("bias")
                   || nm.contains("norm") || nm.contains("embd") || nm.contains("rope")
                   || nm.contains("freq") || nm.contains("tok_embd")
                {
                    best = candidate;
                    break;
                }
            }
        }
        best
    };

    let mut offset = tensor_info_start;
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name_len = read_u64(data, &mut offset)?;
        if offset + name_len as usize > data.len() {
            anyhow::bail!("Tensor name overflows file at offset {}", offset);
        }
        let name = String::from_utf8(data[offset..offset + name_len as usize].to_vec())?;
        offset += name_len as usize;

        let n_dims = read_u32(data, &mut offset)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(data, &mut offset)?);
        }

        let ggml_type = read_u32(data, &mut offset)?;
        let tensor_offset = read_u64(data, &mut offset)?;

        tensors.push(TensorInfo {
            name,
            n_dims,
            dims,
            ggml_type,
            offset: tensor_offset,
        });
    }

    let tensor_info_end = offset;
    let tensor_data_start = align_up(tensor_info_end, 32);

    let num_layers = count_layers(&tensors);

    if args.list_tensors {
        for t in &tensors {
            println!("{}", t.name);
        }
        return Ok(());
    }

    let architecture = get_metadata_string(data, 24, metadata_kv_count, "general.architecture")
        .unwrap_or_else(|| "unknown".to_string());
    let model_name = std::path::Path::new(&args.model)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());

    println!("Model: {}", model_name);
    println!("Architecture: {}", architecture);
    println!("Total layers: {}", num_layers);
    println!("Tensor count: {}", tensor_count);
    println!("File size: {:.2} GB", file_size as f64 / (1024.0 * 1024.0 * 1024.0));

    let layers_per_shard = if let Some(n) = args.layers_per_shard {
        n
    } else if let Some(config_path) = &args.node_config {
        calculate_from_node_config(config_path)?
    } else {
        (num_layers as f64 / 4.0).ceil() as u32
    };

    if layers_per_shard == 0 {
        anyhow::bail!("layers_per_shard must be greater than 0");
    }

    let num_shards = (num_layers as f64 / layers_per_shard as f64).ceil() as u32;

    println!("\nSplit plan:");
    println!("  Layers per shard: {}", layers_per_shard);
    println!("  Total shards: {}", num_shards);

    for i in 0..num_shards {
        let first = i * layers_per_shard;
        let last = std::cmp::min(first + layers_per_shard - 1, num_layers - 1);
        let has_lm_head = (i + 1) == num_shards;
        println!("  Shard {}: layers {}-{} {}",
            i + 1,
            first,
            last,
            if has_lm_head { "(includes LM head)" } else { "" }
        );
    }

    std::fs::create_dir_all(&args.output_dir)?;

    let groups = group_tensors_by_layer(&tensors, num_layers);
    let data_sizes = compute_data_sizes(&tensors, tensor_data_start, file_size);
    let raw_metadata = &data[24..tensor_info_start];

    if args.split {
        for i in 0..num_shards {
            let filename = format!("shard_{:03}.gguf", i + 1);
            let path = args.output_dir.join(&filename);

            if path.exists() && !args.force {
                anyhow::bail!(
                    "Output file {} already exists. Use --force to overwrite.",
                    path.display()
                );
            }

            let first_layer = i * layers_per_shard;
            let last_layer = std::cmp::min(first_layer + layers_per_shard - 1, num_layers - 1);
            let has_lm_head = (i + 1) == num_shards;

            let indices = get_shard_tensor_indices(&groups, first_layer, last_layer, has_lm_head);

            let mut out = std::fs::File::create(&path)?;
            write_shard(
                &mut out,
                version,
                metadata_kv_count,
                &indices,
                &tensors,
                &data_sizes,
                raw_metadata,
                data,
                tensor_data_start,
            )?;

            let file_size_mb = out.metadata()?.len() as f64 / (1024.0 * 1024.0);
            println!(
                "  Shard {}: {} ({} tensors, {:.1} MB)",
                i + 1,
                path.display(),
                indices.len(),
                file_size_mb
            );
        }

        println!("\nAll {} shards written to {}", num_shards, args.output_dir.display());
    } else {
        let plan = serde_json::json!({
            "model_name": model_name,
            "architecture": architecture,
            "total_layers": num_layers,
            "layers_per_shard": layers_per_shard,
            "num_shards": num_shards,
            "source_file": args.model.to_string_lossy(),
            "shards": (0..num_shards).map(|i| {
                let first = i * layers_per_shard;
                let last = std::cmp::min(first + layers_per_shard - 1, num_layers - 1);
                serde_json::json!({
                    "shard_id": i + 1,
                    "first_layer": first,
                    "last_layer": last,
                    "has_lm_head": (i + 1) == num_shards,
                    "filename": format!("shard_{:03}.gguf", i + 1),
                    "estimated_size_mb": (file_size as f64 / num_shards as f64) / (1024.0 * 1024.0),
                })
            }).collect::<Vec<_>>(),
        });

        let plan_path = args.output_dir.join("split_plan.json");
        let plan_json = serde_json::to_string_pretty(&plan)?;
        std::fs::write(&plan_path, &plan_json)?;
        println!("\nSplit plan written to: {}", plan_path.display());
        println!("Run with --split to execute the actual file splitting.");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GGUF binary helpers
// ---------------------------------------------------------------------------

fn read_u32(data: &[u8], offset: &mut usize) -> anyhow::Result<u32> {
    if *offset + 4 > data.len() {
        anyhow::bail!("Unexpected end of file reading u32 at offset {}", offset);
    }
    let val = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

fn read_u64(data: &[u8], offset: &mut usize) -> anyhow::Result<u64> {
    if *offset + 8 > data.len() {
        anyhow::bail!("Unexpected end of file reading u64 at offset {}", offset);
    }
    let val = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(val)
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

fn align_up_u64(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

fn skip_value(data: &[u8], offset: &mut usize, value_type: u32) -> anyhow::Result<()> {
    match value_type {
        0 | 1 | 7 => *offset += 1,
        2 | 3 => *offset += 2,
        4 | 5 | 6 => *offset += 4,
        10 | 11 | 12 => *offset += 8,
        8 => {
            let len = read_u64(data, offset)?;
            *offset += len as usize;
        }
        9 => {
            let element_type = read_u32(data, offset)?;
            let arr_len = read_u64(data, offset)?;
            for _ in 0..arr_len {
                skip_value(data, offset, element_type)?;
            }
        }
        13 => anyhow::bail!("GGUF_TYPE_COUNT encountered as metadata value type"),
        t => anyhow::bail!("Unknown GGUF value type: {}", t),
    }
    Ok(())
}

fn get_metadata_string(
    data: &[u8],
    metadata_start: usize,
    kv_count: u64,
    target_key: &str,
) -> Option<String> {
    let mut offset = metadata_start;
    for _ in 0..kv_count {
        let key_len = read_u64(data, &mut offset).ok()?;
        let key = std::str::from_utf8(&data[offset..offset + key_len as usize]).ok()?;
        offset += key_len as usize;
        let value_type = read_u32(data, &mut offset).ok()?;
        if key == target_key {
            if value_type == 8 {
                let str_len = read_u64(data, &mut offset).ok()?;
                return String::from_utf8(data[offset..offset + str_len as usize].to_vec()).ok();
            }
            return None;
        }
        skip_value(data, &mut offset, value_type).ok()?;
    }
    None
}

// ---------------------------------------------------------------------------
// Layer / tensor bookkeeping
// ---------------------------------------------------------------------------

fn get_layer_id(name: &str) -> Option<u32> {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if let Ok(n) = rest[..dot].parse::<u32>() {
                return Some(n);
            }
        }
    }
    None
}

fn count_layers(tensors: &[TensorInfo]) -> u32 {
    let mut max = 0u32;
    for t in tensors {
        if let Some(layer) = get_layer_id(&t.name) {
            if layer >= max {
                max = layer + 1;
            }
        }
    }
    max
}

/// Groups tensor indices by layer, plus a final group for non-layer tensors.
fn group_tensors_by_layer(tensors: &[TensorInfo], num_layers: u32) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_layers as usize];
    let mut non_layer: Vec<usize> = Vec::new();

    for (i, t) in tensors.iter().enumerate() {
        if let Some(layer) = get_layer_id(&t.name) {
            if (layer as usize) < groups.len() {
                groups[layer as usize].push(i);
            } else {
                non_layer.push(i);
            }
        } else {
            non_layer.push(i);
        }
    }

    groups.push(non_layer);
    groups
}

/// Returns indices of tensors belonging to the given layer range.
fn get_shard_tensor_indices(
    groups: &[Vec<usize>],
    first_layer: u32,
    last_layer: u32,
    has_lm_head: bool,
) -> Vec<usize> {
    let mut indices = Vec::new();
    for layer in first_layer..=last_layer {
        if (layer as usize) < groups.len() - 1 {
            indices.extend_from_slice(&groups[layer as usize]);
        }
    }
    if has_lm_head {
        if let Some(non_layer) = groups.last() {
            indices.extend_from_slice(non_layer);
        }
    }
    indices
}

// ---------------------------------------------------------------------------
// Tensor data size computation
// ---------------------------------------------------------------------------

/// Computes the byte size of each tensor's data region by sorting tensors
/// by their offsets and calculating the gap to the next tensor (or file end).
fn compute_data_sizes(
    tensors: &[TensorInfo],
    tensor_data_start: usize,
    file_size: usize,
) -> Vec<u64> {
    let n = tensors.len();
    let mut sizes = vec![0u64; n];

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| tensors[i].offset);

    for j in 0..sorted.len() {
        let idx = sorted[j];
        let size = if j + 1 < sorted.len() {
            (tensors[sorted[j + 1]].offset - tensors[idx].offset) as u64
        } else {
            (file_size as u64) - (tensor_data_start as u64 + tensors[idx].offset)
        };
        sizes[idx] = size;
    }

    sizes
}

// ---------------------------------------------------------------------------
// Shard file writer
// ---------------------------------------------------------------------------

fn write_shard(
    out: &mut std::fs::File,
    version: u32,
    metadata_kv_count: u64,
    tensor_indices: &[usize],
    all_tensors: &[TensorInfo],
    data_sizes: &[u64],
    raw_metadata: &[u8],
    mmap: &[u8],
    original_tensor_data_start: usize,
) -> anyhow::Result<()> {
    // Sort shard tensor indices by their original offset so data is written
    // in file-order (preserves 32-byte alignment of each block).
    let mut sorted: Vec<usize> = tensor_indices.to_vec();
    sorted.sort_by_key(|&i| all_tensors[i].offset);

    // Compute new per-tensor data offsets (relative to shard's tensor data section).
    let mut new_offsets: HashMap<usize, u64> = HashMap::new();
    {
        let mut cur = 0u64;
        for &idx in &sorted {
            new_offsets.insert(idx, cur);
            cur = align_up_u64(cur + data_sizes[idx], 32);
        }
    }

    // ---- 1. Header (24 bytes) ------------------------------------------------
    out.write_all(&0x46554747u32.to_le_bytes())?;
    out.write_all(&version.to_le_bytes())?;
    out.write_all(&(tensor_indices.len() as u64).to_le_bytes())?;
    out.write_all(&metadata_kv_count.to_le_bytes())?;

    // ---- 2. Metadata KV pairs (raw copy) ------------------------------------
    out.write_all(raw_metadata)?;

    // ---- 3. Tensor infos (immediately follow metadata, no alignment) --------
    for &idx in tensor_indices {
        let t = &all_tensors[idx];
        let new_off = new_offsets[&idx];

        out.write_all(&(t.name.len() as u64).to_le_bytes())?;
        out.write_all(t.name.as_bytes())?;
        out.write_all(&t.n_dims.to_le_bytes())?;
        for d in &t.dims {
            out.write_all(&d.to_le_bytes())?;
        }
        out.write_all(&t.ggml_type.to_le_bytes())?;
        out.write_all(&new_off.to_le_bytes())?;
    }

    // ---- 5. Pad to 32-byte boundary (tensor data section start) -------------
    let pos = out.stream_position()?;
    let new_tensor_data_start = align_up(pos as usize, 32);
    let pad = new_tensor_data_start - (pos as usize);
    if pad > 0 {
        out.write_all(&vec![0u8; pad])?;
    }

    // ---- 6. Tensor data (raw copy) ------------------------------------------
    for &idx in &sorted {
        let t = &all_tensors[idx];
        let start = original_tensor_data_start + (t.offset as usize);
        let size = data_sizes[idx] as usize;
        if size > 0 {
            out.write_all(&mmap[start..start + size])?;
        }
    }

    Ok(())
}

fn calculate_from_node_config(config_path: &PathBuf) -> anyhow::Result<u32> {
    let content = std::fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;
    if let Some(lps) = config.get("layers_per_shard").and_then(|v| v.as_u64()) {
        return Ok(lps as u32);
    }
    Ok(10)
}
