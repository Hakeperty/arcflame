//! GGUF Splitter — Splits a GGUF model file into per-layer shards
//! for distributed inference across ArcFlare nodes.
//!
//! Usage:
//!   gguf-splitter --model model.gguf --layers-per-shard 10 --output-dir ./shards
//!   gguf-splitter --model model.gguf --node-config nodes.json

use clap::Parser;
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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.model.exists() {
        anyhow::bail!("Model file not found: {}", args.model.display());
    }

    if args.list_tensors {
        list_tensors(&args.model)?;
        return Ok(());
    }

    // Phase 1: Read GGUF header and metadata
    let metadata = read_gguf_metadata(&args.model)?;

    println!("Model: {}", metadata.name);
    println!("Architecture: {}", metadata.architecture);
    println!("Total layers: {}", metadata.num_layers);
    println!("Hidden size: {}", metadata.hidden_size);
    println!("File size: {} GB", metadata.file_size_gb);

    // Calculate shard layout
    let layers_per_shard = if let Some(n) = args.layers_per_shard {
        n
    } else if let Some(config) = &args.node_config {
        calculate_from_node_config(&metadata, config)?
    } else {
        // Default: divide layers across 4 shards
        (metadata.num_layers as f64 / 4.0).ceil() as u32
    };

    let num_shards = (metadata.num_layers as f64 / layers_per_shard as f64).ceil() as u32;
    println!("\nSplit plan:");
    println!("  Layers per shard: {}", layers_per_shard);
    println!("  Total shards: {}", num_shards);

    for i in 0..num_shards {
        let first = i * layers_per_shard;
        let last = std::cmp::min(first + layers_per_shard - 1, metadata.num_layers - 1);
        let has_lm_head = (i + 1) == num_shards;
        println!("  Shard {}: layers {}-{} {}",
            i + 1,
            first,
            last,
            if has_lm_head { "(includes LM head)" } else { "" }
        );
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Write split plan
    let plan_path = args.output_dir.join("split_plan.json");
    let plan = SplitPlan {
        model_name: metadata.name,
        architecture: metadata.architecture,
        total_layers: metadata.num_layers,
        layers_per_shard,
        num_shards,
        source_file: args.model.to_string_lossy().to_string(),
        shards: (0..num_shards).map(|i| {
            let first = i * layers_per_shard;
            let last = std::cmp::min(first + layers_per_shard - 1, metadata.num_layers - 1);
            ShardInfo {
                shard_id: (i + 1) as u32,
                first_layer: first,
                last_layer: last,
                has_lm_head: (i + 1) == num_shards,
                filename: format!("shard_{:03}.gguf", i + 1),
                estimated_size_mb: metadata.file_size_gb * 1000.0 / num_shards as f64,
            }
        }).collect(),
    };

    let plan_json = serde_json::to_string_pretty(&plan)?;
    std::fs::write(&plan_path, &plan_json)?;
    println!("\nSplit plan written to: {}", plan_path.display());
    println!("Run with --split to execute the actual file splitting.");

    Ok(())
}

fn list_tensors(path: &PathBuf) -> anyhow::Result<()> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let content = &mmap[..];

    if content.len() < 24 {
        anyhow::bail!("File too small for valid GGUF");
    }

    let magic = u32::from_le_bytes([content[0], content[1], content[2], content[3]]);
    if magic != 0x4747_5546 && magic != 0x4647_5547 {
        let magic_be = u32::from_be_bytes([content[0], content[1], content[2], content[3]]);
        if magic_be != 0x4747_5546 && magic_be != 0x4647_5547 {
            anyhow::bail!("Not a valid GGUF file (magic: {:#x})", magic);
        }
    }

    let version = u32::from_le_bytes([content[4], content[5], content[6], content[7]]);
    println!("GGUF version: {}", version);

    let body = String::from_utf8_lossy(&content);
    let mut layers = std::collections::BTreeSet::new();

    for line in body.split('\0') {
        if line.starts_with("blk.") {
            if let Some(num_str) = line.strip_prefix("blk.") {
                if let Some(dot) = num_str.find('.') {
                    if let Ok(layer_num) = num_str[..dot].parse::<u32>() {
                        layers.insert(layer_num);
                    }
                }
            }
        }
    }

    println!("Found tensors from {} layers:", layers.len());
    for layer in &layers {
        print!(" {}", layer);
    }
    println!();

    for line in body.split('\0') {
        if line.starts_with("output") || line.starts_with("token_embd") {
            println!("Non-layer tensor: {}", line);
        }
    }

    Ok(())
}

fn read_gguf_metadata(path: &PathBuf) -> anyhow::Result<ModelMetadata> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let content = &mmap[..];
    let body = String::from_utf8_lossy(&content);

    let mut num_layers: u32 = 0;
    let hidden_size: u32 = 0;
    let mut architecture = String::new();
    let name = path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());

    for line in body.split('\0') {
        if line.contains("general.architecture") {
            if let Some(arch_start) = line.find('\0') {
                architecture = line[arch_start+1..].trim_matches('\0').to_string();
            }
        }
    }

    for line in body.split('\0') {
        if line.starts_with("blk.") {
            if let Some(num_str) = line.strip_prefix("blk.") {
                if let Some(dot) = num_str.find('.') {
                    if let Ok(n) = num_str[..dot].parse::<u32>() {
                        if n >= num_layers {
                            num_layers = n + 1;
                        }
                    }
                }
            }
        }
    }

    let file_size = file.metadata()?.len();
    let file_size_gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);

    Ok(ModelMetadata {
        name,
        architecture: if architecture.is_empty() { "unknown".to_string() } else { architecture },
        num_layers,
        hidden_size,
        file_size_gb,
    })
}

fn calculate_from_node_config(
    _metadata: &ModelMetadata,
    _config_path: &PathBuf,
) -> anyhow::Result<u32> {
    Ok(10)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ModelMetadata {
    name: String,
    architecture: String,
    num_layers: u32,
    hidden_size: u32,
    file_size_gb: f64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SplitPlan {
    model_name: String,
    architecture: String,
    total_layers: u32,
    layers_per_shard: u32,
    num_shards: u32,
    source_file: String,
    shards: Vec<ShardInfo>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ShardInfo {
    shard_id: u32,
    first_layer: u32,
    last_layer: u32,
    has_lm_head: bool,
    filename: String,
    estimated_size_mb: f64,
}
