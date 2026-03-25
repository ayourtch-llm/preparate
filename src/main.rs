mod gguf;

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

use gguf::{GgufFile, format_size};

#[derive(Parser)]
#[command(name = "preparate", about = "GGUF layer inspection and transplantation tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Display metadata and tensor/layer info from a GGUF file
    Info {
        /// Path to the GGUF file
        file: PathBuf,
    },
    /// Build a new GGUF by selecting layers from a source file.
    ///
    /// The layer list specifies which source layers to include in order.
    /// Layers can be repeated to duplicate them. They are renumbered
    /// sequentially (0, 1, 2, ...) in the output.
    ///
    /// Example: preparate merge model.gguf out.gguf 0,1,2,3,3,4,5
    Merge {
        /// Source GGUF file
        source: PathBuf,
        /// Output GGUF file
        output: PathBuf,
        /// Comma-separated list of source layer numbers (e.g. "0,1,2,3,3,4,5").
        /// Ranges are also supported (e.g. "0-3,3,4-7" expands to 0,1,2,3,3,4,5,6,7).
        layers: String,
    },
}

fn parse_layer_list(s: &str) -> Result<Vec<u32>, String> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start: u32 = start_s
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer number: '{start_s}'"))?;
            let end: u32 = end_s
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer number: '{end_s}'"))?;
            if end < start {
                return Err(format!("invalid range: {start}-{end} (end < start)"));
            }
            for i in start..=end {
                result.push(i);
            }
        } else {
            let n: u32 = part
                .parse()
                .map_err(|_| format!("invalid layer number: '{part}'"))?;
            result.push(n);
        }
    }
    if result.is_empty() {
        return Err("layer list is empty".to_string());
    }
    Ok(result)
}

fn cmd_info(path: PathBuf) {
    let gguf = match GgufFile::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error opening {}: {e}", path.display());
            process::exit(1);
        }
    };

    println!("File: {}", path.display());
    println!("GGUF version: {}", gguf.version);
    println!("Alignment: {} bytes", gguf.alignment);
    println!("Tensor count: {}", gguf.tensors.len());
    println!("Metadata entries: {}", gguf.metadata.len());
    println!();

    // Print metadata
    println!("=== Metadata ===");
    for (key, val) in &gguf.metadata {
        println!("  {key}: {}", val.display_short());
    }
    println!();

    // Print tensors grouped by layer
    let block_layers = gguf.block_layers();
    let non_block = gguf.non_block_tensor_indices();
    let total_blocks = gguf.block_count();

    println!("=== Non-block tensors ({}) ===", non_block.len());
    println!(
        "  {:<4} {:<50} {:<8} {:<25} {:>12}",
        "#", "Name", "Type", "Dimensions", "Size"
    );
    println!("  {}", "-".repeat(103));
    for (i, &idx) in non_block.iter().enumerate() {
        let t = &gguf.tensors[idx];
        let dims: Vec<String> = t.dimensions.iter().map(|d| d.to_string()).collect();
        println!(
            "  {:<4} {:<50} {:<8} {:<25} {:>12}",
            i,
            t.name,
            t.typ.name(),
            dims.join(" x "),
            format_size(t.data_size()),
        );
    }
    println!();

    println!("=== Block layers ({total_blocks} blocks) ===");
    for (&blk_num, tensor_indices) in &block_layers {
        let total_size: u64 = tensor_indices.iter().map(|&i| gguf.tensors[i].data_size()).sum();
        println!("  Block {blk_num} ({} tensors, {}):", tensor_indices.len(), format_size(total_size));
        for &idx in tensor_indices {
            let t = &gguf.tensors[idx];
            let dims: Vec<String> = t.dimensions.iter().map(|d| d.to_string()).collect();
            let suffix = t.block_suffix().unwrap_or(&t.name);
            println!(
                "    {:<45} {:<8} {:<25} {:>12}",
                suffix,
                t.typ.name(),
                dims.join(" x "),
                format_size(t.data_size()),
            );
        }
    }

    // Summary
    let total_data: u64 = gguf.tensors.iter().map(|t| t.data_size()).sum();
    println!();
    println!("Total tensor data: {}", format_size(total_data));
}

fn cmd_merge(source: PathBuf, output: PathBuf, layers_str: String) {
    let layer_list = match parse_layer_list(&layers_str) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error parsing layer list: {e}");
            process::exit(1);
        }
    };

    let gguf = match GgufFile::open(&source) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error opening {}: {e}", source.display());
            process::exit(1);
        }
    };

    let src_blocks = gguf.block_count();
    println!("Source: {} ({src_blocks} blocks)", source.display());
    println!(
        "Layer mapping ({} output layers): {:?}",
        layer_list.len(),
        layer_list
    );

    match gguf::merge_layers(&gguf, &output, &layer_list) {
        Ok(()) => {
            println!("Written: {}", output.display());
        }
        Err(e) => {
            eprintln!("Error writing output: {e}");
            process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { file } => cmd_info(file),
        Commands::Merge {
            source,
            output,
            layers,
        } => cmd_merge(source, output, layers),
    }
}
