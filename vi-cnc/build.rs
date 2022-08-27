use std::error::Error;
use spirv_builder::{SpirvBuilder, MetadataPrintout, Capability};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=../vi-shaders");
    SpirvBuilder::new("../vi-shaders", "spirv-unknown-vulkan1.1")
            .print_metadata(MetadataPrintout::Full)
            .capability(Capability::Int8)
            .capability(Capability::Int16)
            .capability(Capability::Int64)
            .build()?;
    Ok(())
} 