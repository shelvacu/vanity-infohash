#![no_std]

// Note: Must ALSO be changed in vi-shaders main_cs
pub const COMPUTE_THREADS:u32 = 128;

pub const WORKER_LOOPS:u32 = 2_u32.pow(12);

// ResultInt::MAX must be > WORKER_LOOPS
pub type ResultInt = u16;