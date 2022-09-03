#![no_std]

// Note: Must ALSO be changed in vi-shaders main_cs
pub const COMPUTE_THREADS:u32 = 512;

pub const WORKER_LOOPS:u32 = 2_u32.pow(13);

/// The number of hashes to compute in each compute pass
/// 
/// Must be divisible by COMPUTE_THREADS * WORKER_LOOPS
/// 
/// `PASS_SIZE / (COMPUTE_THREADS * WORKER_LOOPS)` is the number passed to dispatch_workgroups
pub const PASS_SIZE:u64 = 2_u64.pow(26);

// ResultInt::MAX must be > WORKER_LOOPS
pub type ResultInt = u16;