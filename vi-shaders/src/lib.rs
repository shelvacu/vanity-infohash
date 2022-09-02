#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]
// HACK can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use spirv_std::glam::UVec3;

fn idx(val: &[u32], index: usize) -> u32 {
    val[index]
}

#[allow(unreachable_code)]
#[spirv(compute(threads(512)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] inputs: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] results: &mut [vi_common::ResultInt],
) {
    let index:u32 = id.x;
    let mut my_res = 0;
    for loop_i in 0..vi_common::WORKER_LOOPS {
        let mut w = [0u32; 16]; 
        for i in 0..16 {
            w[i] = idx(inputs, i);
        }
        let h0 = idx(inputs, 16);
        let h1 = idx(inputs, 17);
        let h2 = idx(inputs, 18);
        let h3 = idx(inputs, 19);
        let h4 = idx(inputs, 20);
        // idx(inputs, 21) is dummy val
        let nonce_start = ((idx(inputs, 22) as u64) << 32) | (idx(inputs, 23) as u64);
        let my_nonce = nonce_start
            .wrapping_add((index as u64) * vi_common::WORKER_LOOPS as u64)
            .wrapping_add(loop_i as u64);
        w[0] = (my_nonce >> 32) as u32;
        w[1] = my_nonce as u32; 

        let mut state:[u32;5] = [h0, h1, h2, h3, h4];
        let block = w;
        let a = state[0];
        let b = state[1];
        let c = state[2];
        let d = state[3];
        let e = state[4];
        let (block, b, e) = r0(block, a, b, c, d, e, 0);
        let (block, a, d) = r0(block, e, a, b, c, d, 1);
        let (block, e, c) = r0(block, d, e, a, b, c, 2);
        let (block, d, b) = r0(block, c, d, e, a, b, 3);
        let (block, c, a) = r0(block, b, c, d, e, a, 4);
        let (block, b, e) = r0(block, a, b, c, d, e, 5);
        let (block, a, d) = r0(block, e, a, b, c, d, 6);
        let (block, e, c) = r0(block, d, e, a, b, c, 7);
        let (block, d, b) = r0(block, c, d, e, a, b, 8);
        let (block, c, a) = r0(block, b, c, d, e, a, 9);
        let (block, b, e) = r0(block, a, b, c, d, e, 10);
        let (block, a, d) = r0(block, e, a, b, c, d, 11);
        let (block, e, c) = r0(block, d, e, a, b, c, 12);
        let (block, d, b) = r0(block, c, d, e, a, b, 13);
        let (block, c, a) = r0(block, b, c, d, e, a, 14);
        let (block, b, e) = r0(block, a, b, c, d, e, 15);
        let (block, a, d) = r1(block, e, a, b, c, d, 0);
        let (block, e, c) = r1(block, d, e, a, b, c, 1);
        let (block, d, b) = r1(block, c, d, e, a, b, 2);
        let (block, c, a) = r1(block, b, c, d, e, a, 3);
        let (block, b, e) = r2(block, a, b, c, d, e, 4);
        let (block, a, d) = r2(block, e, a, b, c, d, 5);
        let (block, e, c) = r2(block, d, e, a, b, c, 6);
        let (block, d, b) = r2(block, c, d, e, a, b, 7);
        let (block, c, a) = r2(block, b, c, d, e, a, 8);
        let (block, b, e) = r2(block, a, b, c, d, e, 9);
        let (block, a, d) = r2(block, e, a, b, c, d, 10);
        let (block, e, c) = r2(block, d, e, a, b, c, 11);
        let (block, d, b) = r2(block, c, d, e, a, b, 12);
        let (block, c, a) = r2(block, b, c, d, e, a, 13);
        let (block, b, e) = r2(block, a, b, c, d, e, 14);
        let (block, a, d) = r2(block, e, a, b, c, d, 15);
        let (block, e, c) = r2(block, d, e, a, b, c, 0);
        let (block, d, b) = r2(block, c, d, e, a, b, 1);
        let (block, c, a) = r2(block, b, c, d, e, a, 2);
        let (block, b, e) = r2(block, a, b, c, d, e, 3);
        let (block, a, d) = r2(block, e, a, b, c, d, 4);
        let (block, e, c) = r2(block, d, e, a, b, c, 5);
        let (block, d, b) = r2(block, c, d, e, a, b, 6);
        let (block, c, a) = r2(block, b, c, d, e, a, 7);
        let (block, b, e) = r3(block, a, b, c, d, e, 8);
        let (block, a, d) = r3(block, e, a, b, c, d, 9);
        let (block, e, c) = r3(block, d, e, a, b, c, 10);
        let (block, d, b) = r3(block, c, d, e, a, b, 11);
        let (block, c, a) = r3(block, b, c, d, e, a, 12);
        let (block, b, e) = r3(block, a, b, c, d, e, 13);
        let (block, a, d) = r3(block, e, a, b, c, d, 14);
        let (block, e, c) = r3(block, d, e, a, b, c, 15);
        let (block, d, b) = r3(block, c, d, e, a, b, 0);
        let (block, c, a) = r3(block, b, c, d, e, a, 1);
        let (block, b, e) = r3(block, a, b, c, d, e, 2);
        let (block, a, d) = r3(block, e, a, b, c, d, 3);
        let (block, e, c) = r3(block, d, e, a, b, c, 4);
        let (block, d, b) = r3(block, c, d, e, a, b, 5);
        let (block, c, a) = r3(block, b, c, d, e, a, 6);
        let (block, b, e) = r3(block, a, b, c, d, e, 7);
        let (block, a, d) = r3(block, e, a, b, c, d, 8);
        let (block, e, c) = r3(block, d, e, a, b, c, 9);
        let (block, d, b) = r3(block, c, d, e, a, b, 10);
        let (block, c, a) = r3(block, b, c, d, e, a, 11);
        let (block, b, e) = r4(block, a, b, c, d, e, 12);
        let (block, a, d) = r4(block, e, a, b, c, d, 13);
        let (block, e, c) = r4(block, d, e, a, b, c, 14);
        let (block, d, b) = r4(block, c, d, e, a, b, 15);
        let (block, c, a) = r4(block, b, c, d, e, a, 0);
        let (block, b, e) = r4(block, a, b, c, d, e, 1);
        let (block, a, d) = r4(block, e, a, b, c, d, 2);
        let (block, e, c) = r4(block, d, e, a, b, c, 3);
        let (block, d, b) = r4(block, c, d, e, a, b, 4);
        let (block, c, a) = r4(block, b, c, d, e, a, 5);
        let (block, b, e) = r4(block, a, b, c, d, e, 6);
        let (block, a, d) = r4(block, e, a, b, c, d, 7);
        let (block, e, c) = r4(block, d, e, a, b, c, 8);
        let (block, d, b) = r4(block, c, d, e, a, b, 9);
        let (block, c, a) = r4(block, b, c, d, e, a, 10);
        let (block, b, e) = r4(block, a, b, c, d, e, 11);
        let (block, a, d) = r4(block, e, a, b, c, d, 12);
        let (block, e, c) = r4(block, d, e, a, b, c, 13);
        let (block, d, b) = r4(block, c, d, e, a, b, 14);
        let (_, c, a) = r4(block, b, c, d, e, a, 15);

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        let result_h0 = state[0];
        if (result_h0 >> 6) == 0x0000_0000 {
            my_res = loop_i + 1;
        }
    }

    results[index as usize] = my_res as vi_common::ResultInt;
}




const fn rol(value: u32, bits: usize) -> u32 {
    (value << bits) | (value >> (32 - bits))
}

const fn blk(block: &[u32; 16], i: usize) -> u32 {
    let value = block[(i + 13) & 15] ^ block[(i + 8) & 15] ^ block[(i + 2) & 15] ^ block[i];
    rol(value, 1)
}

const fn r0(
    block: [u32; 16],
    v: u32,
    mut w: u32,
    x: u32,
    y: u32,
    mut z: u32,
    i: usize,
) -> ([u32; 16], u32, u32) {
    let n = ((w & (x ^ y)) ^ y)
        .wrapping_add(block[i])
        .wrapping_add(0x5a82_7999)
        .wrapping_add(rol(v, 5));
    z = z.wrapping_add(n);
    w = rol(w, 30);
    (block, w, z)
}

const fn r1(
    mut block: [u32; 16],
    v: u32,
    mut w: u32,
    x: u32,
    y: u32,
    mut z: u32,
    i: usize,
) -> ([u32; 16], u32, u32) {
    block[i] = blk(&block, i);
    let n = ((w & (x ^ y)) ^ y)
        .wrapping_add(block[i])
        .wrapping_add(0x5a82_7999)
        .wrapping_add(rol(v, 5));
    z = z.wrapping_add(n);
    w = rol(w, 30);
    (block, w, z)
}

const fn r2(
    mut block: [u32; 16],
    v: u32,
    mut w: u32,
    x: u32,
    y: u32,
    mut z: u32,
    i: usize,
) -> ([u32; 16], u32, u32) {
    block[i] = blk(&block, i);
    let n = (w ^ x ^ y)
        .wrapping_add(block[i])
        .wrapping_add(0x6ed_9eba1)
        .wrapping_add(rol(v, 5));
    z = z.wrapping_add(n);
    w = rol(w, 30);
    (block, w, z)
}

const fn r3(
    mut block: [u32; 16],
    v: u32,
    mut w: u32,
    x: u32,
    y: u32,
    mut z: u32,
    i: usize,
) -> ([u32; 16], u32, u32) {
    block[i] = blk(&block, i);
    let n = (((w | x) & y) | (w & x))
        .wrapping_add(block[i])
        .wrapping_add(0x8f1b_bcdc)
        .wrapping_add(rol(v, 5));
    z = z.wrapping_add(n);
    w = rol(w, 30);
    (block, w, z)
}

const fn r4(
    mut block: [u32; 16],
    v: u32,
    mut w: u32,
    x: u32,
    y: u32,
    mut z: u32,
    i: usize,
) -> ([u32; 16], u32, u32) {
    block[i] = blk(&block, i);
    let n = (w ^ x ^ y)
        .wrapping_add(block[i])
        .wrapping_add(0xca62_c1d6)
        .wrapping_add(rol(v, 5));
    z = z.wrapping_add(n);
    w = rol(w, 30);
    (block, w, z)
}