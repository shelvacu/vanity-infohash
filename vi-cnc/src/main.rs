use std::sync::Arc;
use wgpu::util::DeviceExt;
use pollster::FutureExt as _;
use rand::Rng;

const SHA1_INITIAL_STATE:[u32; 5] = [
    0x67452301,
    0xEFCDAB89,
    0x98BADCFE,
    0x10325476,
    0xC3D2E1F0,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct ExecutionDetails {
    hashstate:   [u32; 5 ], // aka [u8 ; 20] aka 160 bits
    final_block: [u32; 16], // aka [u8 ; 64] aka [u32; 16] aka 512 bits
    nonce_index: u8, //bytes? u32s?
    nonce_start: u64, //inclusive
    nonce_end:   u64, //exclusive
}

impl ExecutionDetails {
    fn num_executions(self) -> u64 {
        self.nonce_end.wrapping_sub(self.nonce_start)
    }

    fn num_threads(self) -> u64 {
        self.num_executions() / (vi_common::WORKER_LOOPS as u64)
    }

    fn num_groups(self) -> u64 {
        self.num_threads() / (vi_common::COMPUTE_THREADS as u64)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_wraps() {
        assert_eq!(
            ExecutionDetails{
                hashstate: [0u32; 5],
                final_block: [0u32; 16],
                nonce_index: 0,
                nonce_start: 0,
                nonce_end: 0,
            }.num_executions(),
            0
        );
        assert_eq!(
            ExecutionDetails{
                hashstate: [0u32; 5],
                final_block: [0u32; 16],
                nonce_index: 0,
                nonce_start: 0,
                nonce_end: 1,
            }.num_executions(),
            1
        );
        assert_eq!(
            ExecutionDetails{
                hashstate: [0u32; 5],
                final_block: [0u32; 16],
                nonce_index: 0,
                nonce_start: u64::MAX - 1,
                nonce_end: 2,
            }.num_executions(),
            4
        );
    }
}

fn run() {
    let state = SHA1_INITIAL_STATE;
    let mut final_block = [0u32; 16];
    // Put the nonce in the first 8 bytes = first 2 u32s
    let message_length:u64 = 8*8;
    final_block[2] = 0x8000_0000;
    final_block[14] = (message_length >> 32) as u32;
    final_block[15] = message_length as u32;

    let nonce_start:u64 = rand::thread_rng().gen();
    dbg!(nonce_start);

    let exec = ExecutionDetails {
        hashstate: state,
        final_block,
        nonce_index: 0,
        nonce_start,
        nonce_end: 0, //to be filled in later
    };

    let start = std::time::Instant::now();
    let mut last_printout = std::time::Instant::now();
    let mut count:u64 = 0;
    execute_gpu(exec, nonce_start, move |successful_nonces| {
        for n in successful_nonces {
            println!("{:16x?}: {}", n, sha1_smol::Sha1::from(n.to_be_bytes()).hexdigest());
        }
        count += vi_common::PASS_SIZE;
        if last_printout.elapsed() > std::time::Duration::from_secs(5) {
            let hps = (count as f32) / start.elapsed().as_secs_f32();
            println!("{:.2} MH/s", hps / 1_000_000f32);
            last_printout = std::time::Instant::now();
        }
    });
}

fn execute_gpu(
    exec: ExecutionDetails,
    nonce_start: u64,
    callback: impl FnMut(Vec<u64>) + std::marker::Send + 'static,
) {
    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }).block_on().unwrap();

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
            limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ).block_on().unwrap();
    dbg!(adapter.get_info());

    execute_gpu_inner(device, queue, exec, nonce_start, callback);
}

fn execute_gpu_inner(
    device: wgpu::Device,
    queue: wgpu::Queue,
    exec: ExecutionDetails,
    nonce_start: u64,
    mut callback: impl FnMut(Vec<u64>) + std::marker::Send + 'static,
) {
    let sha1_shader:wgpu::ShaderModuleDescriptorSpirV = wgpu::include_spirv_raw!(env!("vi_shaders.spv"));

    let cs_module = unsafe { device.create_shader_module_spirv(&sha1_shader) };
    let cs_module_arc = Arc::new(cs_module);

    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);

    let (exec_send, exec_recv) = crossbeam::channel::bounded(8);
    let (resu_send, resu_recv) = crossbeam::channel::bounded(8);
    let mut threads = Vec::new();
    threads.push(std::thread::spawn(move || {
        let mut next_start_nonce = nonce_start;
        loop {
            let mut new_exec = exec;
            new_exec.nonce_start = next_start_nonce;
            next_start_nonce = next_start_nonce.wrapping_add(vi_common::PASS_SIZE);
            new_exec.nonce_end = next_start_nonce;
            exec_send.send(new_exec).unwrap();
        }
    }));
    for _ in 0..4 {
        let my_recv = exec_recv.clone();
        let my_send = resu_send.clone();
        let my_device = Arc::clone(&device_arc);
        let my_queue = Arc::clone(&queue_arc);
        let my_module = Arc::clone(&cs_module_arc);
        threads.push(std::thread::spawn(move || {
            loop {
                let exec = my_recv.recv().unwrap();
                let res = execute_even_more_inner(
                    &*my_device,
                    &*my_queue,
                    &*my_module,
                    exec,
                );
                my_send.send(res).unwrap();
            }
        }));
    }
    threads.push(std::thread::spawn(move || {
        loop {
            callback(resu_recv.recv().unwrap());
        }
    }));

    for t in threads {
        t.join().unwrap();
    }
}

fn execute_even_more_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cs_module: &wgpu::ShaderModule,
    exec: ExecutionDetails,
) -> Vec<u64> {
    assert!(exec.num_threads() < u32::MAX.into());
    let thing = exec.num_threads() as wgpu::BufferAddress;
    let res_ty_size:u64 = std::mem::size_of::<vi_common::ResultInt>().try_into().unwrap();
    let result_bytesize = thing * res_ty_size;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results go here?"),
        size: result_bytesize,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    //output
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size: result_bytesize,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut input_words:Vec<u32> = Vec::new();
    input_words.extend_from_slice(exec.final_block.as_slice());
    input_words.extend_from_slice(exec.hashstate.as_slice());
    input_words.push(0);
    input_words.push((exec.nonce_start >> 32) as u32);
    input_words.push((exec.nonce_start) as u32);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("da blokk"),
        contents: unsafe{ std::slice::from_raw_parts(input_words.as_ptr() as *const u8, input_words.len() * 4) },
        usage: wgpu::BufferUsages::STORAGE,
    });


    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            count: None,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage { read_only: false },
            },
        }, wgpu::BindGroupLayoutEntry {
            binding: 1,
            count: None,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage { read_only: false },
            },
        },],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        },wgpu::BindGroupEntry {
            binding: 1,
            resource: storage_buffer.as_entire_binding(),
        },],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main_cs",
    });

    // Remember, each method on encoder is "recording" a set of steps to be done, and not doing them until later
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("foobaridontknowwhattoputhere");
        cpass.dispatch_workgroups(exec.num_groups().try_into().unwrap(), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, result_bytesize);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping. The example code had this sending over a channel, but I don't think we need to do that.
    buffer_slice.map_async(wgpu::MapMode::Read, |_| ());

    // Poll the device in a blocking manner so that our future resolves.
    // Supposedly "In an actual application, `device.poll(...)` should be called in an event loop or on another thread."
    // But why?
    device.poll(wgpu::Maintain::Wait);

    // Gets contents of buffer
    let data_bufferview = buffer_slice.get_mapped_range();
    let data:Vec<vi_common::ResultInt> = bytemuck::cast_slice(&data_bufferview).to_vec();

    let mut res = Vec::new();
    for (i, v) in data.iter().enumerate() {
        if *v > 0 {
            let nonce_inc:u64 = ((v-1) as u64) + (i as u64) * vi_common::WORKER_LOOPS as u64;
            res.push(
                exec.nonce_start.wrapping_add(nonce_inc)
            );
        }
    }

    // With the current interface, we have to make sure all mapped views are
    // dropped before we unmap the buffer.
    drop(data);
    drop(data_bufferview);
    staging_buffer.unmap(); // Unmaps buffer from memory. This does something different from drop, but I don't understand why.

    res
}

fn main() {
    env_logger::init();
    run();
}
