use wgpu::util::DeviceExt;

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
    #[test]
    fn it_wraps() {
        let n = ExecutionDetails{
            hashstate: [0u32; 5],
            final_block: [0u32; 16],
            nonce_index: 0,
            nonce_start: u64::MAX - 1,
            nonce_end: 2,
        };
        assert_eq!(n, 3);
    }
}

async fn run() {
    let state = SHA1_INITIAL_STATE;
    let mut final_block = [0u32; 16];
    // Put the nonce in the first 8 bytes = first 2 u32s
    let message_length:u64 = 8*8;
    final_block[2] = 0x8000_0000;
    final_block[14] = (message_length >> 32) as u32;
    final_block[15] = message_length as u32;
    // (&mut final_block[(63-8)..63]).copy_from_slice(message_length.to_be9_bytes().as_slice());

    let exec = ExecutionDetails {
        hashstate: state,
        final_block,
        nonce_index: 0,
        nonce_start: 0,
        nonce_end: 2_u64.pow(28),
    };

    // dbg!(sha1_smol::DEFAULT_STATE);
    // sha1_smol::Sha1::from(&[0; 8]).digest_debug(true);
    // dbg!(exec);
    let successful_nonces = execute_gpu(exec).await.unwrap();

    // dbg!(successful_nonces);
    //println!("{:x?}", successful_nonces);
    for n in successful_nonces {
        println!("{:8x?}: {}", n, sha1_smol::Sha1::from(n.to_be_bytes()).hexdigest());
    }
}

async fn execute_gpu(exec: ExecutionDetails) -> Option<Vec<u64>> {
    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }).await?;

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
            limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ).await.unwrap();
    dbg!(adapter.get_info());

    execute_gpu_inner(&device, &queue, exec).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    exec: ExecutionDetails,
) -> Option<Vec<u64>> {
    let sha1_shader:wgpu::ShaderModuleDescriptorSpirV = wgpu::include_spirv_raw!(env!("vi_shaders.spv"));

    let cs_module = unsafe { device.create_shader_module_spirv(&sha1_shader) };

    assert!(exec.num_executions() < u32::MAX.into());
    let size = exec.num_threads() as wgpu::BufferAddress;
    let size = ((size >> 2) << 2) + 4;

    let res_ty_size:u64 = std::mem::size_of::<vi_common::ResultInt>().try_into().unwrap();

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results go here?"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    //output
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size: size * res_ty_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut input_words:Vec<u32> = Vec::new();
    input_words.extend_from_slice(exec.final_block.as_slice());
    input_words.extend_from_slice(exec.hashstate.as_slice());
    input_words.push(0);
    input_words.push((exec.nonce_start >> 32) as u32);
    input_words.push((exec.nonce_start) as u32);

    // dbg!(&input_words);

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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("foobaridontknowwhattoputhere");
        let num_workgroups = size / (vi_common::COMPUTE_THREADS as u64);
        cpass.dispatch_workgroups(exec.num_groups().try_into().unwrap(), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    let start = std::time::Instant::now();
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data_bufferview = buffer_slice.get_mapped_range();
        let data:Vec<vi_common::ResultInt> = bytemuck::cast_slice(&data_bufferview).to_vec();
        // let data:&[u8] = &*data_bufferview;
        let el = start.elapsed();
        dbg!(el, exec.num_executions());
        eprintln!("{} MH/s", (exec.num_executions() as f32) / el.as_secs_f32() / 1_000_000.0);
        // dbg!(data);
        
        let result_scan_start = std::time::Instant::now();
        let mut res = Vec::new();
        for (i, v) in data.iter().enumerate() {
            if *v > 0 {
                let nonce_inc:u64 = ((v-1) as u64) + (i as u64) * vi_common::WORKER_LOOPS as u64;
                res.push(
                    exec.nonce_start.wrapping_add(nonce_inc)
                );
            }
        }
        dbg!(result_scan_start.elapsed());
        // dbg!(&res);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        // staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(res)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}
