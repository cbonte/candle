//! CUDA fast path for GGUF matmul with BF16/F32 activations.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::cuda::{QCudaStorage, MATRIX_ROW_PADDING};
use super::GgmlDType;
use crate::cuda_backend::DeviceId;
use crate::{backend::BackendStorage, CudaDevice, CudaStorage, DType, Result, Shape};

use cudarc::driver::{CudaSlice, DevicePtr};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Quant types supported by the fast MMVQ kernels.
fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

const MMVQ_MAX_BATCH: usize = 8;

// ---------------------------------------------------------------------------
// Per-device Q8_1 scratch workspace (grows-only, reused across calls).
// ---------------------------------------------------------------------------

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

static WORKSPACE: OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>> = OnceLock::new();

/// Pre-grow the per-device Q8_1 scratch workspace to at least
/// `min_bytes`. Use BEFORE CUDA graph capture so the workspace's
/// device pointer stays stable across captures (no reallocation
/// triggered by larger matmul during capture). Returns the current
/// workspace pointer + size after the call.
///
/// When the workspace grows mid-capture,
/// the OLD pointer captured by earlier kernels in the graph is
/// freed → ILLEGAL_ADDRESS on replay. Pre-growing to the model's
/// max scratch_bytes eliminates this.
pub fn ensure_workspace_capacity(dev: &CudaDevice, min_bytes: usize) -> Result<()> {
    let _ = workspace_ensure(dev, min_bytes)?;
    Ok(())
}

/// Compute the maximum scratch byte size needed for a single MMVQ
/// call with input dim `max_k`. For decode (`b_size=1`):
///   bytes = ((max_k + MATRIX_ROW_PADDING) / 32) * 36
pub fn max_scratch_bytes_for_k(max_k: usize) -> usize {
    let k_padded = pad(max_k, MATRIX_ROW_PADDING);
    let num_blocks = k_padded / Q8_1_BLOCK_SIZE;
    num_blocks * Q8_1_TYPE_SIZE
}

/// Drop the per-device Q8_1 scratch workspace and return the reserved
/// VRAM to the cudaMallocAsync mempool. Call this on model unload so
/// VRAM accounting matches reality. Allocations get re-created on next
/// matmul.
pub fn release_workspaces() {
    if let Some(map) = WORKSPACE.get() {
        let mut guard = map.lock().unwrap();
        guard.clear();
    }
}

/// Returns a device pointer to the scratch workspace, growing it if needed.
/// The returned `MutexGuard` must be held alive until the kernels using
/// this pointer have been launched (all launches are on the device's
/// default stream, so they are serialised).
fn workspace_ensure(
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(
    u64,
    std::sync::MutexGuard<'static, HashMap<DeviceId, WorkspaceSlot>>,
)> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let mut guard = map.lock().unwrap();
    let slot = match guard.entry(device_key) {
        std::collections::hash_map::Entry::Occupied(entry) => {
            let slot = entry.into_mut();
            if slot.cap < bytes {
                slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
                slot.cap = bytes;
            }
            slot
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            let slice = unsafe { dev.alloc::<u8>(bytes)? };
            entry.insert(WorkspaceSlot { slice, cap: bytes })
        }
    };
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, guard))
}

// ---------------------------------------------------------------------------
// Launcher dispatch by weight dtype and output dtype.
// ---------------------------------------------------------------------------

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Try the fast MMVQ path. Returns `Ok(None)` when the fast path is not applicable:
/// - unsupported quant dtype
/// - batch too large
/// - non-BF16/F32 input
pub fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;

    // Gate checks.
    let w_dtype = qstorage.dtype();
    if !supports(w_dtype) {
        return Ok(None);
    }
    let input_dtype = rhs.dtype();
    if !matches!(input_dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k {
        return Ok(None);
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = qstorage.device();
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(dev, scratch_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let weight_ptr = qstorage.device_ptr()? as *const std::ffi::c_void;

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let stream = dev.cuda_stream();

    match input_dtype {
        DType::BF16 => {
            let rhs_slice = rhs.as_cuda_slice::<half::bf16>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            let rhs_ptr = rhs_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
            let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_bf16(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        DType::F16 => {
            let rhs_slice = rhs.as_cuda_slice::<half::f16>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            let rhs_ptr = rhs_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
            let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_f16(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        DType::F32 => {
            let rhs_slice = rhs.as_cuda_slice::<f32>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            let rhs_ptr = rhs_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
            let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_f32(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Fused (up + gate + SwiGLU) wrapper.
//
// Computes:  out[i] = (up_proj @ x)[i] * silu( (gate_proj @ x)[i] )
//
// In one kernel launch instead of three (up_matmul, gate_matmul, silu_mul).
// Constraints: Q4_K weights, F32 activation (and F32 output), batch
// size 1 (decode token). Returns Ok(None) on any other shape — caller falls
// back to the unfused 3-launch chain.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn try_fused_silu(
    up_storage: &QCudaStorage,
    gate_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;

    // Gate: both weights Q4_K, ncols_dst=1.
    if up_storage.dtype() != GgmlDType::Q4K || gate_storage.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    // Accept F32 OR BF16 hidden state. BF16 dispatches to the
    // `launch_mmvq_gguf_quantize_q8_1_bf16` kernel variant (already
    // in tree per ffi.rs:778) — saves the F32-cast launch that
    // BF16-LLM callers (e.g. BF16 dense models)
    // would otherwise pay. Q4_K MMVQ output stays F32 either way;
    // downstream attn_output/ffn_down accept any input dtype.
    if rhs.dtype() != DType::F32 && rhs.dtype() != DType::BF16 {
        return Ok(None);
    }
    // up_proj and gate_proj must share shape (same hidden→intermediate map).
    if up_storage.device().id() != gate_storage.device().id() {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k {
        return Ok(None);
    }
    // Decode-only (single-token).
    if b_size != 1 {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = up_storage.device();
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(dev, scratch_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let up_ptr = up_storage.device_ptr()? as *const std::ffi::c_void;
    let gate_ptr = gate_storage.device_ptr()? as *const std::ffi::c_void;

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let stream = dev.cuda_stream();

    // Slice + ptr + output alloc per input dtype.
    // BF16 input now writes BF16 output via the new bf16 fused-silu
    // launcher (avoids the F32→BF16 cast in the residual add downstream).
    // F32 input writes F32 output (unchanged).
    let rhs_slice_f32;
    let rhs_slice_bf16;
    let out_storage = match rhs.dtype() {
        DType::F32 => {
            rhs_slice_f32 = rhs.as_cuda_slice::<f32>()?.slice(o1..o2);
            let rhs_ptr_void = rhs_slice_f32.device_ptr(&stream).0
                as *const std::ffi::c_void;
            let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;
            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                    rhs_ptr_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                ffi::launch_mmvq_gguf_q4_k_f32_fused_silu(
                    up_ptr,
                    gate_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    stream_ptr,
                );
            }
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }
        DType::BF16 => {
            rhs_slice_bf16 = rhs.as_cuda_slice::<half::bf16>()?.slice(o1..o2);
            let rhs_ptr_void = rhs_slice_bf16.device_ptr(&stream).0
                as *const std::ffi::c_void;
            let out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;
            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                    rhs_ptr_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                ffi::launch_mmvq_gguf_q4_k_bf16_fused_silu(
                    up_ptr,
                    gate_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    stream_ptr,
                );
            }
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }
        _ => return Ok(None),
    };

    Ok(Some((out_storage, out_shape.into())))
}

/// GELU variant of `try_fused_silu`. Same SEPARATE-weight gate + up
/// layout, but the activation on `gate` is GELU (tanh approximation,
/// matching HF's gemma4 reference) instead of SiLU. Used by gemma4
/// dense decode where the standard try_fused_silu path is skipped
/// (use_gelu = true).
///
/// Accepts F32 and BF16 input. A BF16 hidden state skips the upfront
/// cast — the BF16 quantize is slightly slower per call, but for a
/// shallow dense stack the per-call delta stays well below the saved
/// cast launch.
#[allow(clippy::too_many_arguments)]
pub fn try_fused_gelu(
    up_storage: &QCudaStorage,
    gate_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;

    if up_storage.dtype() != GgmlDType::Q4K || gate_storage.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    if rhs.dtype() != DType::F32 && rhs.dtype() != DType::BF16 {
        return Ok(None);
    }
    if up_storage.device().id() != gate_storage.device().id() {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;
    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k {
        return Ok(None);
    }
    if b_size != 1 {
        return Ok(None);
    }
    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = up_storage.device();
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (scratch_ptr, _workspace_guard) = workspace_ensure(dev, scratch_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let up_ptr = up_storage.device_ptr()? as *const std::ffi::c_void;
    let gate_ptr = gate_storage.device_ptr()? as *const std::ffi::c_void;

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let stream = dev.cuda_stream();
    let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

    let rhs_slice_f32;
    let rhs_slice_bf16;
    let rhs_ptr: *const std::ffi::c_void = match rhs.dtype() {
        DType::F32 => {
            rhs_slice_f32 = rhs.as_cuda_slice::<f32>()?.slice(o1..o2);
            rhs_slice_f32.device_ptr(&stream).0 as *const std::ffi::c_void
        }
        DType::BF16 => {
            rhs_slice_bf16 = rhs.as_cuda_slice::<half::bf16>()?.slice(o1..o2);
            rhs_slice_bf16.device_ptr(&stream).0 as *const std::ffi::c_void
        }
        _ => return Ok(None),
    };

    unsafe {
        match rhs.dtype() {
            DType::F32 => ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                rhs_ptr,
                scratch_ptr,
                k as i32,
                k_padded as i32,
                b_size as i32,
                stream_ptr,
            ),
            DType::BF16 => ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                rhs_ptr,
                scratch_ptr,
                k as i32,
                k_padded as i32,
                b_size as i32,
                stream_ptr,
            ),
            _ => unreachable!(),
        }
        ffi::launch_mmvq_gguf_q4_k_f32_fused_gelu(
            up_ptr,
            gate_ptr,
            scratch_ptr as *const std::ffi::c_void,
            out_ptr,
            k as i32,
            nrows as i32,
            stride_col_y,
            stride_col_dst,
            stream_ptr,
        );
    }

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Some((out_storage, out_shape.into())))
}

/// Same as `try_fused_silu` but for the load-time-fused `[gate || up]`
/// single-weight layout. Weight `concat_storage` has shape `[2*N, K]`
/// where the first N rows are gate, second N rows are up. Output shape
/// is `[..., N]` (single SwiGLU result, not 2N).
///
/// Used for qwen2/qwen3/qwen3moe-style models where the caller concats
/// gate+up into one weight at load time. The sibling `try_fused_silu`
/// handles the SEPARATE gate/up weight case.
#[allow(clippy::too_many_arguments)]
pub fn try_fused_silu_concat(
    concat_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;

    if concat_storage.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    if rhs.dtype() != DType::F32 {
        return Ok(None);
    }

    let (two_n_rows, ncols) = self_shape.dims2()?;
    if two_n_rows % 2 != 0 {
        return Ok(None);
    }
    let n_rows = two_n_rows / 2;

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k || b_size != 1 {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = concat_storage.device();
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(dev, scratch_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = n_rows as i32;

    // Pointer arithmetic: gate = base, up = base + n_rows * (K/256) * 144.
    // For Q4_K each block is 144 bytes covering 256 quants.
    const Q4K_BLOCK_BYTES: usize = 144;
    const Q4K_BLOCK_QUANTS: usize = 256;
    let blocks_per_row = ncols / Q4K_BLOCK_QUANTS;
    let row_bytes = blocks_per_row * Q4K_BLOCK_BYTES;
    let base_ptr = concat_storage.device_ptr()? as *const std::ffi::c_void;
    let up_offset_bytes = n_rows * row_bytes;
    // gate = first half (vgate in kernel), up = second half (vx in kernel).
    let vgate_ptr = base_ptr;
    let vx_ptr = unsafe { (base_ptr as *const u8).add(up_offset_bytes) } as *const std::ffi::c_void;

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(n_rows);

    let stream = dev.cuda_stream();
    let rhs_slice = rhs.as_cuda_slice::<f32>()?;
    let rhs_slice = rhs_slice.slice(o1..o2);
    let out = unsafe { dev.alloc::<f32>(n_rows * b_size)? };

    let rhs_ptr = rhs_slice.device_ptr(&stream).0 as *const std::ffi::c_void;
    let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

    unsafe {
        ffi::launch_mmvq_gguf_quantize_q8_1_f32(
            rhs_ptr,
            scratch_ptr,
            k as i32,
            k_padded as i32,
            b_size as i32,
            stream_ptr,
        );
        ffi::launch_mmvq_gguf_q4_k_f32_fused_silu(
            vx_ptr,
            vgate_ptr,
            scratch_ptr as *const std::ffi::c_void,
            out_ptr,
            k as i32,
            n_rows as i32,
            stride_col_y,
            stride_col_dst,
            stream_ptr,
        );
    }

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Some((out_storage, out_shape.into())))
}
