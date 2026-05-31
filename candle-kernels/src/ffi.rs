use core::ffi::c_void;
#[allow(dead_code)]
#[allow(improper_ctypes)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    /// Fused gate+up MoE GEMM with SiLU activation and elementwise
    /// multiply. Shares the input quantize and the per-block input load
    /// across both gate and up dot products, then writes a single
    /// `silu(gate) * up` value per output position. Replaces the
    /// two-call `moe_gemm_gguf(gate)` + `moe_gemm_gguf(up)` +
    /// elementwise-silu-and-mul sequence with a single launch.
    pub fn moe_gemm_gguf_gate_up_silu_mul(
        input: *const f32,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        output: *mut c_void, // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Same as moe_gemm_gguf_gate_up_silu_mul but with GELU(tanh-approx)
    /// activation and a CONCATENATED gate||up weight tensor of shape
    /// `[num_experts, 2*size_n, size_k]` (gate = first N rows, up =
    /// next N rows). Used by the gemma4-MoE FFN where the GGUF stores
    /// gate and up fused as `ffn_gate_up_exps.weight`.
    pub fn moe_gemm_gguf_gate_up_gelu_mul_concat(
        input: *const f32,
        gate_up_weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        output: *mut c_void, // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,        // expert ffn dim (half of stored 2N)
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Fused MoE down-projection + topk reduction. Each (token, expert)
    /// row's weighted partial result is added into the [num_real_tokens,
    /// hidden] output via atomicAdd. Caller must pre-zero the output.
    /// Saves the explicit reshape+sum that the unfused down + topk
    /// reduction sequence requires.
    pub fn moe_gemm_gguf_down_reduce(
        input: *const f32,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f32,         // [num_real_tokens, size_n] pre-zeroed
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    /// Building block: device-side expert offsets from a sorted
    /// `expert_ids` array. Emits `expert_offsets[num_experts + 1]` such
    /// that `expert_offsets[e] = first index where expert_ids[i] >= e`
    /// (and `expert_offsets[num_experts] = M`). Per-expert pair counts
    /// are `expert_offsets[e+1] - expert_offsets[e]`.
    ///
    /// O(num_experts × log M) work; one block per expert, single-thread
    /// binary search. Used to unblock per-expert dispatch in
    /// `moe_gemm_gguf_*` for prefill batches where the current
    /// per-(token,expert) kernel emits an O(num_tokens × topk) grid.
    pub fn moe_expert_offsets(
        sorted_expert_ids: *const i32,
        expert_offsets: *mut i32,        // [num_experts + 1]
        m: i32,
        num_experts: i32,
        stream: i64,
    );

    /// Step 3: gather input rows for per-expert dispatch.
    /// Writes a contiguous [n_e, K] F16 buffer where row i is
    /// `inputs[sorted_token_ids[start+i] / topk]`. One CUDA block per
    /// output row, one thread per K element (strided).
    pub fn moe_gather_input_rows_f32_to_f16(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_e: i32,
        start: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Step 3: GELU·mul + scatter for per-expert dispatch.
    /// Input is the cuBLAS GEMM output `[n_e, 2N]` F16; for each row,
    /// applies `gelu_tanh(gate) * up` and scatters the resulting
    /// `[N]` row into the final F32 output at index
    /// `sorted_token_ids[start+i]`.
    pub fn moe_gelu_mul_scatter_f16_to_f32(
        in_f16: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        out_f32: *mut f32,
        n_e: i32,
        start: i32,
        n: i32,
        stream: i64,
    );

    /// Step 4 batched gather: writes a padded
    /// `[N_active, max_n_e, K]` F16 workspace from F32 inputs across
    /// ALL active experts in ONE launch. Each (act_idx, row) reads
    /// `sorted_token_ids[expert_offsets[active_expert_ids[act_idx]] + row]`,
    /// divides by topk, and copies the input row in. Padding rows
    /// (row ≥ n_e[act_idx]) are skipped — caller must pre-zero out.
    pub fn moe_batched_gather_input_rows_f32_to_f16(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_active: i32,
        max_n_e: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Step 5 (Q4_K MMA path): batched F32 → Q8_1 gather+quantize.
    /// Same dispatch contract as the F32→F16 gather, but writes Q8_1 blocks
    /// (`[N_active, max_n_e, K/32]` block_q8_1). Q8_1 is the input format
    /// consumed by the `mma.sync.m16n8k32.s8.s8.s32` Q4_K MMA kernel.
    pub fn moe_batched_gather_input_rows_f32_to_q81(
        inputs: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_q81: *mut core::ffi::c_void,
        n_active: i32,
        max_n_e: i32,
        k: i32,
        topk: i32,
        stream: i64,
    );

    /// Step 4 batched scatter: reads the padded GEMM output
    /// `[N_active, max_n_e, 2N]` F16, applies `gelu_tanh(gate)·up` per
    /// valid row, and scatters into `out[sorted_token_ids[...], :]`
    /// F32 — all active experts in ONE launch.
    pub fn moe_batched_gelu_mul_scatter_f16_to_f32(
        in_f16: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f32: *mut f32,
        n_active: i32,
        max_n_e: i32,
        n: i32,
        stream: i64,
    );

    /// F32 input variant of the GELU·mul + scatter — pairs with the
    /// Q4_K MMA kernel that writes F32 directly (no F16 rounding step,
    /// which compounds drift over 30+ cascading layers in real models).
    pub fn moe_batched_gelu_mul_scatter_f32_to_f32(
        in_f32: *const f32,
        sorted_token_ids: *const i32,
        active_expert_ids: *const i32,
        expert_offsets: *const i32,
        out_f32: *mut f32,
        n_active: i32,
        max_n_e: i32,
        n: i32,
        stream: i64,
    );

    /// Step 4 batched dequant: dequantizes N_active experts'
    /// `[rows_per_expert, cols]` Q4_K weight slabs into a contiguous
    /// `[n_active, rows_per_expert, cols]` F16 workspace in ONE launch.
    /// Replaces N_active separate dequant launches and the per-call
    /// scratch-buffer copies the host-loop variant required.
    /// `active_expert_ids` is a host-supplied i32 array indicating which
    /// experts to materialise (e.g. the sparse list of experts with at
    /// least one token assigned).
    pub fn moe_batched_dequant_q4k_f16(
        all_weights: *const core::ffi::c_void,
        active_expert_ids: *const i32,
        out_f16: *mut core::ffi::c_void,
        n_active: i32,
        rows_per_expert: i32,
        cols: i32,
        stream: i64,
    );

    /// Cast and copy: dst[i] = (f32) src[i] for n elements. Used to
    /// initialize the moe_gemm_gguf_down_reduce output buffer with the
    /// residual values so the post-MLP residual add is folded into the
    /// final atomicAdd accumulation.
    pub fn cast_init_f32_from_dtype(
        dst: *mut f32,
        src: *const c_void,
        n: i32,
        dtype: i32,                 // 0=f16, 1=bf16
        stream: i64,
    );

    /// Fused MoE routing: softmax → top-k → optional renorm in a single
    /// CUDA launch. Replaces ~6 candle ops (softmax_last_dim,
    /// arg_sort_last_dim, narrow, contiguous, gather, sum_keepdim,
    /// broadcast_div) per MoE layer.

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // ============== Dense GGUF MMVQ launchers (from mmvq_gguf.cu) ==============

    // BF16 output launchers
    pub fn launch_mmvq_gguf_q4_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // F32 output launchers
    pub fn launch_mmvq_gguf_q4_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // small_k launchers (ncols_dst=1 only, rows_per_block=4).
    // Caller must ensure nrows_x % 4 == 0 before calling.
    pub fn launch_mmvq_gguf_q4_0_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_1_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_0_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_1_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q8_0_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q2_k_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q3_k_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_k_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_k_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q6_k_bf16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);

    pub fn launch_mmvq_gguf_q4_0_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_1_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_0_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_1_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q8_0_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q2_k_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q3_k_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_k_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_k_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q6_k_f16_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);

    pub fn launch_mmvq_gguf_q4_0_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_1_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_0_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_1_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q8_0_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q2_k_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q3_k_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q4_k_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q5_k_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);
    pub fn launch_mmvq_gguf_q6_k_f32_plain_smallk(vx: *const c_void, vy: *const c_void, dst: *mut c_void, ncols_x: i32, nrows_x: i32, stride_col_y: i32, stride_col_dst: i32, stream: *mut c_void);

    // Quantize launchers (activation → Q8_1)
    pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    // Fused (up + gate + SwiGLU) MMVQ (Q4_K, F32 out, ncols=1).
    // See libs/candle/candle-kernels/src/mmvq_gguf.cu launch_mmvq_gguf_q4_k_f32_fused_silu.
    pub fn launch_mmvq_gguf_q4_k_f32_fused_silu(
        vx: *const c_void,
        vgate: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        stream: *mut c_void,
    );
    // GELU sibling of the fused (gate + up + activation) Q4_K MMVQ.
    // Used by dense decode with a fused GELU·mul FFN.
    pub fn launch_mmvq_gguf_q4_k_f32_fused_gelu(
        vx: *const c_void,
        vgate: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        stream: *mut c_void,
    );
    // BF16-output sibling of the fused (gate + up + silu) Q4_K MMVQ.
    // Writes BF16 directly to dst — saves the F32→BF16 cast that the
    // downstream residual broadcast_add would otherwise need.
    pub fn launch_mmvq_gguf_q4_k_bf16_fused_silu(
        vx: *const c_void,
        vgate: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        stream: *mut c_void,
    );

    // ============== Dense GGUF MMQ launchers (from mmq_gguf/) ==============

    // MMQ quantize launchers (f32 -> block_q8_1_mmq, 3 scale layouts)
    pub fn launch_mmq_quantize_q8_1_D4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_DS4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_D2S6(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    // MMQ matmul launchers (one per quant type)
    pub fn launch_mmq_gguf_q4_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q8_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q2_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q3_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q6_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );

    /// Fused MoE gate matmul + softmax + top-k. Combines the per-layer
    /// `logits = gate.forward(x)` (cublas SGEMV) and `topk_softmax(logits)`
    /// launches into one kernel. Loads `x` into shared memory once per
    /// token, then each warp lane computes a slice of the n_experts logits
    /// and the existing softmax+topk reduction proceeds in-place.
    /// `n_experts` ∈ {32, 64, 128, 256}; `hidden` ≤ 8192.

    /// Multi-block F32 GEMV for the MoE router gate matmul. Replaces a
    /// cublas SGEMV (overhead-bound at ~16us/call on 2048×128) with a
    /// hand-rolled kernel that splits experts across many blocks for
    /// full SM saturation. `hidden` must be a multiple of 32.

    /// Dense (non-MoE) gate+up+silu*mul fused kernel.
    /// `gate_w` and `up_w` are quantized [N, K]; `input` is F32 [K];
    /// `output` is F32 [N] pre-allocated.
    /// Replaces 3 launches (ffn_up matmul + narrow + fused_silu_mul) with
    /// 1 quantize + 1 fused matmul.
    pub fn dense_gate_up_silu_mul_v2(
        input: *const f32,
        gate_w: *const c_void,
        up_w: *const c_void,
        output: *mut f32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

}
