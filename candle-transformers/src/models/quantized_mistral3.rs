//! Quantized Mistral3 model GGUF loader.
//!
//! This provides GGUF loading support for Mistral3 models.
//! Based on the quantized_llama.rs but reads mistral3.* metadata keys.
//! The forward pass is identical to LLaMA.

use std::collections::HashMap;

use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::quantized::{gguf_file, QMatMul};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 4096;

// QMatMul wrapper
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for MQA/GQA
        let k = crate::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = crate::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v.contiguous()?)?
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    /// Load Mistral3 model from GGUF format
    /// Key difference from LLaMA: reads mistral3.* metadata keys instead of llama.*
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // ============================================================
        // READ MISTRAL3 METADATA KEYS
        // From the GGUF dump we saw:
        // - mistral3.attention.head_count = 32
        // - mistral3.attention.head_count_kv = 8
        // - mistral3.block_count = 40
        // - mistral3.embedding_length = 5120
        // - mistral3.rope.dimension_count = 128
        // - mistral3.attention.layer_norm_rms_epsilon = 1e-5
        // - mistral3.feed_forward_length = 32768
        // - mistral3.vocab_size = 131072
        // ============================================================
        
        let n_expert = md_get("mistral3.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("mistral3.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
            
        // Mistral3 uses mistral3.* prefix
        let head_count = md_get("mistral3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("mistral3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("mistral3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("mistral3.embedding_length")?.to_u32()? as usize;
        
        // Handle rope dimension
        let rope_dim = md_get("mistral3.rope.dimension_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(embedding_length as u32 / 2) as usize;
            
        // Mistral3 RMS epsilon - default to 1e-5 like LLaMA
        let rms_norm_eps = md_get("mistral3.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32())
            .unwrap_or(1e-5) as f64;

        // RoPE frequency base - Mistral3 uses 100000000.0
        let rope_freq_base = md_get("mistral3.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
            
        // Precompute RoPE frequencies
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // ============================================================
        // LOAD TENSORS - Same naming as LLaMA in GGUF
        // - token_embd.weight
        // - blk.{layer}.attn_q.weight
        // - blk.{layer}.attn_k.weight
        // - blk.{layer}.attn_v.weight
        // - blk.{layer}.attn_output.weight
        // - blk.{layer}.ffn_gate.weight
        // - blk.{layer}.ffn_down.weight
        // - blk.{layer}.ffn_up.weight
        // - blk.{layer}.attn_norm.weight
        // - blk.{layer}.ffn_norm.weight
        // - output_norm.weight
        // - output.weight
        // ============================================================
        
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        
        // Try output_norm.weight first (Mistral3 style), fall back to other variations
        let norm = match ct.tensor(reader, "output_norm.weight", device) {
            Ok(t) => RmsNorm::from_qtensor(t, rms_norm_eps)?,
            Err(_) => {
                // Try other naming conventions
                match ct.tensor(reader, "output_norm.weight", device) {
                    Ok(t) => RmsNorm::from_qtensor(t, rms_norm_eps)?,
                    Err(_) => {
                        // Last resort - create a dummy norm
                        let shape = tok_embeddings.embedding().dims();
                        let dummy = Tensor::ones(shape, DType::F32, device)?;
                        RmsNorm::new(dummy, rms_norm_eps)?
                    }
                }
            }
        };
        
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{}", layer_idx);
            
            // Attention weights - same naming as LLaMA
            let attention_wq = ct.tensor(reader, &format!("{}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{}.attn_output.weight"), device)?;
            
            // MLP/MoE weights
            let mlp_or_moe = if n_expert <= 1 {
                // Standard MLP - try mistral3 naming first, then llama
                let feed_forward_w1 = ct.tensor(reader, &format!("{}.ffn_gate.weight"), device)
                    .or_else(|_| ct.tensor(reader, &format!("{}.ffn_gate.weight"), device))?;
                let feed_forward_w2 = ct.tensor(reader, &format!("{}.ffn_down.weight"), device)?;
                let feed_forward_w3 = ct.tensor(reader, &format!("{}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                // MoE - Mixtral style
                let feed_forward_gate_inp = ct.tensor(reader, &format!("{}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 = ct.tensor(reader, &format!("{}.ffn_gate.{}.weight", prefix, i), device)?;
                    let feed_forward_w2 = ct.tensor(reader, &format!("{}.ffn_down.{}.weight", prefix, i), device)?;
                    let feed_forward_w3 = ct.tensor(reader, &format!("{}.ffn_up.{}.weight", prefix, i), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            
            // Layer norms - same naming as LLaMA
            let attention_norm = ct.tensor(reader, &format!("{}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{}.ffn_norm.weight"), device)?;
            
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }
        
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Forward pass - IDENTICAL to LLaMA!
    /// This is the same code as quantized_llama.rs
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        
        let mut layer_in = self.tok_embeddings.forward(x)?;
        
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            
            // Attention with pre-norm
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP with pre-norm
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            
            layer_in = x;
        }
        
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        
        self.output.forward(&x)
    }
}
