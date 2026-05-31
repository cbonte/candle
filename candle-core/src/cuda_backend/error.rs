use crate::{DType, Layout};

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error(transparent)]
    Compiler(#[from] cudarc::nvrtc::CompileError),

    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),

    #[error(transparent)]
    Curand(#[from] cudarc::curand::result::CurandError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: cudarc::driver::DriverError,
        module_name: String,
    },
}

impl From<CudaError> for crate::Error {
    fn from(val: CudaError) -> Self {
        crate::Error::Cuda(Box::new(val)).bt()
    }
}

/// Lift cudarc DriverError directly into our Error so call sites using
/// `?` against APIs that return `Result<_, DriverError>` (e.g.
/// `dev.cuda_stream().context().bind_to_thread()?`) compile without
/// the cudnn feature being enabled. The cudnn module had an
/// equivalent impl gated behind `feature = "cudnn"`, which was the
/// pre-existing reason `cargo test -p candle-core --features cuda`
/// failed at `quantized/cuda.rs:1688` for non-cudnn builds.
impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<CudaError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Cuda(Box::new(e.into())).bt())
    }
}
