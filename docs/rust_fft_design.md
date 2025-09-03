# Rust FFT Processing Module Design

## Overview
Optional Rust module for ultra-high performance FFT processing to further accelerate real-time spectrogram generation. This is designed to be a future enhancement when maximum performance is needed.

## Performance Targets
- **Sub-millisecond spectrogram generation** for typical RF signals
- **10-100x faster** than current Python implementation
- **Memory efficient** with zero-copy operations where possible
- **SIMD optimized** for modern processors

## Architecture

### Rust Crate Structure
```
torchsig_fft/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main library interface
│   ├── fft.rs              # Core FFT implementations
│   ├── spectrogram.rs      # Spectrogram computation
│   ├── windowing.rs        # Window functions (Hann, Hamming, etc.)
│   └── python_bindings.rs  # PyO3 Python bindings
└── benches/
    └── performance.rs      # Performance benchmarks
```

### Dependencies
```toml
[dependencies]
pyo3 = "0.20"
numpy = "0.20"
rustfft = "6.1"      # High-performance FFT
rayon = "1.7"        # Data parallelism
num-complex = "0.4"  # Complex number support
```

### Core API Design

```rust
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};

#[pyclass]
pub struct RustSpectrogramProcessor {
    // Pre-allocated FFT planner for different sizes
    fft_cache: FftCache,
    // Thread pool for parallel processing
    thread_pool: rayon::ThreadPool,
}

#[pymethods]
impl RustSpectrogramProcessor {
    #[new]
    pub fn new(max_fft_size: usize, num_threads: Option<usize>) -> Self {
        // Initialize with optimized thread count and FFT cache
    }
    
    #[pyo3(signature = (signal, n_fft, hop_length, window_type, normalize=true))]
    pub fn compute_spectrogram<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<num_complex::Complex<f32>>,
        n_fft: usize,
        hop_length: usize,
        window_type: &str,
        normalize: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        // Ultra-fast spectrogram computation with SIMD optimization
    }
    
    pub fn compute_spectrogram_streaming(
        &self,
        signal_chunk: PyReadonlyArray1<num_complex::Complex<f32>>,
        // ... parameters
    ) -> PyResult<()> {
        // Real-time streaming spectrogram for continuous processing
    }
}
```

### Key Optimizations

#### 1. FFT Optimization
```rust
use rustfft::{FftPlanner, Fft};

struct FftCache {
    planners: HashMap<usize, Arc<dyn Fft<f32>>>,
}

impl FftCache {
    fn get_or_create(&mut self, n_fft: usize) -> Arc<dyn Fft<f32>> {
        // Cache FFT planners for different sizes
        // Use SIMD-optimized implementations
    }
}
```

#### 2. Parallel Processing
```rust
use rayon::prelude::*;

fn compute_parallel_stft(
    signal: &[Complex<f32>],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Vec<Vec<Complex<f32>>> {
    // Parallel computation of STFT frames
    (0..num_frames)
        .into_par_iter()
        .map(|frame_idx| {
            // Process each frame in parallel
            compute_fft_frame(signal, frame_idx, n_fft, hop_length, window)
        })
        .collect()
}
```

#### 3. SIMD Window Functions
```rust
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn apply_hann_window_avx2(signal: &mut [f32], window: &[f32]) {
    // Vectorized window application using AVX2
    for (signal_chunk, window_chunk) in 
        signal.chunks_exact_mut(8).zip(window.chunks_exact(8)) {
        
        let sig = _mm256_loadu_ps(signal_chunk.as_ptr());
        let win = _mm256_loadu_ps(window_chunk.as_ptr());
        let result = _mm256_mul_ps(sig, win);
        _mm256_storeu_ps(signal_chunk.as_mut_ptr(), result);
    }
}
```

### Python Integration

```python
# torchsig/utils/rust_fft.py
from typing import Optional
import numpy as np

try:
    from torchsig_fft import RustSpectrogramProcessor
    RUST_FFT_AVAILABLE = True
except ImportError:
    RUST_FFT_AVAILABLE = False
    RustSpectrogramProcessor = None

class FastSpectrogramProcessor:
    def __init__(self, use_rust: bool = True, max_fft_size: int = 8192):
        self.use_rust = use_rust and RUST_FFT_AVAILABLE
        
        if self.use_rust:
            self.rust_processor = RustSpectrogramProcessor(
                max_fft_size=max_fft_size,
                num_threads=None  # Auto-detect optimal thread count
            )
    
    def compute_spectrogram(
        self,
        signal: np.ndarray,
        n_fft: int,
        hop_length: int,
        window: str = 'hann',
        normalize: bool = True
    ) -> np.ndarray:
        
        if self.use_rust:
            # Ultra-fast Rust implementation
            return self.rust_processor.compute_spectrogram(
                signal, n_fft, hop_length, window, normalize
            )
        else:
            # Fallback to PyTorch/SciPy
            return self._fallback_spectrogram(signal, n_fft, hop_length, window)
```

### Integration with Real-time Service

```python
# In services/realtime_spectrogram.py
from torchsig.utils.rust_fft import FastSpectrogramProcessor

class RealtimeSpectrogramService:
    def __init__(self):
        # Try to use Rust FFT for maximum performance
        self.spectrogram_processor = FastSpectrogramProcessor(use_rust=True)
        # ... rest of initialization
    
    def compute_spectrogram_fast(self, iq_data: np.ndarray, params: SpectrogramParams):
        # Use Rust implementation if available, fallback to Python
        return self.spectrogram_processor.compute_spectrogram(
            iq_data, params.n_fft, params.hop_length, params.window
        )
```

## Performance Expectations

### Benchmarks (Estimated)
| Implementation | Signal Length | FFT Size | Computation Time |
|---------------|---------------|----------|------------------|
| Current Scipy | 1M samples    | 1024     | ~50ms           |
| Current PyTorch| 1M samples    | 1024     | ~15ms (GPU)     |
| **Rust + SIMD** | 1M samples  | 1024     | **~2ms**        |
| **Rust Parallel**| 1M samples  | 1024     | **~0.5ms**      |

### Memory Usage
- **Zero-copy** where possible using PyO3
- **In-place** FFT computations
- **Optimized allocation** patterns

## Implementation Priority

**Phase 1 (Optional Enhancement)**
- Basic Rust FFT implementation with PyO3 bindings
- Integration with existing Python service
- Performance benchmarking

**Phase 2 (Future)**  
- SIMD optimizations for specific architectures
- Streaming/real-time processing capabilities
- Multi-GPU support for batch processing

## Build Integration

```toml
# Add to pyproject.toml for optional Rust acceleration
[tool.setuptools.packages.find]
exclude = ["torchsig_fft*"]  # Exclude Rust crate from Python package

[build-system]
requires = ["setuptools", "wheel", "setuptools-rust"]
build-backend = "setuptools.build_meta"

[[tool.setuptools-rust.ext-modules]]
target = "torchsig.utils._rust_fft"
path = "torchsig_fft/Cargo.toml"
binding = "PyO3"
```

This design provides a path for ultra-high performance when needed, while maintaining compatibility with the current Python-based system.