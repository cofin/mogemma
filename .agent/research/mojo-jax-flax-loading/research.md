# Research: Loading JAX/Flax Formats Natively in Mojo

## Executive Summary

Loading JAX/Flax native formats (Orbax/OCDBT and TensorStore/Zarr) directly into Mojo poses significant architectural challenges because these formats are chunked and compressed. Unlike Safetensors, they cannot be zero-copy memory-mapped (`mmap`). 

Native loading in Mojo would require either a massive C++/Python interop bridging layer or a full native implementation of the Zarr and OCDBT specifications, including decompression (e.g. zstd/blosc). The current conversion-to-safetensors approach is the most efficient and memory-safe path for production inference.

## Current Python Implementation

Currently in `mogemma`, Orbax checkpoints are loaded in Python using `tensorstore`:
- `src/py/mogemma/orbax_loader.py` opens the OCDBT store via `tensorstore`.
- It iterates through the Zarr arrays and reads them into fully uncompressed, contiguous `numpy.ndarray` buffers (`np.ascontiguousarray`).
- The raw memory pointers (`ctypes.data`) are then passed to Mojo via the FFI metadata dictionary.

**Drawback:** This requires loading the *entire* model into RAM simultaneously. For a 27B parameter model, this requires over 50GB of RAM just to hold the weights (at float16), causing OOM (Out Of Memory) errors on consumer hardware.

## Mojo Native Options

As of Mojo v0.26.1+, the standard library (`std`) is lean and does not include:
1. **Zarr specification support** (for chunked N-dimensional arrays).
2. **OCDBT/MessagePack parsers** (for Orbax manifests and key-value stores).
3. **Decompression libraries** (like `zstd` or `blosc` commonly used in Zarr).

To implement this purely natively in Mojo, one would have to:
- Write an OCDBT key-value parser to read the B-tree format.
- Write a Zarr metadata parser (JSON) and chunk loader.
- Bind to native C decompression libraries (e.g., `libzstd`) via Mojo's FFI.
- Allocate contiguous memory buffers and decompress chunks into them.

*This is a massive engineering effort with low ROI compared to using the existing Python ecosystem for conversion.*

## Mojo Python Interop Options

Mojo has robust Python interoperability (`python.Python`), which means the Python `tensorstore` library can technically be invoked directly from Mojo:

```mojo
from python import Python

fn load_orbax(model_path: String) raises:
    var ts = Python.import_module("tensorstore")
    var np = Python.import_module("numpy")
    # Execute the tensorstore.open(...) logic
```

**Pros:**
- Shifts the orchestration logic into Mojo.
- Allows direct copying of chunks into pre-allocated Mojo `Tensor` or `UnsafePointer` buffers chunk-by-chunk, potentially avoiding the intermediate full-model `numpy` allocation.

**Cons:**
- Executes the exact same Python C-extension (`tensorstore`) under the hood, requiring the Python GIL.
- Does not solve the fundamental problem: the arrays are compressed and chunked. You still must allocate RAM for the uncompressed weights. It does not enable zero-copy `mmap` inference.

## Mojo C/C++ FFI Options

TensorStore is originally a C++ library.
- Mojo's C FFI can interoperate with C++ libraries that expose C wrappers.
- We could write a C-wrapper for `libtensorstore` and call it natively from Mojo.
- **Pros:** Bypasses Python and the GIL completely.
- **Cons:** Extremely high build complexity (compiling TensorStore as a static/shared C library across platforms). Still requires full memory allocation for uncompressed weights.

## Performance & Memory Implications

The critical difference between Safetensors and Orbax/Zarr is **Memory Mapping (`mmap`)**:
- **Safetensors (Current Target):** The data is stored uncompressed and contiguous on disk. The OS maps the file into memory (`mmap`). The weights are paged into RAM on-demand, allowing instantaneous model loading and minimal RAM footprint.
- **Orbax/Zarr (JAX/Flax):** The data is chunked and compressed. To perform matrix multiplications in a high-performance transformer, the weights must be contiguous in memory. Therefore, the chunks must be decompressed and concatenated into RAM before inference begins.
  - *Startup Latency:* Reading, decompressing, and concatenating 50GB of chunks takes significant time.
  - *Memory Overhead:* You need physical RAM equal to the fully uncompressed model size. You cannot rely on OS paging to manage memory limits gracefully.

## Recommended Approach

**Do not implement native JAX/Flax (Orbax/Zarr) loading for inference in Mojo.**

1. **Retain the Pre-conversion Pipeline:** The current architectural decision to convert Orbax checkpoints to `.safetensors` once (via `src/py/mogemma/convert.py`) and then use zero-copy `mmap` for inference is correct. It is the only way to achieve instant startup and low memory overhead in a Mojo-native environment.
2. **Optimize the Python Converter (If Needed):** If `orbax_loader.py` causes OOM errors during conversion, refactor it to process and save tensors to `.safetensors` shard-by-shard, rather than loading the entire Orbax model into RAM at once.
3. **Mojo's Role:** Mojo should focus strictly on executing the inference math on contiguous, uncompressed memory pointers (safetensors). Rely on Python tools (`tensorstore`) for the heavy lifting of parsing and transforming the Google GCS Orbax formats offline.