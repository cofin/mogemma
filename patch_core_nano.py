import re

with open("src/mo/mogemma/core.mojo", "r") as f:
    content = f.read()

# 1. Add imports for _gpu functions in core.mojo
import_line = "from mogemma.layers import ("
if import_line in content:
    content = content.replace(import_line, "from mogemma.layers import forward_nano_layer_gpu, \\\n" + import_line)

# 2. Extract _forward_nano_layers_runtime
m = re.search(r"fn _forward_nano_layers_runtime\(.*?\nfn _forward_step_nano_runtime\(", content, re.DOTALL)
if m:
    cpu_code = m.group(0).replace("fn _forward_step_nano_runtime(", "")
    
    gpu_code = cpu_code.replace("_forward_nano_layers_runtime", "_forward_nano_layers_gpu_runtime")
    gpu_code = gpu_code.replace("forward_nano_layer(", "forward_nano_layer_gpu(")
    
    # Insert before _forward_step_nano_runtime
    content = content.replace("fn _forward_step_nano_runtime(", gpu_code + "\nfn _forward_step_nano_runtime(")

# 3. Extract _forward_step_nano_runtime
m2 = re.search(r"fn _forward_step_nano_runtime\(.*?\nfn _apply_runtime_init_options\(", content, re.DOTALL)
if m2:
    cpu_code2 = m2.group(0).replace("fn _apply_runtime_init_options(", "")
    
    gpu_code2 = cpu_code2.replace("_forward_step_nano_runtime", "_forward_step_nano_gpu_runtime")
    gpu_code2 = gpu_code2.replace("_forward_nano_layers_runtime(", "_forward_nano_layers_gpu_runtime(")
    
    # Insert before _apply_runtime_init_options
    content = content.replace("fn _apply_runtime_init_options(", gpu_code2 + "\nfn _apply_runtime_init_options(")

# 4. Update step_mojo to dispatch
step_nano_call = """        if step_backend == "cuda":
            raise Error("CUDA not supported for nano")

        _forward_step_nano_runtime("""

new_step_nano_call = """        if step_backend == "cuda":
            _forward_step_nano_gpu_runtime(
                out_logits_ptr,
                token_id,
                pos,
                runtime_obj,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                per_layer_dim,
                vocab_size,
                freqs_cos_ptr,
                freqs_sin_ptr,
                kv_cache_k_ptr,
                kv_cache_v_ptr,
                max_seq_len,
                kv_share_start,
                scratch_ptr
            )
        else:
            _forward_step_nano_runtime("""

if step_nano_call in content:
    content = content.replace(step_nano_call, new_step_nano_call)
    
    # close the else block
    content = content.replace("kv_share_start,\n            scratch_ptr\n        )", "kv_share_start,\n                scratch_ptr\n            )")

with open("src/mo/mogemma/core.mojo", "w") as f:
    f.write(content)
