from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

fn _ensure_step_logits(logits_obj: PythonObject, np: PythonObject) raises -> PythonObject:
    var logits = np.asarray(logits_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(logits.shape)) != 1:
        raise Error("step output must be a 1D float32 tensor")
    if Int(py=logits.shape[0]) <= 0:
        raise Error("step output must contain at least one element")
    return logits


fn _ensure_embedding_matrix(embeddings_obj: PythonObject, expected_rows: Int, np: PythonObject) raises -> PythonObject:
    var embeddings = np.asarray(embeddings_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(embeddings.shape)) != 2:
        raise Error("generate_embeddings output must be a 2D float32 matrix")
    if Int(py=embeddings.shape[0]) != expected_rows:
        raise Error("generate_embeddings output row count does not match inputs")
    if Int(py=embeddings.shape[1]) != 768:
        raise Error("generate_embeddings output must have 768 columns")
    return embeddings


fn step_mojo(
    llm: PythonObject,
    token_id_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    var token_id = Int(py=token_id_obj)
    var temp = Float32(py=temp_obj)
    var top_k = Int(py=top_k_obj)
    var top_p = Float32(py=top_p_obj)
    
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    if builtins.hasattr(llm, "step"):
        var logits = llm.step(token_id, temp, top_k, top_p)
        return _ensure_step_logits(logits, np)

    if builtins.hasattr(llm, "next_token"):
        var logits = llm.next_token(token_id, temp, top_k, top_p)
        return _ensure_step_logits(logits, np)

    raise Error("step requires llm.step or llm.next_token")

fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var np_input = np.asarray(input_array, dtype=np.int32)
    var batch_size = Int(py=np_input.shape[0])

    if builtins.hasattr(llm, "generate_embeddings"):
        var embeddings = llm.generate_embeddings(np_input)
        return _ensure_embedding_matrix(embeddings, batch_size, np)

    if builtins.hasattr(llm, "encode"):
        var embeddings = llm.encode(np_input)
        return _ensure_embedding_matrix(embeddings, batch_size, np)

    raise Error("generate_embeddings requires llm.generate_embeddings or llm.encode")

fn init_model_mojo(
    model_path_obj: PythonObject
) raises -> PythonObject:
    var model_path = String(model_path_obj)
    if model_path == "":
        raise Error("init_model requires a non-empty model_path")

    var pathlib = Python.import_module("pathlib")
    var resolved_path = pathlib.Path(model_path)
    if not resolved_path.exists():
        var msg = "init_model failed: model path does not exist: "
        msg = msg + model_path
        raise Error(msg)
    if not resolved_path.is_dir():
        var msg = "init_model failed: model path is not a directory: "
        msg = msg + model_path
        raise Error(msg)

    try:
        var max_llm = Python.import_module("max.entrypoints.llm")
        var pipeline_config = Python.import_module("max.pipelines").PipelineConfig(
            model_path=model_path
        )
        var llm = max_llm.LLM(pipeline_config)
        return llm
    except e:
        print("Mojo Error loading model:", e)
        var msg = "init_model failed while loading model from: "
        msg = msg + model_path
        msg = msg + " ("
        msg = msg + String(e)
        msg = msg + ")"
        raise Error(msg)

@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[init_model_mojo]("init_model")
        b.def_function[generate_embeddings_mojo]("generate_embeddings")
        b.def_function[step_mojo]("step")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
