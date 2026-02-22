from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

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
    return np.zeros(256000, dtype=np.float32)

fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    print("Mojo: generate_embeddings called")
    var np = Python.import_module("numpy")
    var batch_size = Int(py=input_array.shape[0])
    if batch_size < 1:
        batch_size = 1
    return np.random.rand(batch_size, 768).astype(np.float32)

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
