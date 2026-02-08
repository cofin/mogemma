from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    # llm is the MAX LLM instance passed back from Python
    print("Mojo: generate_embeddings called with llm instance")
    
    # In a real implementation:
    # var results = llm.generate(input_array)
    # return results
    
    var np = Python.import_module("numpy")
    return np.random.rand(1, 768).astype(np.float32)

fn init_model_mojo(
    model_path_obj: PythonObject
) raises -> PythonObject:
    var model_path = String(model_path_obj)
    print("Mojo: Initializing Gemma 3 model from", model_path)
    
    try:
        var max_llm = Python.import_module("max.entrypoints.llm")
        var pipeline_config = Python.import_module("max.pipelines").PipelineConfig(
            model_path=model_path
        )
        var llm = max_llm.LLM(pipeline_config)
        return llm
    except e:
        print("Mojo Error loading model:", e)
        # Convert error to string to raise
        raise Error(String(e))

@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[init_model_mojo]("init_model")
        b.def_function[generate_embeddings_mojo]("generate_embeddings")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
