from python import Python

fn main() raises:
    var builtins = Python.import_module("builtins")
    var py_dict = Python.dict()
    py_dict["final_logit_softcapping"] = 30.0

    print("builtins.bool 30.0:", builtins.bool(py_dict.get("final_logit_softcapping")))

    var soft_cap = Float32(Float64(py=builtins.getattr(py_dict, "get")("final_logit_softcapping", 0.0)))
    print("soft_cap:", soft_cap)
