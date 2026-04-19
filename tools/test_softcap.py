import ctypes
ctypes.cdll.LoadLibrary('.venv/lib/python3.12/site-packages/modular/lib/libKGENCompilerRTShared.so')
from mogemma import SyncGemmaModel
model = SyncGemmaModel('google/gemma-4-E2B-it')
print("Keys:", model._llm.keys())
print("Value:", model._llm.get("final_logit_softcapping"))
