import obstore as obs
store = obs.store.GCSStore("gemma-data", config={"skip_signature": "true"})
gen = obs.list(store, "checkpoints/gemma3-270m-it/")
item = list(next(gen))[0]
path = item["path"]
print(f"Path: {path}")
result = obs.get(store, path)
b = result.bytes()
print("Type:", type(b))
print("Dir:", dir(b))
print("Isinstance bytes:", isinstance(b, bytes))
print("Methods available:", [m for m in dir(b) if not m.startswith('_')])
try:
    print("to_bytes():", type(b.to_bytes()))
except Exception as e:
    pass
