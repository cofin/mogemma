from memory import alloc, UnsafePointer

struct MyStruct(Movable):
    var x: Int
    fn __init__(out self):
        self.x = 42

fn main() raises:
    var ptr = alloc[MyStruct](1)
    
    var s = MyStruct()
    ptr.init_pointee_move(s^)
    print(ptr[].x)
    
    ptr.destroy_pointee()
    ptr.free()
