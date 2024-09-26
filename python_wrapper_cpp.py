import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./tensor_lib.so')  # Make sure this path is correct

# Define the function signatures
lib.create_tensor_float.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.create_tensor_float.restype = ctypes.c_void_p

lib.free_tensor_float.argtypes = [ctypes.c_void_p]
lib.free_tensor_float.restype = None

lib.print_tensor_float.argtypes = [ctypes.c_void_p]
lib.print_tensor_float.restype = None

lib.set_tensor_data_float.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.set_tensor_data_float.restype = None

lib.tensor_add_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.tensor_add_float.restype = None

lib.tensor_multiply_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.tensor_multiply_float.restype = None

def create_tensor(shape):
    shape_array = (ctypes.c_int * len(shape))(*shape)
    return lib.create_tensor_float(shape_array, len(shape))

def free_tensor(tensor):
    lib.free_tensor_float(tensor)

def print_tensor(tensor):
    lib.print_tensor_float(tensor)

def set_tensor_data(tensor, data):
    data_array = (ctypes.c_float * len(data))(*data)
    lib.set_tensor_data_float(tensor, data_array, len(data))

class Tensor_C(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("shape", ctypes.POINTER(ctypes.c_int)),
                ("ndim", ctypes.c_int),
                ("size", ctypes.c_int)]

class Tensor:
    def __init__(self, c_pointer):
        self.c_pointer = c_pointer

    def __add__(self, other):
        result = create_tensor(self.shape())
        lib.tensor_add_float(self.c_pointer, other.c_pointer, result)
        return Tensor(result)

    def __mul__(self, other):
        result = create_tensor(self.shape())
        lib.tensor_multiply_float(self.c_pointer, other.c_pointer, result)
        return Tensor(result)

    def shape(self):
        tensor = ctypes.cast(self.c_pointer, ctypes.POINTER(Tensor_C)).contents
        return [tensor.shape[i] for i in range(tensor.ndim)]

def main():
    print("Debug: Starting main")
    
    # Create tensors
    shape = [2, 2]
    tensor_a = create_tensor(shape)
    tensor_b = create_tensor(shape)

    # Set data
    set_tensor_data(tensor_a, [1.0, 2.0, 3.0, 4.0])
    set_tensor_data(tensor_b, [5.0, 6.0, 7.0, 8.0])

    # Print tensors
    print("Tensor a:")
    print_tensor(tensor_a)
    print("Tensor b:")
    print_tensor(tensor_b)

    # Wrap the C pointers in our Python Tensor class
    a = Tensor(tensor_a)
    b = Tensor(tensor_b)

    # Now we can use Python's + operator
    c = a + b
    print("Tensor c (a + b):")
    print_tensor(c.c_pointer)

    # Multiply tensors
    d = a * b
    print("Tensor d (a * b):")
    print_tensor(d.c_pointer)

    # Free tensors
    free_tensor(tensor_a)
    free_tensor(tensor_b)
    free_tensor(c.c_pointer)
    free_tensor(d.c_pointer)

    print("Debug: Ending main")

if __name__ == "__main__":
    main()