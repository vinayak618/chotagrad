import ctypes
import numpy as np

# Load the C++ library
lib = ctypes.CDLL("./tensor_lib.so")

# Define the Tensor structure for ctypes
class CTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("shape", ctypes.POINTER(ctypes.c_int)),
                ("ndim", ctypes.c_int),
                ("size", ctypes.c_int)]

# Set up function signatures
lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.create_tensor.restype = ctypes.c_void_p

lib.free_tensor.argtypes = [ctypes.c_void_p]
lib.free_tensor.restype = None

lib.tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.tensor_add.restype = None

lib.tensor_multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.tensor_multiply.restype = None

lib.print_tensor.argtypes = [ctypes.c_void_p]
lib.print_tensor.restype = None

class Tensor:
    def __init__(self, data):
        # print("Debug: Initializing Tensor in Python")
        if isinstance(data, np.ndarray):
            self.np_array = data.astype(np.float32)
        elif isinstance(data, list):
            self.np_array = np.array(data, dtype=np.float32)
        else:
            raise ValueError("Input must be a NumPy array or a list")

        shape = self.np_array.shape
        ndim = self.np_array.ndim
        c_shape = (ctypes.c_int * ndim)(*shape)
        
        # print(f"Debug: Creating tensor with shape {shape} and ndim {ndim}")
        self.c_tensor = lib.create_tensor(c_shape, ndim)
        if not self.c_tensor:
            raise MemoryError("Failed to create tensor in C++")
        
        # Create a ctypes array from the numpy array
        c_array = (ctypes.c_float * self.np_array.size)(*self.np_array.flatten())
        
        # Copy the data to the C++ tensor
        ctypes.memmove(CTensor.from_address(self.c_tensor).data, c_array, self.np_array.nbytes)
        # print("Debug: Finished initializing Tensor")

    def __del__(self):
        # print("Debug: Deleting Tensor in Python")
        if hasattr(self, 'c_tensor') and self.c_tensor:
            lib.free_tensor(self.c_tensor)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError("Can only add Tensor objects")
        
        result = Tensor(np.zeros_like(self.np_array))  # Create a new tensor with the same shape
        lib.tensor_add(self.c_tensor, other.c_tensor, result.c_tensor)
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError("Can only multiply Tensor objects")
        
        result = Tensor(np.zeros_like(self.np_array))  # Create a new tensor with the same shape
        lib.tensor_multiply(self.c_tensor, other.c_tensor, result.c_tensor)
        return result

    def print(self):
        # print("Debug: Calling print method in Python")
        lib.print_tensor(self.c_tensor)
        # print("Debug: Finished print method in Python")

# Example usage
if __name__ == "__main__":
    print("Debug: Starting main")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    
    print("Tensor a:")
    a.print()
    print("Tensor b:")
    b.print()
    
    c = a + b
    print("a + b:")
    c.print()
    
    d = a * b
    print("a * b:")
    d.print()
    
    print("Debug: Finished main")