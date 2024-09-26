import ctypes
import numpy as np

# Load the C library
lib = ctypes.CDLL("./tensor_lib.so")  # Make sure to compile the C code into a shared library

# Define the Tensor structure for ctypes
class CTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("shape", ctypes.POINTER(ctypes.c_int)),
                ("ndim", ctypes.c_int),
                ("size", ctypes.c_int)]

# Set up function signatures
lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.create_tensor.restype = ctypes.POINTER(CTensor)

lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.free_tensor.restype = None

lib.tensor_add.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.tensor_add.restype = None

lib.tensor_multiply.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.tensor_multiply.restype = None

lib.print_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.print_tensor.restype = None

class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.np_array = data.astype(np.float32)
        elif isinstance(data, list):
            self.np_array = np.array(data, dtype=np.float32)
        else:
            raise ValueError("Input must be a NumPy array or a list")

        shape = self.np_array.shape
        ndim = self.np_array.ndim
        c_shape = (ctypes.c_int * ndim)(*shape)
        
        self.c_tensor = lib.create_tensor(c_shape, ndim)
        if not self.c_tensor:
            raise MemoryError("Failed to create tensor in C")
        
        # Create a ctypes array from the numpy array
        c_array = (ctypes.c_float * self.np_array.size)(*self.np_array.flatten())
        
        # Copy the data to the C tensor
        ctypes.memmove(self.c_tensor.contents.data, c_array, self.np_array.nbytes)

    def __del__(self):
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
        lib.print_tensor(self.c_tensor)

# Example usage
if __name__ == "__main__":
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