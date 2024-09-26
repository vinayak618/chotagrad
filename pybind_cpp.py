import tensor_lib

# Create a 2x3 tensor
t1 = tensor_lib.Tensor([2, 3])
t1.random()  # Fill with random values

# Create another 2x3 tensor
t2 = tensor_lib.Tensor([2, 3])
t2.fill(2.0)  # Fill with 2.0

# Add tensors
t3 = t1.add(t2)

# Print sum
print(t3.sum())

# Access data and shape
print(t3.data)
print(t3.shape)