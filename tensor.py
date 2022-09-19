import torch
import numpy as np
"""
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. 
In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data
"""

"""
Initializing a tensor
"""

# From a data 
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# From np array
np_array = np.array(data)
tensor_array = torch.from_numpy(np_array)

# from another tensor
x_ones = torch.ones_like(x_data) #create tensors with all 1 as values but keep the property of x_data
x_rand = torch.rand_like(x_data, dtype=float) #create tensors with all 1 as values but keep the property of x_data and change the type of element in tensor to float

# with the pre-defined shape
shape = (2,3)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)


"""
Attribute of tensors
"""
tensor = torch.rand(2,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


"""
Operations on tensor
"""
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


# Indexing and slicing like numpy
tensor =  torch.ones(4,4)
print(f"tensor: {tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
# assign all value in one column
tensor[:,1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1) #join from left to right
print(t1)
t2 = torch.cat([tensor, tensor, tensor], dim=0) #join from above to below
print(t2)

# Arithmetic operations
# This computes the matrix multiplication(dot product) between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

"""
Single-element tensors If you have a one-element tensor, for 
example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():
"""
agg = tensor.sum()
# Convert single element tensor to value
agg_item = agg.item() 
print(agg_item, type(agg_item))


"""
An in-place operation is an operation that changes directly the content of a given Tensor without making a copy.
Inplace operations in pytorch are always postfixed with a _, like .add_() or .scatter_(). Python operations like += or *= are also inplace operations.
"""

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


"""
Convert between tensor and numpy is swallow copy, it mean any change in tensor will make impact in numpy and vice versa
"""
# Tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

#Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")





