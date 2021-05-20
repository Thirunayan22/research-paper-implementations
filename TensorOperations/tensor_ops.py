import torch


## INITIALIZING TENSORS ###
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,requires_grad=True,device=device)

x_empty=torch.empty(3,3)
x_eye = torch.eye(3,3)
x_rand = torch.rand(3,3)
x_lin = torch.linspace(3,3)
x_range = torch.arange(4)
print("X EMPTY",x_empty)
print(x_eye)
print(x_rand)
print(x_lin)
print(x_range)
print("IDENTITY ")

"""
Walk through of a lot of different useful Tensor Operations, where we
go through what I think are four main parts in:
1. Initialization of a Tensor
2. Tensor Mathematical Operations and Comparison
3. Tensor Indexing
4. Tensor Reshaping
But also other things such as setting the device (GPU/CPU) and converting
between different types (int, float etc) and how to convert a tensor to an
numpy array and vice-versa.
Programmed by Aladdin Persson
* 2020-06-27: Initial coding
"""

import torch

# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!

# Initializing a Tensor in this case of shape 2x3 (2 rows, 3 columns)
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# A few tensor attributes
print(
    f"Information about tensor: {my_tensor}"
)  # Prints data of the tensor, device and grad info
print(
    "Type of Tensor {my_tensor.dtype}"
)  # Prints dtype of the tensor (torch.float32, etc)
print(
    f"Device Tensor is on {my_tensor.device}"
)  # Prints cpu/cuda (followed by gpu number)
print(f"Shape of tensor {my_tensor.shape}")  # Prints shape, in this case 2x3
print(f"Requires gradient: {my_tensor.requires_grad}")  # Prints true/false

# Other common initialization methods (there exists a ton more)
x = torch.empty(size=(3, 3))  # Tensor of shape 3x3 with uninitialized data
x = torch.zeros((3, 3))  # Tensor of shape 3x3 with values of 0
x = torch.rand(
    (3, 3)
)  # Tensor of shape 3x3 with values from uniform distribution in interval [0,1)
x = torch.ones((3, 3))  # Tensor of shape 3x3 with values of 1
x = torch.eye(5, 5)  # Returns Identity Matrix I, (I <-> Eye), matrix of shape 2x3
x = torch.arange(
    start=0, end=5, step=1
)  # Tensor [0, 1, 2, 3, 4], note, can also do: torch.arange(11)
x = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
x = torch.empty(size=(1, 5)).normal_(
    mean=0, std=1
)  # Normally distributed with mean=0, std=1
x = torch.empty(size=(1, 5)).uniform_(
    0, 1
)  # Values from a uniform distribution low=0, high=1
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3

# How to make initialized tensors to other types (int, float, double)
# These will work even if you're on CPU or CUDA!
tensor = torch.arange(4)  # [0, 1, 2, 3] Initialized as int64 by default
print(f"Converted Boolean: {tensor.bool()}")  # Converted to Boolean: 1 if nonzero
print(f"Converted int16 {tensor.short()}")  # Converted to int16
print(
    f"Converted int64 {tensor.long()}"
)  # Converted to int64 (This one is very important, used super often)
print(f"Converted float16 {tensor.half()}")  # Converted to float16
print(
    f"Converted float32 {tensor.float()}"
)  # Converted to float32 (This one is very important, used super often)
print(f"Converted float64 {tensor.double()}")  # Converted to float64

# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_again = (
    tensor.numpy()
)  # np_array_again will be same as np_array (perhaps with numerical round offs)

# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# -- Addition --
z1 = torch.empty(3)
torch.add(x, y, out=z1)  # This is one way
z2 = torch.add(x, y)  # This is another way
z = x + y  # This is my preferred way, simple and clean.

# -- Subtraction --
z = x - y  # We can do similarly as the preferred way of addition

# -- Division (A bit clunky) --
z = torch.true_divide(x, y)  # Will do element wise division if of equal shape

# -- Inplace Operations --
t = torch.zeros(3)

t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
t += x  # Also inplace: t = t + x is not inplace, bit confusing.

# -- Exponentiation (Element wise if vector or matrices) --
z = x.pow(2)  # z = [1, 4, 9]
z = x ** 2  # z = [1, 4, 9]


# -- Simple Comparison --
z = x > 0  # Returns [True, True, True]
z = x < 0  # Returns [False, False, False]

# -- Matrix Multiplication --
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # Matrix multiplication of x1 and x2, out shape: 2x3
x3 = x1.mm(x2)  # Similar as line above

# -- Matrix Exponentiation --
matrix_exp = torch.rand(5, 5)
print(
    matrix_exp.matrix_power(3)
)  # is same as matrix_exp (mm) matrix_exp (mm) matrix_exp

# -- Element wise Multiplication --
z = x * y  # z = [9, 16, 21] = [1*9, 2*8, 3*7]

# -- Dot product --
z = torch.dot(x, y)  # Dot product, in this case z = 1*9 + 2*8 + 3*7

# -- Batch Matrix Multiplication --
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # Will be shape: (b x n x p)

# -- Example of broadcasting --
x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))
z = (
    x1 - x2
)  # Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
z = (
    x1 ** x2
)  # Shape of z is 5x5: How? Broadcasting! Element wise exponentiation for every row

# Other useful tensor operations
sum_x = torch.sum(
    x, dim=0
)  # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
values, indices = torch.max(x, dim=0)  # Can also do x.max(dim=0)
values, indices = torch.min(x, dim=0)  # Can also do x.min(dim=0)
abs_x = torch.abs(x)  # Returns x where absolute function has been applied to every element
z = torch.argmax(x, dim=0)  # Gets index of the maximum value
z = torch.argmin(x, dim=0)  # Gets index of the minimum value
mean_x = torch.mean(x.float(), dim=0)  # mean requires x to be float
z = torch.eq(x, y)  # Element wise comparison, in this case z = [False, False, False]
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0) # RELU outputs
# All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
# If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
z = torch.any(x)  # will return True if any element in the tensor evaluates to a boolean value of true, can also do x.any() instead of torch.any(x)
z = torch.all(
    x
)  # will return False (since not all are True), can also do x.all() instead of torch.all()


### Tensor indexing ####

batch_size = 10
features = 25
x = torch.rand(batch_size,features)
print(x[0])

x = torch.arange(10)

## if element in x is > 5 don't change x else multiple by 2
print(torch.where(x>5,x,x*2))
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]
print(x)

x = torch.tensor([0,1,2,4,52,52,1,4,2,59,32,2])
print(x.ndimension())

x = torch.arange(10)
print(x.numel()) # Number of elements
