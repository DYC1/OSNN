# %%
import jax.numpy as np
from jax.typing import ArrayLike as Tensor

from jax import Array
from jax import grad, jit, vmap, value_and_grad, hessian
from jax import random
import os
import subprocess
from tqdm.auto import tqdm
import datetime
import matplotlib.pyplot as plt
from functools import partial
import time
import subprocess
import os
from typing import List, Tuple, Union, Callable, Sequence,Any
import flax.linen as nn
from flax.typing import PRNGKey,VariableDict
import optax



#%%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_min_memory_gpu():
    """
    Get the GPU with the minimum memory usage.
    This function uses the `nvidia-smi` command to query the memory usage of all available GPUs.
    It returns the index of the GPU with the minimum memory usage.
    Returns:
        str: The index of the GPU with the minimum memory usage as a string.
        None: If `nvidia-smi` command fails or no GPUs are available.
    Raises:
        RuntimeError: If `nvidia-smi` command fails or no GPUs are available (commented out).
    """

    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        gpu_memory_list = output.decode('utf-8').strip().split('\n')
        gpu_memory_list = [int(memory) for memory in gpu_memory_list]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
        # raise RuntimeError("Failed to get GPU memory information using nvidia-smi.")

    if len(gpu_memory_list) == 0:
        # raise RuntimeError("No GPUs available.")
        return None

    min_memory = min(gpu_memory_list)
    min_memory_gpu = gpu_memory_list.index(min_memory)

    return (f'{min_memory_gpu}')


os.environ["CUDA_VISIBLE_DEVICES"] = get_min_memory_gpu()
Function = Callable[[Tensor], Array]
NN=Callable[[Tensor,VariableDict],Array]
# global Device,Dtype
key = random.PRNGKey(42)
uniform = lambda shape: random.uniform(key, shape)
pi = np.pi
try:
    print(Timetxt)
except:
    Timetxt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# %%
def CreateGrad(fun: Function, dim: int) -> Function:
    """
    Creates a gradient function for the given function `fun` with respect to the specified dimension `dim`.
    Args:
        fun (Function): The function for which the gradient is to be computed.
        dim (int): The dimension with respect to which the gradient is to be computed.
    Returns:
        Function: A function that computes the gradient of `fun` with respect to the specified dimension.
    """

    return jit(lambda _xx: ((vmap(grad(lambda _x: fun(_x.reshape(-1, dim)).reshape())))(_xx)))


def CreateLaplace(fun: Function, dim: int) -> Function:
    """
    Create a Laplace operator for a given function.
    This function takes a function `fun` and an integer `dim` representing the dimension,
    and returns a new function that computes the Laplace operator of `fun`.
    Parameters:
    fun (Function): The input function for which the Laplace operator is to be computed.
    dim (int): The dimension of the input space.
    Returns:
    Function: A new function that computes the Laplace operator of `fun`.
    """

    return jit(lambda _xx: ((vmap(lambda _x: np.trace(hessian(lambda _x: fun(_x.reshape(-1, dim)).reshape())(_x))))(_xx)))
@jit
def L2Norm(x:Tensor)->Array:
    def L2Norm(x: Tensor) -> Array:
        """
        Compute the square of L2 norm (Euclidean norm) of a tensor.
        Parameters:
        x (Tensor): Input tensor.
        Returns:
        Array: The square of L2 norm of the input tensor.
        """

    return (np.mean((np.square(x))))



# %%


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) class.
    Attributes:
    -----------
    layer_sizes : Sequence[int]
        A sequence of integers representing the sizes of each layer in the MLP.
    Methods:
    --------
    setup(Activation=nn.tanh):
        Initializes the layers of the MLP and sets the activation function.
    __call__(x):
        Forward pass through the MLP. Applies each layer and activation function
        to the input `x` and returns the output of the final layer.
    """

    layer_sizes: Sequence[int] = None  

    def setup(self, Activation=nn.tanh):
        
        self.layers = [nn.Dense(features=size)
                       for size in self.layer_sizes[1:]]
        self.act = Activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        return self.layers[-1](x)


class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) class that defines a simple neural network with dense layers and a specified activation function.
    Attributes:
        layer_sizes (Sequence[int]): A sequence of integers representing the sizes of each layer in the network.
    Methods:
        setup(Activation=nn.tanh):
            Initializes the layers of the network with the specified activation function.
        __call__(x):
            Performs a forward pass through the network with the input `x`.
    Args:
        Activation (callable, optional): The activation function to use in the network. Defaults to `nn.tanh`.
    Returns:
        The output of the network after performing a forward pass.
    """

    layer_sizes: Sequence[int] = None  

    def setup(self, Activation=nn.tanh):
        self.layers = [nn.Dense(features=size)
                       for size in self.layer_sizes[1:]]
        self.act = Activation

    def __call__(self, x):
        x = self.act(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.act(layer(x)) + x
        return self.layers[-1](x)

# %%


def CreateNN(NN, InputDim: int, OutputDim: int, Depth: int, width, Activation=nn.tanh) -> Tuple[nn.Module, VariableDict]:
    """
    Create a neural network with the specified architecture.
    Args:
        NN: The neural network class to instantiate.
        InputDim (int): The dimension of the input layer.
        OutputDim (int): The dimension of the output layer.
        Depth (int): The number of hidden layers.
        width: The width (number of neurons) of each hidden layer.
        Activation: The activation function to use in the network (default is nn.tanh).
    Returns:
        Tuple[nn.Module, VariableDict]: A tuple containing the instantiated neural network and its parameters.
    """
    
    _nn = NN(layer_sizes=[InputDim]+[width]*Depth+[OutputDim])
    _x = np.zeros((1, InputDim))
    params = _nn.init(key, _x)
    return _nn, params

# %%
def CreateGradNN(fun: NN, dim: int) -> NN:
    """
    Creates a gradient neural network (NN) function.
    This function takes a neural network function `fun` and a dimension `dim`, 
    and returns a new function that computes the gradient of `fun` with respect 
    to its inputs. The returned function is JIT-compiled for performance.
    Args:
        fun (NN): The neural network function to compute the gradient of.
        dim (int): The dimension of the input to the neural network function.
    Returns:
        NN: A new neural network function that computes the gradient of `fun`.
    """

    return jit(lambda _xx,para: ((vmap(grad(lambda _x: fun(_x.reshape(-1, dim),para).reshape())))(_xx)))


def CreateLaplaceNN(fun: NN, dim: int) -> NN:
    """
    Create a Laplace from a given function.
    This function takes a neural network function `fun` and a dimension `dim`, 
    and returns a new neural network function that computes the Laplacian of 
    the original function. The Laplacian is computed using the Hessian matrix 
    of the function.
    Args:
        fun (NN): The original neural network function.
        dim (int): The dimension of the input to the neural network function.
    Returns:
        NN: A new neural network function that computes the Laplacian of the 
        original function.
    """
    
    return jit(lambda _xx,para: ((vmap(lambda _x: np.trace(hessian(lambda _x: fun(_x.reshape(-1, dim),para).reshape())(_x))))(_xx)))

import orbax.checkpoint as ocp
checkpath = ocp.test_utils.erase_and_create_empty(f'./data/{Timetxt}')
checkpointer = ocp.StandardCheckpointer()