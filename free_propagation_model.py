import tensorflow as tf
from typing import Callable, List, Tuple
import numpy as np


def pprint_sparse_tensor(st: tf.SparseTensor) -> None:
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    s += "}>"
    print(s)
    

def generate_selection_tensor(
    shape: tf.TensorShape,
    range_start: int,
    range_stop: int,
    dtype: str = "float32"
) -> tf.SparseTensor:
    """
    Function to produce (\mathbf{S}^{(\alpha)})^{T}
    """
    if range_start == range_stop:
        return tf.sparse.transpose(tf.sparse.from_dense(tf.zeros(shape, dtype=dtype)))
    
    ones_indices = [[i, i] for i in range(range_start, range_stop)]
    
    return tf.sparse.transpose(tf.SparseTensor(
        indices=ones_indices,
        values=tf.ones(len(ones_indices), dtype=dtype),
        dense_shape=shape,
    ))


class FreePropModel:
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        weights: tf.Variable,       
        iterations: int,
        input_activation_fn: Callable[[tf.Tensor], tf.Tensor],
        output_activation_fn: Callable[[tf.Tensor], tf.Tensor],
        hidden_activation_fn: Callable[[tf.Tensor], tf.Tensor],
        dtype: str = "float32",
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        # (\mathbf{W})^{T}
        self.weights = weights
        # l
        self.iterations = iterations
        # a_{d}
        self.input_activation_fn = input_activation_fn
        # a_{r}
        self.output_activation_fn = output_activation_fn
        #a_{h}
        self.hidden_activation_fn = hidden_activation_fn
        self.dtype = dtype
        
        # \mathbf{S}^{(d)}
        self._input_selection_tensor = generate_selection_tensor(
            self.weights.shape,
            0,
            self.input_shape,
            dtype=self.dtype,
        )
        # \mathbf{S}^{(r)}
        self._output_selection_tensor = generate_selection_tensor(
            self.weights.shape,
            self.input_shape,
            self.input_shape + self.output_shape,
            dtype=self.dtype,
        )
        
        # \mathbf{S}^{(h)}
        self._hidden_selection_tensor = generate_selection_tensor(
            self.weights.shape,
            self.input_shape + self.output_shape,
            self.weights.shape[0],
            dtype=self.dtype,
        )
        
    def fit(self, x: tf.Tensor) -> tf.Tensor:
        # \mathbf{N}
        node_tensor = tf.concat([x, tf.zeros((x.shape[0], self.weights.shape[0] - x.shape[1]))], axis=1)
        node_tensor = self.input_activation_fn(node_tensor)
        node_tensor = tf.transpose(node_tensor)
        
        for _ in tf.range(self.iterations):
            # (\mathbf{T})^{T}
            T = tf.tensordot(self.weights, node_tensor, axes=[[1], [0]])
            # (\mathbf{T}^{(d)})^{T}
            T_d = tf.sparse.sparse_dense_matmul(self._input_selection_tensor, self.input_activation_fn(T))
            # (\mathbf{T}^{(r)})^{T}
            T_r = tf.sparse.sparse_dense_matmul(self._output_selection_tensor, self.output_activation_fn(T))
            # (\mathbf{T}^{(h)})^{T}
            T_h = tf.sparse.sparse_dense_matmul(self._hidden_selection_tensor, self.hidden_activation_fn(T))
            
            node_tensor = T_d + T_r + T_h
        
        node_tensor = tf.transpose(node_tensor)
        
        # return \mathbf{\hat{Y}}
        return node_tensor[:, self.input_shape:self.input_shape + self.output_shape]