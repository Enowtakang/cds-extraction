from tensorflow import (matmul, math, cast, float32)
from keras.layers import Layer
from keras.backend import softmax
from numpy import random


"""
Implementing the scaled-Dot Product Attention
"""


class DotProductAttention(Layer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing
        # the layer, and then scaling
        scores = matmul(queries, keys, transpose_b=True) / (
            math.sqrt(cast(d_k, float32))
        )
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of
        # the value vectors
        return matmul(weights, values)


"""
Testing out the code
"""
# Maximum length of the input sequence
input_seq_length = 5
# Dimensionality of the linearly projected queries and keys
d_k = 64
# Dimensionality of the linearly projected values
d_v = 64
# Batch size from the training process
batch_size = 64

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()

# print(attention(queries, keys, values, d_k))
