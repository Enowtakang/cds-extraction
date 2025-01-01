from tensorflow import (
    matmul, math, cast, float32, reshape, shape, transpose)
from keras.layers import Dense, Layer
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
Implementing the Multi-Head Attention
"""


class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):

        super().__init__(**kwargs)

        # Scaled dot product attention
        self.attention = DotProductAttention()

        # Number of attention heads to use
        self.heads = h

        # Dimensionality of the linearly projected queries and keys
        self.d_k = d_k

        # Dimensionality of the linearly projected values
        self.d_v = d_v

        # Dimensionality of the model
        self.d_model = d_model

        # Learned projection matrix for the queries
        self.W_q = Dense(d_k)

        # Learned projection matrix for the keys
        self.W_k = Dense(d_k)

        # Learned projection matrix for the values
        self.W_v = Dense(d_v)

        # Learned projection matrix for the multi-head output
        self.W_o = Dense(d_model)

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = reshape(
                x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(
                x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute
        # all heads in parallel
        # Resulting tensor shape:
        # (batch_size, heads, input_seq_length, -1)
        q_reshaped = self.reshape_tensor(
            self.W_q(queries), self.heads, True
        )
        # Rearrange the keys to be able to compute
        # all heads in parallel
        # Resulting tensor shape:
        # (batch_size, heads, input_seq_length, -1)
        k_reshaped = self.reshape_tensor(
            self.W_k(keys), self.heads, True
        )
        # Rearrange the values to be able to compute
        # all heads in parallel
        # Resulting tensor shape:
        # (batch_size, heads, input_seq_length, -1)
        v_reshaped = self.reshape_tensor(
            self.W_v(values), self.heads, True
        )
        # Compute the multi-head attention output using
        # the reshapes queries, keys and values
        # Resulting tensor shape:
        # (batch_size, heads, input_seq_length, -1)
        o_reshaped = self.attention(
            q_reshaped, k_reshaped, v_reshaped,
            self.d_k, mask
        )
        # Rearrange back the output into concatenated form
        # Resulting tensor shape:
        # (batch_size, input_seq_length, d_v)
        output = self.reshape_tensor(
            o_reshaped, self.heads, False
        )
        # To generate the multi-head attention, apply one
        # final linear projection to the output.
        # Resulting tensor shape:
        # (batch_size, input_seq_length, d_model)
        return self.W_o(output)


"""
Testing the code
"""

# Maximum length of the input sequence
input_seq_length = 5
# Number of self-attention heads
h = 8
# Dimensionality of the linearly projected queries and keys
d_k = 64
# Dimensionality of the linearly projected values
d_v = 64
# Dimensionality of the model sub-layers' outputs
d_model = 512
# Batch size from the training process
batch_size = 64

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multi_head_attention = MultiHeadAttention(h, d_k, d_v, d_model)

# print(multi_head_attention(queries, keys, values).shape)
# print(multi_head_attention(queries, keys, values))

"""
Running this script produces an output of shape:
        batch_size, 
        sequence_length and 
        model dimensionality
"""
