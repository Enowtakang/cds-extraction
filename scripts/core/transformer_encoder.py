from keras.layers import (
    LayerNormalization, Layer, Dense, ReLU, Dropout)
from scripts.core.multiHeadAttention import MultiHeadAttention
from scripts.core.transformer_pos_enc_layer import (
    PositionEmbeddingFixedWeights)
from numpy import random
from keras.layers import Input
from keras import Model


"""
Implementing the add and normalization layer
"""


class AddNormalization(Layer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # The "Layer Normalization" layer
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the
        # same shape in order to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)


"""
Implementing the feedforward layer
"""


class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):

        super().__init__(**kwargs)

        # First fully connected layer
        self.fully_connected_1 = Dense(d_ff)

        # Second fully connected layer
        self.fully_connected_2 = Dense(d_model)

        # ReLU activation layer
        self.activation = ReLU()

    def call(self, x):
        # The input is passed into the two fully connected
        # layers, with the ReLU in between
        x_fc_1 = self.fully_connected_1(x)

        return self.fully_connected_2(self.activation(x_fc_1))


"""
Implementing the encoding layer
"""


class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model,
                 d_ff, rate, sequence_length, **kwargs):

        super().__init__(**kwargs)

        self.build(input_shape=[None, sequence_length, d_model])

        self.d_model = d_model

        self.sequence_length = sequence_length

        self.multi_head_attention = MultiHeadAttention(
            h, d_k, d_v, d_model
        )
        self.dropout_1 = Dropout(rate)

        self.add_norm_1 = AddNormalization()

        self.feed_forward = FeedForward(d_ff, d_model)

        self.dropout_2 = Dropout(rate)

        self.add_norm_2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(
            self.sequence_length, self.d_model
        ))
        return Model(
            inputs=[input_layer],
            outputs=self.call(
                input_layer, None, True)
        )

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        multi_head_output = self.multi_head_attention(
            x, x, x, padding_mask
        )

        # Add a dropout layer
        multi_head_output = self.dropout_1(
            multi_head_output, training=training
        )

        # Followed by an Add and Norm layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        add_norm_output = self.add_norm_1(
            x, multi_head_output)

        # Followed by a fully connected layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        feed_forward_output = self.feed_forward(add_norm_output)

        # Add another dropout layer
        feed_forward_output = self.dropout_2(
            feed_forward_output, training=training
        )

        # Followed by another Add and Norm layer
        return self.add_norm_2(
            add_norm_output, feed_forward_output)


"""
Implementing the encoder
"""


class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length,
                 h, d_k, d_v, d_model, d_ff, n, rate,
                 **kwargs):

        super().__init__(**kwargs)

        self.pos_encoding = PositionEmbeddingFixedWeights(
            sequence_length, vocab_size, d_model
        )

        self.dropout = Dropout(rate)

        self.encoder_layer = [
            EncoderLayer(h, d_k, d_v, d_model, d_ff, rate,
                         sequence_length) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        pos_encoding_output = self.pos_encoding(input_sentence)

        # Add ddropout layer
        x = self.dropout(
            pos_encoding_output, training=training)

        # Pass on the positional encoded values to
        # each encoded layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x


"""
Testing the code
"""
# Vocabulary size for the encoder
enc_vocab_size = 20
# Maximum length of the input sequence
input_seq_length = 5
# Number of self-attention heads
h = 8
# Dimensionality of the linearly projected queries and keys
d_k = 64
# Dimensionality of the linearly projected values
d_v = 64
# Dimensionality of the inner fully connected layer
d_ff = 2048
# Dimensionality of the model sub-layers' outputs
d_model = 512
# Number of layers in the encoder stack
n = 6
# Batch size from the training process
batch_size = 64
# Frequency of dropping the input units in the dropout layers
dropout_rate = 0.1

input_seq = random.random((batch_size, input_seq_length))

encoder = Encoder(enc_vocab_size, input_seq_length,
                  h, d_k, d_v, d_model, d_ff, n,
                  dropout_rate)

# print(encoder(input_seq, None, True))

"""
Running this script produces an output of shape:
        batch_size, 
        sequence_length and 
        model dimensionality
"""
