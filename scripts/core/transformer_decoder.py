from keras.layers import Layer, Dropout
from scripts.core.multiHeadAttention import MultiHeadAttention
from scripts.core.transformer_pos_enc_layer import (
    PositionEmbeddingFixedWeights)
from scripts.core.transformer_encoder import (
    AddNormalization, FeedForward)
from numpy import random
from keras.layers import Input
from keras import Model


"""
Implementing the decoder layer
"""


class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model,
                 d_ff, rate, sequence_length, **kwargs):

        super().__init__(**kwargs)

        self.build(input_shape=[None, sequence_length, d_model])

        self.d_model = d_model

        self.sequence_length = sequence_length

        self.multi_head_attention_1 = MultiHeadAttention(
            h, d_k, d_v, d_model
        )
        self.dropout_1 = Dropout(rate)

        self.add_norm_1 = AddNormalization()

        self.multi_head_attention_2 = MultiHeadAttention(
            h, d_k, d_v, d_model
        )

        self.dropout_2 = Dropout(rate)

        self.add_norm_2 = AddNormalization()

        self.feed_forward = FeedForward(d_ff, d_model)

        self.dropout_3 = Dropout(rate)

        self.add_norm_3 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(
            self.sequence_length, self.d_model
        ))
        return Model(
            inputs=[input_layer],
            outputs=self.call(
                input_layer,
                input_layer,
                None,
                None,
                True)
        )

    def call(self, x, encoder_output, lookahead_mask,
             padding_mask, training):
        # Multi-head attention layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        multi_head_output_1 = self.multi_head_attention_1(
            x, x, x, lookahead_mask
        )

        # Add a dropout layer
        multi_head_output_1 = self.dropout_1(
            multi_head_output_1, training=training
        )

        # Followed by an Add and Norm layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        add_norm_output_1 = self.add_norm_1(
            x, multi_head_output_1)

        # Followed by another attention layer
        multi_head_output_2 = self.multi_head_attention_2(
            add_norm_output_1, encoder_output,
            encoder_output, padding_mask
        )

        # Add another dropout layer
        multi_head_output_2 = self.dropout_2(
            multi_head_output_2, training=training
        )

        # Followed by another Add and Norm layer
        add_norm_output_2 = self.add_norm_1(
            add_norm_output_1, multi_head_output_2)

        # Followed by a fully connected layer
        # Expected output shape:
        # (batch_size, sequence_length, d_model)
        feed_forward_output = self.feed_forward(add_norm_output_2)

        # Add in another dropout layer
        feed_forward_output = self.dropout_3(
            feed_forward_output, training=training
        )

        # Followed by another Add and Norm layer
        return self.add_norm_3(
            add_norm_output_2, feed_forward_output)


"""
Implementing the Decoder
"""


class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length,
                 h, d_k, d_v, d_model, d_ff, n, rate,
                 **kwargs):

        super().__init__(**kwargs)

        self.pos_encoding = PositionEmbeddingFixedWeights(
            sequence_length, vocab_size, d_model
        )

        self.dropout = Dropout(rate)

        self.decoder_layer = [
            DecoderLayer(
                h, d_k, d_v, d_model, d_ff, rate, sequence_length
            ) for _ in range(n)
        ]

    def call(self, output_target, encoder_output,
             lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        # Expected output shape:
        # (number_of_sentences, sequence_length, d_model)
        pos_encoding_output = self.pos_encoding(output_target)

        # Add ddropout layer
        x = self.dropout(
            pos_encoding_output, training=training)

        # Pass on the positional encoded values to
        # each encoded layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask,
                      padding_mask, training)

        return x


"""
Testing the code
"""
# Vocabulary size for the decoder
dec_vocab_size = 20
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
enc_output = random.random((batch_size, input_seq_length, d_model))

decoder = Decoder(dec_vocab_size, input_seq_length,
                  h, d_k, d_v, d_model, d_ff, n,
                  dropout_rate)

# print(decoder(input_seq, enc_output, None, True))

"""
Running this script produces an output of shape:
        batch_size, 
        sequence_length and 
        model dimensionality
"""
