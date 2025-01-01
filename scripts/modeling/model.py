from keras import Model
from keras.layers import Dense
from scripts.core.transformer_encoder import Encoder, EncoderLayer
from scripts.core.transformer_decoder import Decoder, DecoderLayer
from tensorflow import (
    math, cast, float32, linalg, ones, maximum, newaxis)


class TransformerModel(Model):
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 encoder_sequence_length,
                 decoder_sequence_length,
                 h, d_k, d_v, d_model, d_ff_inner,
                 n, rate, **kwargs):

        super().__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(
            encoder_vocab_size, encoder_sequence_length,
            h, d_k, d_v, d_model, d_ff_inner, n, rate
        )

        # Set up the decoder
        self.decoder = Decoder(
            decoder_vocab_size, decoder_sequence_length,
            h, d_k, d_v, d_model, d_ff_inner, n, rate
        )

        # Define the final dense layer
        self.model_last_layer = Dense(decoder_vocab_size)

    def padding_mask(self, input):
        # Create a mask which marks the zero padding values
        # in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        """
        The shape of the mask should be broadcastable to the 
            shape of the attention weights that it will be
            masking later on
        """
        return mask[:, newaxis, newaxis, :]

    def look_ahead_mask(self, shape):
        # Mask out feature entries by marking them with 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):
        """
        Create padding mask to mask the encoder inputs and the
            encoder outputs in the decoder
        """
        encoder_padding_mask = self.padding_mask(encoder_input)

        """
        Create and combine padding and look-ahead masks to be 
            fed into the decoder
        """
        decoder_in_padding_mask = self.padding_mask(decoder_input)
        decoder_in_lookahead_mask = self.look_ahead_mask(
            decoder_input.shape[1]
        )
        decoder_in_lookahead_mask = maximum(
            decoder_in_padding_mask, decoder_in_lookahead_mask
        )

        """
        Feed the input into the encoder
        """
        encoder_output = self.encoder(
            encoder_input, encoder_padding_mask, training
        )

        """
        Feed the encoder output into the decoder
        """
        decoder_output = self.decoder(
            decoder_input, encoder_output,
            decoder_in_lookahead_mask,
            encoder_padding_mask,
            training
        )

        """
        Pass the decoder output through a final dense layer
        """
        model_output = self.model_last_layer(decoder_output)

        return model_output


"""
Testing the code
"""
# Vocabulary size for the encoder
enc_vocab_size = 20
# Vocabulary size for the decoder
dec_vocab_size = 20
# Maximum length of the input sequence
enc_seq_length = 5
# Maximum length of the output sequence
dec_seq_length = 5
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
# Frequency of dropping the input units in the dropout layers
dropout_rate = 0.1


"""Create Model"""

training_model = TransformerModel(
    enc_vocab_size, dec_vocab_size, enc_seq_length,
    dec_seq_length, h, d_k, d_v, d_model, d_ff, n,
    dropout_rate
)


"""
Print Model Summaries for the EncoderLayer and the 
    DecoderLayer
"""
# Summary for EncoderLayer
encoder = EncoderLayer(
    h, d_k, d_v, d_model, d_ff, dropout_rate, enc_seq_length
)
encoder.build_graph().summary()

# Summary for DecoderLayer
decoder = DecoderLayer(
    h, d_k, d_v, d_model, d_ff, dropout_rate, enc_seq_length
)

# decoder.build_graph().summary()
