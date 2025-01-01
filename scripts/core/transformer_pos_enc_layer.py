import tensorflow as tf
from tensorflow import (
    convert_to_tensor, string)
from keras.layers import (
    TextVectorization, Embedding, Layer)
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.data import Dataset


class PositionEmbeddingLayer(Layer):
    def __init__(self, seq_length, vocab_size,
                 output_dim, **kwargs):

        super().__init__(**kwargs)

        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim
        )

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(
            position_indices
        )
        return embedded_words + embedded_indices


number = 10000


class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_length, vocab_size,
                 output_dim, **kwargs):

        super().__init__(**kwargs)

        word_embedding_matrix = self.get_position_encoding(
            vocab_size, output_dim
        )
        pos_embedding_matrix = self.get_position_encoding(
            seq_length, output_dim
        )
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim,
            weights=[pos_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=number):
        p = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                p[k, 2 * i] = np.sin(k / denominator)
                p[k, 2 * i + 1] = np.cos(k / denominator)
        return p

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(
            position_indices
        )
        return embedded_words + embedded_indices


"""
Attempt to run the layer
"""
output_sequence_length = 5
vocab_size = 10
sentences = [["A A T C"], ["T G C"]]
sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(
    output_sequence_length=output_sequence_length,
    max_tokens=vocab_size
)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to vectors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

# print("Vocabulary: ", vectorize_layer.get_vocabulary())
# print("vectorized words: ", vectorized_words)

output_length = 6
word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)

# print(embedded_words)


embedding = PositionEmbeddingFixedWeights(
    seq_length=output_sequence_length,
    vocab_size=vocab_size,
    output_dim=output_length
)

output = embedding(vectorized_words)

# print("Output from embedded layer: ", output)


"""
Visualizing the final embedding
"""
# Row 1906 Col A
seq_1 = ("C C G A A T G A T A T G A T T T C T C G T A T G A T T G G G T T C A T A A A T C G T A A A G C T G A G G A T " 
         "A G T G G T A T T A G A T C T G T G G A G T C G T T T A G G C A G A T T T C C G A T G T C G T G C T T A T A " 
         "A T T G T A C C A C A G A T A G C C T T G T C T G C A G A G T T G T C A T T A A A G C T T G T C G A T T C A " 
         "G C C A A T A T T T T G G A G G C T G T A A A T G A C C A G G A G G T C A C A A T C A A T A G T G T T G G T " 
         "G G T C C A T G C G T C G T T G T G A T G A A T T G T G C T C A C T C G A T T C C G A A T G A G G A C A G G " 
         "A C T C A T G T A A A C G G A T C C")
# Row 14 Col A
seq_2 = ("G G T C C T T T A G T T T C A C T T G C T A A A C A T A A T G G T A A T G T T G A A G T C T C T A A G C C A T G G T C T T C T T C T G A C G A A A A G C T T G C T T T G A C T A A G G C T A T G G A T A C A T C C A A A G G A A A G A T A C T G T T G A A C A C A G A G G G A A C A T C T T C C C T T G G A A C C T A T G A A T C T G A T T C T A T C A C A G A A T C A G A A G G T T A C G A T C T T T C T G C A A G A A T G A T A G T A G A T A C A A A C C A T C A T A T C T C A A A C T G G A A A A A T G A T C T T T T T G T T G G C A A C G G G A A G C A A A A T G C A A A C A A G G T C A T C A A G A T C T G T C C A A C T T G G G A C A G C A G A A A A C A A T A C A T G A T G A T T T C C A G G A T T G T G A T A T G G G T C T G C C C C A C T A T A C C A A A C C C T A C A G G A A A A C T T G T G G T T G C C C T G G T T G A T C C C A A C A T G C C A T C T G A A A A G C A A G T C A T T C T G A A G G G T C A G G G G A C A A T A A T T G A T C C T A T A T G T T T T G T C T T T T A T C T G A A C T G G T C T A T T C C G A A A A T G A A T A A C A C T C C A G A A A A C T G C T G T C A G C T G C A T T T G A T G T G C A G C C A A G A A T A C A A G A A G G G G G T T T C T T T T G G T A G T A T C A T G T A C T C T T G G A C A A A G G A G T T T T G T G A T T C A C C C A G A G C T G A T A A A G A T A A A A G T T G C A T G G T C A T A C C T C T A A A C A G A G C T A T T A G A G C T A G A T C T C A A G C A T T C A T T G A G G C T T G C A A G C T A A T A A T C C C T A A A G G G A A C A G T G A A A A G C A G A T T A A A A A A C A G C T T A A A G A A T T G A G C T T A C A T C T T G A G A G A T C A G T T G A A G A A G A A G A G G A A G G G A T T T")

# print(len(seq_1), len(seq_2))

total_vocabulary = 200
seq_length = 20
final_output_length = 50

phrase_vectorization_layer = TextVectorization(
    output_sequence_length=seq_length,
    max_tokens=total_vocabulary
)
# Learn the dictionary
phrase_vectorization_layer.adapt([seq_1, seq_2])
# Convert all sentences to tensors
phrase_tensors = convert_to_tensor([seq_1, seq_2], dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_phrases = phrase_vectorization_layer(phrase_tensors)

random_weights_embedding_layer = PositionEmbeddingLayer(
    seq_length, total_vocabulary, final_output_length
)

fixed_weights_embedding_layer = PositionEmbeddingFixedWeights(
    seq_length, total_vocabulary, final_output_length
)

random_embedding = random_weights_embedding_layer(vectorized_phrases)
fixed_embedding = fixed_weights_embedding_layer(vectorized_phrases)

"""
Plot random Embedding
"""


def plot_random_embedding():
    fig = plt.figure(figsize=(15, 5))
    title = ["Row1906ColA", "Row14ColA"]
    for i in range(2):
        ax = plt.subplot(1, 2, 1 + i)
        matrix = tf.reshape(
            random_embedding[i, :, :],
            (seq_length, final_output_length)
        )
        cax = ax.matshow(matrix)
        plt.gcf().colorbar(cax)
        plt.title(title[i], y=1.2)
    fig.suptitle("Random Weight Embedding")
    plt.savefig("_Random Embedding.png")


"""
Plot fixed Embedding
"""


def plot_fixed_embedding():
    fig = plt.figure(figsize=(15, 5))
    title = ["Row1906ColA", "Row14ColA"]
    for i in range(2):
        ax = plt.subplot(1, 2, 1 + i)
        matrix = tf.reshape(
            fixed_embedding[i, :, :],
            (seq_length, final_output_length)
        )
        cax = ax.matshow(matrix)
        plt.gcf().colorbar(cax)
        plt.title(title[i], y=1.2)
    fig.suptitle("Fixed Weight Embedding")
    plt.savefig("_Fixed Embedding.png")


# plot_random_embedding()
# plot_fixed_embedding()
