import pandas as pd
from tensorflow import convert_to_tensor, int64
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


root = "C:/Users/HP/PycharmProjects/Zattention/data/"
filename = "processed 1.xlsx"
dataset_path = root + filename

number_of_records = 3000


class PrepareDataset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Number of sentences to include in the dataset
        self.n_sentences = number_of_records
        # Ratio of the training data split
        self.train_split = 0.9

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)
        return len(tokenizer.word_index) + 1

    def __call__(self, file_name, **kwargs):
        # Load clean dataset
        clean_dataset = pd.read_excel(
            file_name, header=None, engine='openpyxl')

        # Reduce dataset size
        dataset = clean_dataset.iloc[:self.n_sentences, :].copy()

        # Include start and end of string tokens
        for i in range(dataset.shape[0]):
            dataset.loc[i, 0] = "<START> " + dataset.loc[i, 0] + " <EOS>"
            dataset.loc[i, 1] = "<START> " + dataset.loc[i, 1] + " <EOS>"

        # Random shuffle the dataset
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # split the dataset
        train = dataset.iloc[:int(self.n_sentences * self.train_split)]

        # Prepare tokenizer for the encoder input
        encoder_tokenizer = self.create_tokenizer(train.iloc[:, 0])
        encoder_seq_length = self.find_seq_length(train.iloc[:, 0])
        encoder_vocab_size = self.find_vocab_size(
            encoder_tokenizer, train.iloc[:, 0])

        # Encode and pad the input sequences
        trainX = encoder_tokenizer.texts_to_sequences(train.iloc[:, 0])
        trainX = pad_sequences(
            trainX, maxlen=encoder_seq_length, padding='post')
        trainX = convert_to_tensor(trainX, dtype=int64)

        # Prepare tokenizer for the decoder input
        decoder_tokenizer = self.create_tokenizer(train.iloc[:, 1])
        decoder_seq_length = self.find_seq_length(train.iloc[:, 1])
        decoder_vocab_size = self.find_vocab_size(
            decoder_tokenizer, train.iloc[:, 1])

        # Encode and pad the input sequences
        trainY = decoder_tokenizer.texts_to_sequences(train.iloc[:, 1])
        trainY = pad_sequences(
            trainY, maxlen=decoder_seq_length, padding='post')
        trainY = convert_to_tensor(trainY, dtype=int64)

        return (
            trainX, trainY, train, encoder_seq_length,
            decoder_seq_length, encoder_vocab_size, decoder_vocab_size)


"""
Testing the code
"""
# Prepare the training data
dataset = PrepareDataset()

(trainX, trainY, train_orig,
 encoder_seq_length, decoder_seq_length,
 encoder_vocab_size, decoder_vocab_size) = dataset(dataset_path)

# print(train_orig.iloc[0, 0], '\n', trainX[0, :])
