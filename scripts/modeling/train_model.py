from time import time
from model import TransformerModel
from tensorflow.keras.metrics import Mean
from prepare_dataset import PrepareDataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import (
    data, train, math, reduce_sum, cast, equal, argmax, float32, int64,
    GradientTape, TensorSpec, function, )


"""
Define the model parameters
"""
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


"""
Define training parameters
"""
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

step_number = 13 # line 240

"""
Implementing a learning rate scheduler
"""
steps = 1200


class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=steps, **kwargs):

        super().__init__(**kwargs)

        self.d_model = cast(d_model, float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        """
        Linearly increasing the learning rate for
        the first warmup steps, and decreasing it thereafter
        """
        step_num = cast(step_num, float32)  # Cast step_num to float32
        arg_1 = step_num ** -0.5
        arg_2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(
            arg_1, arg_2)


"""
Instantiate an Adam optimizer
"""
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

"""
Prepare the training and test splits of the dataset
"""
# Specify file path
root = "C:/Users/HP/PycharmProjects/Zattention/data/"
filename = "processed 1.xlsx"
dataset_path = root + filename
# Instantiate the PrepareDataset class
dataset = PrepareDataset()
# Prepare the dataset
(trainX, trainY, train_orig,
 encoder_seq_length, decoder_seq_length,
 encoder_vocab_size, decoder_vocab_size) = dataset(dataset_path)

"""
Prepare the dataset batches
"""
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

"""
Create model
"""
training_model = TransformerModel(
    encoder_vocab_size, decoder_vocab_size, encoder_seq_length,
    decoder_seq_length, h, d_k, d_v, d_model, d_ff, n,
    dropout_rate)

"""
Defining the loss function
"""


def loss_function(target, prediction):
    """
    Create a mask so that the zero padding values are not
    included in the computation of loss
    """
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)

    """
    Compute a sparse categorical cross-entropy loss on the
    unmasked values
    """
    loss = sparse_categorical_crossentropy(
        target, prediction, from_logits=True) * mask

    """
    Compute the mean loss over the unmasked values
    """
    return reduce_sum(loss) / reduce_sum(mask)


"""
Defining the accuracy function
"""


def accuracy_function(target, prediction):
    """
    Create a mask so that the zero padding values are not
    included in the computation of loss
    """
    mask = math.logical_not(equal(target, 0))

    """
    Find equal prediction and target values, and apply 
    the padding mask 
    """
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(mask, accuracy)

    """
    Cast the True/False values to 32-bit-precision 
    floating point numbers 
    """
    mask = cast(mask, float32)
    accuracy = cast(accuracy, float32)

    """
    Compute the mean accuracy over unmasked values
    """
    return reduce_sum(accuracy) / reduce_sum(mask)


"""
Include metrics monitoring
"""
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

"""
Create a checkpoint object and manager to manage multiple
checkpoints 
"""
check_point = train.Checkpoint(
    model=training_model, optimizer=optimizer)
check_point_manager = train.CheckpointManager(
    check_point, "./checkpoints", max_to_keep=3)

"""
Speeding up the training process
"""


@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        """
        Run the forward pass of the model to generate a prediction
        """
        prediction = training_model(
            encoder_input, decoder_input)

        """
        Compute the training loss
        """
        loss = loss_function(decoder_output, prediction)

        """
        Compute the training accuracy
        """
        accuracy = accuracy_function(decoder_output, prediction)

    """
    Retrieve gradients of the trainable variables with respect
    to the training loss
    """
    gradients = tape.gradient(
        loss, training_model.trainable_weights)

    """
    Update the values of the trainable variables by gradient 
    descent
    """
    optimizer.apply_gradients(zip(
        gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)


"""
Continuation of algorithm
"""

for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))

    start_time = time()

    """
    Iterate over the dataset batches
    """

    for step, (train_batchX, train_batchY) in enumerate(
            train_dataset):
        """
        Define the encoder and decoder inputs, and the 
        decoder output
        """
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % step_number == 0:
            print(
                f"""Epoch {
                epoch + 1} Step {step} Loss {
                train_loss.result():.4f} """
                +
                f"""Accuracy {train_accuracy.result():.4f}""")

    """
    Print epoch number and loss value at the end of every epoch
    """

    print(f"""Epoch {epoch + 1}: Training Loss {
    train_loss.result():.4f}, """ + f"""Training Accuracy {
    train_accuracy.result():.4f}""")

    """
    Save a checkpoint after every 5 epochs
    """

    if (epoch + 1) % 5 == 0:
        save_path = check_point_manager.save()
        print(f"Saved checkpoint at epoch {epoch + 1}")

print("Total time taken: %.2fs" % (time() - start_time))
