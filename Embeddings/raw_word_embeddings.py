import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

BATCH_SIZE = 32
SEED = 123

VALIDATION_SPLIT = 0.1

train_ds = tf.keras.util.text_dataset_from_directory(
    'rawdata/train',
    batch_size = BATCH_SIZE, 
    validation_split = VALIDATION_SPLIT,
    subset = 'training',
    seed=SEED
)

val_ds = tf.keras.util.text_dataset_from_directory(
    'rawdata/train',
    batch_size = BATCH_SIZE, 
    validation_split = VALIDATION_SPLIT,
    subset = 'validation',
    seed=SEED
)

