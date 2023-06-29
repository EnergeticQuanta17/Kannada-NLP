from tensorflow.keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

def printModelSummary(model):
    """Prints a summary of the model."""
    model.summary()

    # Print the layer details.
    for layer in model.layers:
        print(layer.name, layer.get_config())

    # Print the model inputs and outputs.
    print(model.inputs)
    print(model.outputs)

def trainModelUsingBiLSTM(maxWordLen=100, maxSentLen=10, totalChars=50, totalPOS=80, totalChunks=100, epochs=1):
    """Train a model using BILSTM C2W."""
    embeddingLayer = Embedding(
        totalChars + 2, 50, input_length=maxWordLen, trainable=True)
    charInput = Input(shape=(maxWordLen,))
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM(50, return_sequences=False, dropout=0.3), merge_mode='sum')(charEmbedding)
    charModel = Model(charInput, charOutput)
    charSeq = Input(shape=(maxSentLen, maxWordLen))
    charTD = TimeDistributed(charModel)(charSeq)
    wordSeq = Input(shape=(maxSentLen, 300))
    merge = concatenate([wordSeq, charTD], axis=-1)
    wordSeqLSTM = Bidirectional(LSTM(350, input_shape=(
        maxSentLen, 350), return_sequences=True, dropout=0.3), merge_mode='sum')(merge)
    wordSeqTDForPOS = TimeDistributed(Dense(totalPOS), name='pos')(wordSeqLSTM)
    activationForPOS = Activation('softmax', name='activationForPOS')(wordSeqTDForPOS)
    wordSeqTDForChunk = TimeDistributed(Dense(totalChunks), name='chunk')(wordSeqLSTM)
    activationForChunk = Activation('softmax', name='activationForChunk')(wordSeqTDForChunk)
    dictLosses = {'activationForPOS': 'sparse_categorical_crossentropy', 'activationForChunk': 'sparse_categorical_crossentropy'}
    finalModel = Model(
            inputs=[charSeq, wordSeq], outputs=[activationForPOS, activationForChunk])
    
    printModelSummary(finalModel)
    
trainModelUsingBiLSTM()