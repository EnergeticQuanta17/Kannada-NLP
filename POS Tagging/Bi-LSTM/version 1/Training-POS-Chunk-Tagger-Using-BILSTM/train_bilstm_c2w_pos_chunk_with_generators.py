"""Train BILSTM C2W model for POS tagging chunking, uses generators while creating vectors"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed, Embedding
from tensorflow.keras.utils import pad_sequences
from argparse import ArgumentParser
from pickle import load
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
import fasttext
from re import findall
from re import S


# set a random seed
np.random.seed(1337)

def printModelSummary(model):
    """Prints a summary of the model."""
    model.summary()

    # Print the layer details.
    for layer in model.layers:
        print(layer.name, layer.get_config())

    # Print the model inputs and outputs.
    print(model.inputs)
    print(model.outputs)


def readLinesFromFile(filePath):
    """Read lines from a file."""
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return fileRead.readlines()


def writeDataToFile(data, filePath):
    """Write data to a file."""
    with open(filePath, 'w') as fileWrite:
        fileWrite.write(data + '\n')


def createVectors(lines, wordEmbeddings, char2Index, pos2Index, chunk2Index):
    """Creates vectors using fasttext word embedding of 300 dimensions."""
    maxSentenceLength, maxWordLength = 150, 30
    allsentenceVectors = []
    sentenceVectors = []
    allPOSTagVectors, allChunkTagVectors, posTagsForSent, chunkTagsForSent = [], [], [], []
    charSequencesForWords, charSequencesForSentences = [], []
    while True:
        for line in lines:
            line = line.strip()
            if line:
                word, posTag, chunkTag = line.split('\t')
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = []
                sentenceVectors.append(
                    wordEmbeddings.get_word_vector(word).tolist())
                posTagsForSent.append(pos2Index[posTag])
                chunkTagsForSent.append(chunk2Index[chunkTag])
            else:
                if len(sentenceVectors) > 0:
                    allsentenceVectors.append(sentenceVectors)
                    allPOSTagVectors.append(posTagsForSent)
                    allChunkTagVectors.append(chunkTagsForSent)
                    sentenceVectors, posTagsForSent, chunkTagsForSent = [], [], []
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = []
                    charSequenceForWord = []
                    wholeCharSeqForData = []
                    finalSentenceVectors = pad_sequences(
                        allsentenceVectors, padding='post', maxlen=maxSentenceLength)
                    finalPOSTagSequences = pad_sequences(
                        allPOSTagVectors, padding='post', maxlen=maxSentenceLength)
                    finalChunkTagSequences = pad_sequences(
                        allChunkTagVectors, padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = pad_sequences(charSequencesForSentences[0], maxlen=maxWordLength, padding='post')
                    finalCharSequences = pad_sequences(
                        [wholeCharSeqForData], padding='post', maxlen=maxSentenceLength)
                    finalPOSTagSequences = finalPOSTagSequences.reshape((-1, maxSentenceLength, 1))
                    finalChunkTagSequences = finalChunkTagSequences.reshape((-1, maxSentenceLength, 1))
                    wholeCharSeqForData = []
                    charSequencesForSentences = []
                    allsentenceVectors = []
                    allPOSTagVectors = []
                    allChunkTagVectors = []
                    yield [finalCharSequences, finalSentenceVectors], [finalPOSTagSequences, finalChunkTagSequences]


def createVectorsForTest(lines, wordEmbeddings, char2Index):
    """Create vectors for test sentences."""
    maxSentenceLength, maxWordLength = 150, 30
    allsentenceVectors = []
    sentenceVectors = []
    charSequencesForWords, charSequencesForSentences = [], []
    while True:
        for line in lines:
            line = line.strip()
            if line:
                word, others = line.split('\t', 1)
                charSequenceForWord = [char2Index[char] if char in char2Index else len(
                    char2Index) + 1 for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = []
                sentenceVectors.append(
                    wordEmbeddings.get_word_vector(word).tolist())
            else:
                if len(sentenceVectors) > 0:
                    allsentenceVectors.append(sentenceVectors)
                    sentenceVectors = []
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = []
                    charSequenceForWord = []
                    wholeCharSeqForData = []
                    finalSentenceVectors = pad_sequences(
                        allsentenceVectors, padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = pad_sequences(charSequencesForSentences[0], maxlen=maxWordLength, padding='post')
                    finalCharSequences = pad_sequences(
                        [wholeCharSeqForData], padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = []
                    charSequencesForSentences = []
                    allsentenceVectors = []
                    yield [[finalCharSequences, finalSentenceVectors]]


def loadObjectFromPickleFile(pickleFilePath):
    """Load object from a pickle file."""
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    """Create a reverse index dictionary where key is label and value is index."""
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, trainGen, valGen, steps, valSteps, totalChars, totalPOS, totalChunks, weightFile, epochs=1):
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
    weightFile = weightFile + '-{epoch:d}-{loss:.2f}.wts'
    checkpointCallback = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0,
                                         save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    finalModel = Model(
            inputs=[charSeq, wordSeq], outputs=[activationForPOS, activationForChunk])
    finalModel.compile(optimizer='adam',
                       loss=dictLosses,
                       metrics=['accuracy'])
    print(finalModel.summary())
    history = finalModel.fit(trainGen,
                       steps_per_epoch=steps, epochs=epochs, callbacks=[checkpointCallback], validation_data=valGen, validation_steps=valSteps)
    return finalModel, history


def sentenceLengthsInLines(lines):
    """Find the sentence lengths in conll lines."""
    lengths = []
    sentLen = 0
    for line in lines:
        if line.strip():
            sentLen += 1
        else:
            lengths.append(sentLen)
            sentLen = 0
    return lengths


def predictTags(trainedModel, generator, lines, index2POS, index2Chunk, steps, batchSize):
    """Predict tags using the training model."""
    sentenceLengths = sentenceLengthsInLines(lines)
    predictedTags = ''
    posTags, chunkTags = trainedModel.predict(generator, steps=steps, batch_size=batchSize)
    assert posTags.shape[0] == chunkTags.shape[0]
    for indexSample in range(posTags.shape[0]):
        for index in range(sentenceLengths[indexSample]):
            predictedTags += index2POS[np.argmax(posTags[indexSample][index])] + \
                '\t' + \
                index2Chunk[np.argmax(chunkTags[indexSample][index])] + '\n'
        predictedTags += '\n'
    return predictedTags


def main():
    """Pass arguments and call the functions."""
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--train', dest='tr',
                        help='Enter the input file containing pos/chunk labeled data')
    parser.add_argument('--val', dest='val', help='Enter the validation data')
    parser.add_argument('--test', dest='test', help='Enter the validation data')
    parser.add_argument('--embed', dest='embed',
                        help='enter word embeddings file')
    parser.add_argument('--lang', dest='lang',
                        help='enter the language')
    parser.add_argument('--wt', dest='wt',
                        help='enter weight file')
    parser.add_argument('--epoch', dest='epoch',
                        help='enter the no of epochs', type=int)
    args = parser.parse_args()
    if not args.tr or not args.val or not args.embed or not args.lang or not args.wt or not args.epoch:
        print("Passed Arguments are not correct")
        exit(1)
    else:
        index2Char =loadObjectFromPickleFile(args.lang + '-index-to-char.pkl') 
        index2POS = loadObjectFromPickleFile(args.lang + '-index-to-pos.pkl')
        index2Chunk = loadObjectFromPickleFile(args.lang + '-index-to-chunk.pkl')
        char2Index = createReverseIndex(index2Char)
        pos2Index = createReverseIndex(index2POS)
        chunk2Index = createReverseIndex(index2Chunk)
        print(pos2Index)
        print(chunk2Index)
        wordEmbeddings = fasttext.load_model(args.embed)
        trainFileDesc = open(args.tr, 'r', encoding='utf-8')
        trainData = trainFileDesc.read().strip() + '\n\n'
        trainFileDesc.close()
        totalSamples = len(findall('\n\n', trainData, S))
        print('Total Samples', totalSamples)
        batchSize = 16
        if totalSamples % batchSize == 0:
            steps = totalSamples // batchSize
        else:
            steps = totalSamples // batchSize + 1
        print('--TRAIN GEN--')
        trainLines = trainData.split('\n')
        valFileDesc = open(args.val, 'r', encoding='utf-8')
        valData = valFileDesc.read().strip() + '\n\n'
        valFileDesc.close()
        totalValSamples = len(findall('\n\n', valData, S))
        print('Total Val Samples', totalValSamples)
        valLines = valData.split('\n')
        if totalValSamples % batchSize == 0:
            valSteps = totalValSamples // batchSize
        else:
            valSteps = totalValSamples // batchSize + 1
        print('--VAL GEN--')
        testFileDesc = open(args.test, 'r', encoding='utf-8')
        testData = testFileDesc.read().strip() + '\n\n'
        testFileDesc.close()
        testLines = testData.split('\n')
        totalTestSamples = len(findall('\n\n', testData, S))
        if totalTestSamples % batchSize == 0:
            testSteps = totalTestSamples // batchSize
        else:
            testSteps = totalTestSamples // batchSize + 1
        maxWordLength, maxSentenceLength = 30, 150
        trainGen = createVectors(
            trainLines, wordEmbeddings, char2Index, pos2Index, chunk2Index)
        valGen = createVectors(
            valLines, wordEmbeddings, char2Index, pos2Index, chunk2Index)
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        biLSTM, history = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, trainGen, valGen, steps, valSteps,
                                       len(char2Index), len(pos2Index), len(chunk2Index), args.wt, args.epoch)
        valGen = createVectorsForTest(valLines, wordEmbeddings, char2Index)
        predictedDevTags = predictTags(biLSTM, valGen, valLines, index2POS, index2Chunk, totalValSamples, 1)
        writeDataToFile(predictedDevTags, args.lang + '-dev-predicted-bilstm-epoch' + str(args.epoch) + '.txt')
        testGen = createVectorsForTest(testLines, wordEmbeddings, char2Index)
        predictedTestTags = predictTags(biLSTM, testGen, testLines, index2POS, index2Chunk, totalTestSamples, 1)
        writeDataToFile(predictedTestTags, args.lang + '-test-predicted-bilstm-epoch' + str(args.epoch) + '.txt')


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    main()
