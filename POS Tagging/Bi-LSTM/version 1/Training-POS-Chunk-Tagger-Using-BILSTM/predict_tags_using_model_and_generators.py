"""Predict labels for sentences."""
from tensorflow.keras.utils import pad_sequences
from argparse import ArgumentParser
from pickle import load
import numpy as np
import tensorflow as tf
import fasttext
from re import findall
from re import S
from keras.models import load_model


np.random.seed(1337)


def readLinesFromFile(filePath):
    """Read lines from a file."""
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return fileRead.readlines()


def writeDataToFile(data, filePath):
    """Write data to a file."""
    with open(filePath, 'w') as fileWrite:
        fileWrite.write(data + '\n')


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
    """Create a reverse index dictionary."""
    return {val: key for key, val in dictItems.items()}


def sentenceLengthsInLines(lines):
    """Find sentence lengths in conll lines."""
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
    """Predict tags using the trained model."""
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
    """Pass arguments and call functions here."""
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--test', dest='test', help='Enter the validation data')
    parser.add_argument('--embed', dest='embed',
                        help='enter word embeddings file')
    parser.add_argument('--lang', dest='lang',
                        help='enter the language')
    parser.add_argument('--model', dest='model',
                        help='enter model file')
    parser.add_argument('--output', dest='out', help='enter output file')
    args = parser.parse_args()
    index2Char =loadObjectToPickleFile(args.lang + '-index-to-char.pkl') 
    index2POS = loadObjectToPickleFile(args.lang + '-index-to-pos.pkl')
    index2Chunk = loadObjectToPickleFile(args.lang + '-index-to-chunk.pkl')
    char2Index = createReverseIndex(index2Char)
    pos2Index = createReverseIndex(index2POS)
    chunk2Index = createReverseIndex(index2Chunk)
    wordEmbeddings = fasttext.load_model(args.embed)
    loadedModel = load_model(args.model)
    testFileDesc = open(args.test, 'r', encoding='utf-8')
    testData = testFileDesc.read().strip() + '\n\n'
    testFileDesc.close()
    testLines = testData.split('\n')
    totalTestSamples = len(findall('\n\n', testData, S))
    maxWordLength, maxSentenceLength = 30, 150
    testSentLens = sentenceLengthsInLines(testLines)
    testGen = createVectorsForTest(testLines, wordEmbeddings, char2Index)
    predictedTestTags = predictTags(loadedModel, testGen, testLines, index2POS, index2Chunk, totalTestSamples, batchSize=1)
    writeDataToFile(predictedTestTags, args.out)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    main()
