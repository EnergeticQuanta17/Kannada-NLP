sentence_id = input("Enter the sentence id: ")

with open('../Parsing/full_dataset_113.pickle', 'rb') as file:
    retrieved_sentences = pickle.load(file)

print("Number of sentences retrieved: ", len(retrieved_sentences))


X_predict = []
Y_predict = []
for sentence in retrieved_sentences:
    tempX = []
    tempY = []
    for chunk in sentence.list_of_chunks:
        for word in chunk.list_of_words:
            words.append(word.kannada_word)
            tags.append(word.pos)

            tempX.append(word.kannada_word)
            tempY.append(word.pos)
    
    X_train.append(tempX)
    Y_train.append(tempY)

    if(sentence.id == sentence_id):
        X_predict.append(tempX)
        Y_predict.append(tempY)

words = set(words)
tags = set(tags)


print("\n\n--------------PART OF SPEECH TAGS--------------")
for tag in tags:
    print(tag)
print("\n\n-----------------------------------------------")

print("First 10 words:")
for word in list(words)[:10]:
    print(word)
print()


print('Vocabulary Size: ', len(words))
print('Total POS Tags: ', len(tags))

word2int = {}
int2word = {}

for i, word in enumerate(words):
    word2int[word] = i+1
    int2word[i+1] = word

tag2int = {}
int2tag = {}

for i, tag in enumerate(tags):
    tag2int[tag] = i+1
    int2tag[i+1] = tag

X_train_numberised = []
Y_train_numberised = []

for sentence in X_train:
    tempX = []
    for word in sentence:        
        tempX.append(word2int[word])
    X_train_numberised.append(tempX)

for tags in Y_train:
    tempY = []
    for tag in tags:
        tempY.append(tag2int[tag])
    Y_train_numberised.append(tempY)

X_train_numberised = np.asarray(X_train_numberised)
Y_train_numberised = np.asarray(Y_train_numberised)

pickle_files = [X_train_numberised, Y_train_numberised, word2int, int2word, tag2int, int2tag]


# Loading the model

from keras.models import load_model
model = load_model('Models/model.h5')

test_results = model.evaluate(X_predict, Y_predict, verbose=10)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

y_pred = model.predict(X_predict)
y_test = Y_predict

print("Shape of Y-pred: ", y_pred.shape)
print("Shape of y_test: ", y_test.shape)