import nltk
import gensim
from keras.models import model_from_json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer

path_to_classifier = '/home/maher/keywordExtraction/model/make a model/onehotClassifier'


def predict(string, bag_of_words, most_common, classifier, label_encoder, onehot_encoder):
    inWords = nltk.word_tokenize(string)
    inWords = [w for w in inWords if w in bag_of_words]

    test_integer_encoded = label_encoder.transform(inWords)
    test_integer_encoded = test_integer_encoded.reshape(len(test_integer_encoded), 1)
    X = onehot_encoder.transform(test_integer_encoded)
    X = np.array([[w] for w in X])

    pred = classifier.predict(X)
    for i in range(len(pred)) :
            if(pred[i] > 0.85 and inWords[i] not in most_common):
                pred[i] = 1
            else :
                pred[i] = 0

    stemmer = SnowballStemmer("english")
    simi_out = [inWords[i] for i in range(len(inWords)) if pred[i] == 1]
    stemmed_out = []
    out = []
    for word in simi_out:
        if stemmer.stem(word) not in stemmed_out:
            out.append(word)
            stemmed_out.append(stemmer.stem(word))

    return np.unique(out)



print('loading the classifier ...')
# load json and create model
# TODO: change the path
json_file = open(path_to_classifier+'/classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_to_classifier + "/model.h5")
print("Loaded model from disk")

print('compiling the classifier')
loaded_model.compile(loss='binary_crossentropy',
                     optimizer= 'RMSprop',
                     metrics=['accuracy'])

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(path_to_classifier + '/label_encoder.npy')

with open(path_to_classifier + "/onehot_encoder.txt", "rb") as fp:   # Unpickling
    onehot_encoder = pickle.load(fp)
with open(path_to_classifier + "/common_words.txt", "rb") as fp:   # Unpickling
    common_words = pickle.load(fp)
with open(path_to_classifier + "/bag_of_words.txt", "rb") as fp:   # Unpickling
    bag_of_words = pickle.load(fp)



def getTags(doc):

    tags = predict(doc, bag_of_words, common_words, loaded_model, label_encoder, onehot_encoder)
    return tags

