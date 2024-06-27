import pymorphy3
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime as dt
from pprint import pprint

ma = pymorphy3.MorphAnalyzer()

def load_model(filename: str = "model.json", weightsfile: str = "model.weights.h5") -> Sequential:
    # load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightsfile)
    print("Loaded model from disk")
    return loaded_model

def clean_text(text):
    text = text.replace("\\", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>2)
    return text

text = input()

descriptions = [clean_text(text)]
operations = {}
reverse_operations = dict(map(reversed,operations.items()))
print("opers:", reverse_operations)
total_operations = len(operations) + 1
maxSequenceLength = len(descriptions[0])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions)
textSequences = tokenizer.texts_to_sequences(descriptions)
X_test = textSequences
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
X_test = 
model = load_model("model_mlp.json", "model_mlp.weights.h5")

predictions = model.predict(X_test, verbose="auto")
result = {}
for i, probability in enumerate(predictions[0][1:]):
    result[reverse_operations[i+1]] = probability
result = [[k, v] for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
# for name, probability in list(sorted(result.items, key=lambda x:result[x[0]], reverse=True)):
#     print(f'{name}: {probability}')
pprint(result[:5])
