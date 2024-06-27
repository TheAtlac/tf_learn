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

def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))
    
    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))
    
    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test

max_features = 1000

start_time = dt.now()
file = open("results.txt", mode='a')

file.write(f"data formatting started at {start_time}")

# df = pd.DataFrame(data)
df_test = pd.read_json("datatest.json")
df_test['Description'] = df_test.apply(lambda x: clean_text(x[u'words']), axis=1)

df_test = df_test.sample(frac=1).reset_index(drop=True)

descriptions_test = df_test['Description']

local_operations = {}
for key,value in enumerate(df_test[u'operation'].unique()):
    local_operations[value] = key + 1

total_operations = len(operations) + 1
df_test['operation_id'] = df_test[u'operation'].map(operations)
operations_test = df_test[u'operation_id']

# total_operations = len(df_test[u'operation_id'].unique()) + 1
print('Всего категорий: {}'.format(total_operations))

max_words = 0
for desc in descriptions_test:
    words = len(desc.split())
    if words > max_words:
        max_words = words
maxSequenceLength = max_words
print('Максимальная длина описания: {} слов'.format(maxSequenceLength))

tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(descriptions_test.tolist())

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences_test = tokenizer_test.texts_to_sequences(descriptions_test.tolist())
X_test = textSequences_test
y_test = operations_test


total_unique_words = len(tokenizer_test.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

# vocab_size = round(total_unique_words/10)
vocab_size = max_features


print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer_test = Tokenizer(num_words=vocab_size)
tokenizer_test.fit_on_texts(descriptions_test)

X_test = tokenizer_test.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_test = to_categorical(y_test, total_operations)
print('y_test shape:', y_test.shape)
duration1 = dt.now() - start_time
file.write(f"data formatted in {duration1}")
# количество эпох\итераций для обучения
epochs = 1
batch_size = 32
# max_features = vocab_size

model = load_model("model_mlp.json", "model_mlp.weights.h5")

score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

duration2 = dt.now() - start_time - duration1
file.write(f"finished in {duration2}")
file.close()
