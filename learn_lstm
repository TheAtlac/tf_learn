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

def save_model(model: Sequential, filename: str = "model.json", weightsfile: str = "model.weights.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weightsfile)
    print("Saved model to disk")


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
file = open("results_lstm.txt", mode='a')

file.write(f"data formatting started at {start_time}\n")

df = pd.read_json("datapart.json")
df['Description'] = df.apply(lambda x: clean_text(x[u'words']), axis=1)

df = df.sample(frac=1).reset_index(drop=True)

descriptions = df['Description']

# operations = {}
# for key,value in enumerate(df[u'operation'].unique()):
#     operations[value] = key + 1
total_operations = len(operations) + 1
# total_operations = len(operations)
df['operation_id'] = df[u'operation'].map(operations)
operations = df[u'operation_id']


# total_operations = len(df[u'operation_id'].unique()) + 1
print('Всего категорий: {}'.format(total_operations))
max_words = 0
for desc in descriptions:
    words = len(desc.split())
    if words > max_words:
        max_words = words
maxSequenceLength = max_words
print('Максимальная длина описания: {} слов'.format(maxSequenceLength))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())


# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())
X_train = textSequences
y_train = operations


print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

total_unique_words = len(tokenizer.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

# vocab_size = round(total_unique_words/10)
vocab_size = max_features


print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(descriptions)


X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
print('Размерность X_train:', X_train.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = to_categorical(y_train, total_operations)
print('y_train shape:', y_train.shape)
duration1 = dt.now() - start_time
file.write(f"data formatted in {duration1}\n")

# количество эпох\итераций для обучения
epochs = 5
batch_size = 32
# max_features = vocab_size

print(u'Собираем модель...')
model = Sequential()
model.add(Embedding(max_features, maxSequenceLength))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(total_operations, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    verbose=1)

duration2 = dt.now() - start_time - duration1
file.write(f"finished in {duration2}\n")
file.write("-----------------------------------\n")
file.close()
save_model(model, "model_lstm.json", "model_lstm.weights.h5")
