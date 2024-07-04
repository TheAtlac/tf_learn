# -*- coding: utf-8 -*-
import sqlite3
import pymorphy3
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GRU, LSTM, Conv1D, GlobalMaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from pprint import pprint
import pickle
import csv

import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords

from tensorflow.python.client import device_lib
import os
import json
import time

def time_of_function(function):
    def wrapped(*args, **kwargs):
        start_time = time.perf_counter_ns()
        res = function(*args, **kwargs)
        print(time.perf_counter_ns() - start_time)
        duration = time.perf_counter_ns() - start_time
        return res, duration
    return wrapped

class NeuralNetworkOperator:
    def __init__(self, config:dict, operations_dict:dict[str: int]={}) -> None:
        self.config = config
        self.ma = pymorphy3.MorphAnalyzer()
        self.nlp = spacy.load("ru_core_news_sm")
        self.stop_words = set(stopwords.words('russian'))
        self.operations_dict = operations_dict
        self.total_operations = len(operations_dict)
    
    def clean_sequence(self, text):
        text = text.replace("\\", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
        text = " ".join(self.ma.parse(str(word))[0].normal_form for word in text.split() if word not in self.stop_words)
        return text

    def clean_data(self, df:pd.DataFrame, sample=False):
        df['words'] = df.apply(lambda x: self.clean_sequence(x[u'words']), axis=1)
        if sample:
            df = df.sample(frac=1).reset_index(drop=True)
        # return df


    def create_tokenizer(self, descriptions):
        tokenizer = Tokenizer(num_words=self.config["max_features"])
        tokenizer.fit_on_texts(descriptions.tolist())
        return tokenizer
    
    def prepare_from_cleaned(self, df:pd.DataFrame, operations_dict:dict[str: int], tokenizer:Tokenizer, drop=False):
        df = df.sample(frac=1).reset_index(drop=drop)
        descriptions = df['words']
        if operations_dict == {}:
            operations_dict = self.generate_operations([df])
        self.total_operations = len(operations_dict)

        # print("opers:", df['words'])
        # print("opers:", df['operation'])
        operations = df['operation'].map(operations_dict)
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        # print("opers:", df['words'])
        # print("opers:", df['operation'])
        textSequences = sequence.pad_sequences(textSequences, maxlen=self.config['max_len'])
        operations = to_categorical(operations, self.total_operations)
        return textSequences, operations
    
    def prepare_for_test(self, df:pd.DataFrame, operations_dict:dict[str: int], tokenizer:Tokenizer):
        descriptions = df['words']
        if operations_dict == {}:
            operations_dict = self.generate_operations([df])
        self.total_operations = len(operations_dict)
        operations = df['operation'].map(operations_dict)
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())
        textSequences = sequence.pad_sequences(textSequences, maxlen=self.config['max_len'])
        return textSequences, operations

    
    @staticmethod
    def split_data(strings, labels, train_test_split=0.9):
        data_size = len(strings)
        test_size = int(data_size - round(data_size * train_test_split))
        x_train = strings[test_size:]
        y_train = labels[test_size:]
        
        x_test = strings[:test_size]
        y_test = labels[:test_size]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def save_tokenizer(tokenizer, filename: str):
        print("Сохранение токена")
        with open(filename, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Токен сохранен")

    @staticmethod
    def load_tokenizer(filename: str ='tokenizer.pickle'):
        with open(filename, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    
    @staticmethod
    def generate_operations(*dfs:list[pd.DataFrame], file_to_save=""):
        res = pd.concat(dfs, axis=0)
        opers = enumerate((sorted(res["operation"].unique())))
        operations_dict ={}
        # reverse_operations_dict = {}
        for i, oper in opers:
            operations_dict[oper] = i
            # reverse_operations_dict[i] = oper
        # print(operations_dict)
        if file_to_save:
            with open(file_to_save, 'w', encoding="utf-8") as fp:
                json.dump(operations_dict, fp, ensure_ascii=False, indent=4)
        return operations_dict
    
    def create_gru(self, neurons:list[int]=[16]):
        model = Sequential()
        model.add(Embedding(self.config['max_features'], 32, input_length=self.config['max_len']))
        for num in neurons:
            model.add(GRU(num))
        model.add(Dense(self.total_operations, activation="softmax"))
        return model
    
    def create_lstm(self, neurons:list[int]=[32]):
        model = Sequential()
        model.add(Embedding(self.config['max_features'], 32, input_length=self.config['max_len']))
        for num in neurons:
            model.add(LSTM(num))
        model.add(Dense(self.total_operations, activation="softmax"))
        return model
    
    def create_embedding(self, neurons:list[int]=[16]):
        model = Sequential()
        model.add(Embedding(self.config['max_features'], 32, input_length=self.config['max_len']))
        model.add(Conv1D(125, 5, padding="valid", activation="relu"))
        model.add(GlobalMaxPooling1D())
        for num in neurons:
            model.add(Dense(num, activation="relu"))
        model.add(Dense(self.total_operations, activation="softmax"))
        return model
    

    def create_mlp(self, neurons:list[int]=[512]):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.config['max_len'], ), activation='relu', kernel_initializer='he_uniform'))
        for num in neurons:
            model.add(Dense(num, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dropout(0.05))
        model.add(Dense(self.total_operations, activation="softmax"))
        return model
    
    def model_fabric_method(self, model_name:str, *args):
        if model_name == "gru":
            return self.create_gru(*args)
        if model_name == "lstm":
            return self.create_lstm(*args)
        if model_name == "embedding":
            return self.create_embedding(*args)
        if model_name == "mlp":
            return self.create_mlp(*args)
        else:
            print("Модели с таким названием нет")
            return None
        
    @time_of_function
    def learn_model(self, model: Sequential, x, y, epochs_num:int=5, path_to_save:str="", lr=0.01, verbose=1):
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=lr),
                    # optimizer="adam",
                    metrics=['accuracy'])
        callbacks = []
        if path_to_save:
            checkpoint_callback = ModelCheckpoint(path_to_save, monitor="val_accuracy", save_best_only=True)
            callbacks.append(checkpoint_callback)
        
        print(model.summary())

        history = model.fit(x, y,
                            epochs=epochs_num,
                            batch_size=self.config["batch_size"], validation_split=0.1,
                            callbacks=callbacks,
                            verbose=verbose)
        return history
    
    @staticmethod
    def full_cycle(train_filename="data.json",
                    test_filename="datatest.json",
                    cleaned_train='',
                    cleaned_test='',
                    model_name="mlp",
                    neurons = [32],
                    learning_rate=1e-3,
                    epochs_num=50,
                    config={"max_len": 1000, "max_features": 10000, "batch_size": 256},
                    tokenizer_savepath="",
                    model_savepath="",
                    operations_savepath="",
                    verbose=1):
        nn = NeuralNetworkOperator(config)
        df_train = pd.read_json(train_filename)
        df_test = pd.read_json(test_filename)
        operations_dict = nn.generate_operations(df_train, df_test, file_to_save=operations_savepath)
        nn.clean_data(df_train)
        nn.clean_data(df_test)
        if cleaned_train:
            df_train.to_pickle(cleaned_train)
        if cleaned_test:
            df_test.to_pickle(cleaned_test)
        
        tokenizer = nn.create_tokenizer(df_train["words"])
        if tokenizer_savepath:
            nn.save_tokenizer(tokenizer, tokenizer_savepath)
        x_train, y_train = nn.prepare_from_cleaned(df_train, operations_dict, tokenizer)
        x_test, y_test = nn.prepare_from_cleaned(df_test, operations_dict, tokenizer)
        model = nn.model_fabric_method(model_name, neurons)
        history, duration = nn.learn_model(model, x_train, y_train, epochs_num=epochs_num, path_to_save=model_savepath, lr=learning_rate, verbose=verbose)
        
        score = model.evaluate(x_test, y_test,
                            batch_size=config["batch_size"], verbose=1)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))
        return score
    
    @staticmethod
    def particle_cycle_by_files(train_filename="cleaned_data.pkl",
                    test_filename="cleaned_datatest.pkl",
                    operations_path="operations.json",
                    tokenizer_path="tokenizer.pkl",
                    model_name="mlp",
                    neurons=[32],
                    learning_rate=1e-3,
                    epochs_num=50,
                    config={"max_len": 1000, "max_features": 10000, "batch_size": 256},
                    model_savepath="",
                    results_filename='',
                    verbose=1,
                    drop=False):
        df_train = pd.read_pickle(train_filename)
        df_test = pd.read_pickle(test_filename)
        nn = NeuralNetworkOperator(config)
        with open(operations_path, 'r', encoding="utf8") as f:
            operations_dict = json.loads(f.read())
        tokenizer = nn.load_tokenizer(tokenizer_path)
        
        x_train, y_train = nn.prepare_from_cleaned(df_train, operations_dict, tokenizer, drop=drop)
        x_test, y_test = nn.prepare_from_cleaned(df_test, operations_dict, tokenizer)
        model = nn.model_fabric_method(model_name, neurons)
        # model = nn.create_gru(neurons=[32], lr=0.05)
        
        path = f"models/{model_name}_{epochs_num}_{'%.5f' % learning_rate}_{len(neurons)}_{neurons[0]}.keras"

        history, duration = nn.learn_model(model, x_train, y_train, epochs_num=epochs_num, path_to_save=path, lr=learning_rate, verbose=verbose)
    
        score = model.evaluate(x_test, y_test,
                        batch_size=config["batch_size"], verbose=1)
        if results_filename:
            with open(results_filename, 'a') as f:
                row = [model_name, epochs_num, '%.3f' % learning_rate, len(neurons), neurons[0], '%.3f' % score[1], int(duration/10**9)]
                print(row)
                file_writer = csv.writer(f, delimiter = ";", lineterminator="\r")
                file_writer.writerow(row)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

        # График точности модели
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        # plt.plot(history.history['loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.ylabel('val_accuracy')
        # plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return score

    def batch(self, train_filename="cleaned_data.pkl",
                    test_filename="cleaned_datatest.pkl",
                    tokenizer_path="tokenizer.pkl",
                    operations_path="",
                    model_name="mlp",
                    neurons_lists=[[32]],
                    learning_rates=[5e-4, 1e-3],
                    epochs_nums=[10, 25, 50, 100],
                    results_filename="results.csv",
                    verbose=1,
                    drop=False
                    ):
        
        df_train = pd.read_pickle(train_filename)
        df_test = pd.read_pickle(test_filename)
        if operations_path:
            with open(operations_path, 'r', encoding="utf8") as f:
                operations_dict = json.loads(f.read())
        else:
            operations_dict = self.operations_dict
        tokenizer = self.load_tokenizer(tokenizer_path)
        
        x_train, y_train = self.prepare_from_cleaned(df_train, operations_dict, tokenizer, drop)
        x_test, y_test = self.prepare_from_cleaned(df_test, operations_dict, tokenizer)
        # with open(results_filename, 'a') as f:
        #     heads = ["модель", "эпохи", "скорость обучения", "слоёв", "нейронов в слое", "точность", "время(c)"]
        #     file_writer = csv.writer(f, delimiter = ";", lineterminator="\r")
        #     file_writer.writerow(heads)
        for epochs_num in epochs_nums:
            for learning_rate in learning_rates:
                for neurons in neurons_lists:
                    
                    path = f"models/{model_name}_{epochs_num}_{'%.5f' % learning_rate}_{len(neurons)}_{neurons[0]}.keras"
                    model = self.model_fabric_method(model_name, neurons)

                    history, duration = self.learn_model(model, x_train, y_train, epochs_num=epochs_num, path_to_save=path, lr=learning_rate, verbose=verbose)
                
                    score = model.evaluate(x_test, y_test,
                                    batch_size=self.config["batch_size"], verbose=1)
                    with open(results_filename, 'a') as f:
                        row = [model_name, epochs_num, '%.3f' % learning_rate, len(neurons), neurons[0], '%.3f' % score[1], int(duration/10**9)]
                        print(row)
                        file_writer = csv.writer(f, delimiter = ";", lineterminator="\r")
                        file_writer.writerow(row)
    
    def predict(self, model, text, tokenizer_path:str='tokenizer.pkl', operations_path="", size=5):
        if operations_path:
            with open(operations_path, 'r', encoding="utf8") as f:
                operations_dict = json.loads(f.read())
        else:
            operations_dict = self.operations_dict
        reverse_operations = dict(map(reversed,operations_dict.items()))
        self.total_operations = len(operations_dict)

        text = self.clean_sequence(text)
        tokenizer = self.load_tokenizer(tokenizer_path) 
        textSequences = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(textSequences, maxlen=self.config['max_len'])
        prediction = model.predict(x)
        ind = sorted(range(self.total_operations), key=lambda x: prediction[0][x])[-size:][::-1]
        res = list(zip(map(lambda x: reverse_operations[x], ind), prediction[0][ind]))
        pprint(res)
        print('операция: {}({})'.format(reverse_operations[np.argmax(prediction[0])] , np.argmax(prediction[0])))
        return res
    
    def test(self, model_path, test_filename="cleaned_datatest.pkl",
                    tokenizer_path="new_tokenizer.pkl", operations_path="", size=5):
        model = load_model(model_path)
        if operations_path:
            with open(operations_path, 'r', encoding="utf8") as f:
                operations_dict = json.loads(f.read())
        else:
            operations_dict = self.operations_dict
        df_test = pd.read_pickle(test_filename)
        operations_dict = self.operations_dict
        tokenizer = self.load_tokenizer(tokenizer_path)
        total_operations = len(operations_dict)
        x_test, y_test = self.prepare_for_test(df_test, operations_dict, tokenizer)
        total = len(y_test)
        step = total / 10
        s1 = s2 = 0
        predictions = model.predict(np.array(x_test), verbose=0)
        for i in range(total):
            # if i % step == 0 and i !=0:
            #     print(f'{i}/{total}'.rjust(10), end="\t\t")
            #     print("точность:", "%.3f" % (s1/i), end="\t\t")
            #     print(f"из {size} вариантов:", "%.3f" % (s2/i))
            prediction = predictions[i]
            ind = sorted(range(total_operations), key=lambda x: prediction[x])[-size:][::-1]
            # print(*zip(ind, prediction[0][ind]))
            # print('получено операция: {}({})/{}({})'.format(reverse_operations[np.argmax(prediction[0])] , np.argmax(prediction[0]), reverse_operations[y_test[i]], y_test[i]))
            # print('---------------------------------')
            if y_test[i] == ind[0]:
                s1 += 1
                s2 += 1
            elif y_test[i] in ind:
                s2 += 1
        print("точность:", s1/i, end=" ")
        print(f"точность из {size} вариантов:", s2/i)

def mlp_learn():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    neurs= [[20480, 20480, 20000, 20000]]
    #  + list(np.arange(1e-3, 1e-2, 1e-3)
    # lrs = list(np.arange(1e-3, 1e-2, 5e-3))
    lrs = [2e-3, 1e-3, 8e-4, 5e-4]
    # epochs_list = [5, 10, 15, 20, 25]
    # epochs_list = list(range(500, 1000, 500))
    epochs_list = [500, 2000, 500]
    nn.batch(train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
            tokenizer_path="tokenizer.pkl", model_name="mlp", neurons_lists=neurs,
            learning_rates=lrs, epochs_nums=epochs_list, verbose=0)
    
def gru_learn():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    neurs= [[16]]
    # neurs = [[16, 16]]
    # lrs = list(np.arange(1e-3, 1e-2, 1e-3)) + list(np.arange(1e-4, 1e-3, 1e-4))
    # lrs = list(np.arange(1e-3, 1e-2, 3e-3))
    epochs_list = list(range(10, 55, 10))
    lrs = list(np.arange(1e-3, 1e-2, 3e-3))
    nn.batch(train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
            tokenizer_path="tokenizer.pkl", model_name="gru", neurons_lists=neurs,
            learning_rates=lrs, epochs_nums=epochs_list, verbose=0)

def lstm_learn():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    neurs= [[64]]
    # neurs = [[16]]
    lrs = list(np.arange(1e-3, 1e-2, 3e-3))
    #  + list(np.arange(1e-4, 1e-3, 3e-4))
    # lrs = [1e-3]
    epochs_list = list(range(10, 55, 10))
    nn.batch(train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
            tokenizer_path="tokenizer.pkl", model_name="lstm", neurons_lists=neurs,
            learning_rates=lrs, epochs_nums=epochs_list, verbose=0)

def embedding_learn():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    neurs= [[128]]
    # neurs = [[16, 16]]
    # lrs = list(np.arange(1e-3, 1e-2, 1e-3)) + list(np.arange(1e-4, 1e-3, 1e-4))
    # lrs = list(np.arange(1e-4, 1e-3, 3e-4))
    # lrs = [1e-4]
    lrs = list(np.arange(1e-3, 1e-2, 3e-3))
    epochs_list = [15]
    nn.batch(train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
            tokenizer_path="tokenizer.pkl", model_name="embedding", neurons_lists=neurs,
            learning_rates=lrs, epochs_nums=epochs_list, verbose=1)

def single_learn_example():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    neurs= [256]
    # neurs = [[16, 16]]
    # lrs = list(np.arange(1e-3, 1e-2, 1e-3)) + list(np.arange(1e-4, 1e-3, 1e-4))
    # lrs = list(np.arange(1e-4, 1e-3, 3e-4))
    lr = 1e-3
    epochs = 13
    nn.particle_cycle_by_files(train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
            tokenizer_path="tokenizer.pkl", model_name="embedding", neurons=neurs,
            learning_rate=lr, epochs_num=epochs, verbose=1, drop=True, results_filename='results.csv')

def predict_example():
    text="Прошу починить принтер на 3 этаже в 123 кабинете как можно скорее"
    # text = input()
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    model = load_model("best_gru.keras")
    res = nn.predict(model, text)
    print(res[0])

def full_cycle_example():
    NeuralNetworkOperator.full_cycle(train_filename="data.json",
                                    test_filename="datatest.json",
                                    cleaned_train='cleaned_data2.pkl',
                                    cleaned_test='cleaned_datatest2.pkl',
                                    tokenizer_savepath="tokenizer2.pkl",
                                    model_name="gru", neurons=[128],
                                    learning_rate=8e-4, epochs_num=25,
                                    model_savepath="best_full2.keras",
                                    operations_savepath="operations2.json",
                                    verbose=1)

def particle_cycle_example():
    NeuralNetworkOperator.particle_cycle_by_files(
        train_filename="cleaned_data.pkl", test_filename="cleaned_datatest.pkl",
          tokenizer_path="tokenizer.pkl", model_name="gru", neurons=[128],
            learning_rate=8e-4, epochs_num=15, model_savepath="best_gru.keras")

def test_example():
    with open("operations.json", 'r', encoding="utf8") as f:
        operations_dict = json.loads(f.read())

    with open("config.json", 'r') as f:
        config = json.loads(f.read())
    nn = NeuralNetworkOperator(config, operations_dict)
    nn.test("models/embedding_13_0.00100_1_256.keras", "cleaned_datatest.pkl", 'tokenizer.pkl')

if __name__ == "__main__":
    # full_cycle_example()
    test_example()
    # single_learn_example()
    # embedding_learn()