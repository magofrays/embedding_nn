import numpy as np
import os
import logging
import gmpy2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from random import randint

from process_data import clean_word
import json

class SimpleNN:
    def __init__(self, context_len = 6):
        self.nn_model = Sequential()
        self.train_data = ()
        self.data = []
        self.unique_words = []
        self.context_len = context_len
        self.embedding_size = 0
        self.count_words = {}
        self.word_to_embedding = {}
    
    def index_to_one_hot(self, index, length):
        one_hot = np.zeros(length)
        one_hot[index] = 1
        return one_hot
    
    def add_data(self, new_data):
        self.data.extend(new_data)
    
    def train_data_generator(self):
        def generator():
            for i in range(0, len(self.data) - self.context_len - 1):
                split = self.data[i:i + self.context_len + 1]
                X_train = self.convert_input_to_embedding(split[:-1])
                Y_train = self.index_to_one_hot(
                    self.unique_words.index(split[-1]),
                    len(self.unique_words)
                )
                yield (X_train, Y_train)
                
        return generator
    
    def validate_data_generator(self):
        def generator():
            for t in range(1000):
                i = randint(0, len(self.data) - self.context_len-1)
                split = self.data[i:i + self.context_len + 1]
                X_train = self.convert_input_to_embedding(split[:-1])
                Y_train = self.index_to_one_hot(
                    self.unique_words.index(split[-1]),
                    len(self.unique_words)
                )
                yield (X_train, Y_train)
                
        return generator
    
    def compile(self):
        self.nn_model.add(Flatten(input_shape=(self.context_len, self.embedding_size)))
        self.nn_model.add(Dense(1000, activation='relu'))
        self.nn_model.add(Dense(1000, activation='relu'))
        self.nn_model.add(Dense(1000, activation='relu'))
        self.nn_model.add(Dense(len(self.unique_words), activation='softmax'))
        self.nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def create_dataset(self):
        return tf.data.Dataset.from_generator(
            self.train_data_generator(),
            output_signature=(
                tf.TensorSpec(shape=(self.context_len, self.embedding_size), dtype=tf.float32),
                tf.TensorSpec(shape=(len(self.unique_words),), dtype=tf.float32)
            )
        ).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    def create_val_dataset(self):
        return tf.data.Dataset.from_generator(
            self.validate_data_generator(),
            output_signature=(
                tf.TensorSpec(shape=(self.context_len, self.embedding_size), dtype=tf.float32),
                tf.TensorSpec(shape=(len(self.unique_words),), dtype=tf.float32)
            )
        ).shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    def train(self, epochs=10):
        dataset = self.create_dataset()
        val_dataset = self.create_val_dataset()
        
        self.nn_model.fit(
            dataset,
            epochs=epochs,
            validation_data=val_dataset,
        )
    
    def computing_perplexity(self, words, token=False):
        add_sum = 0
        for i in range(len(words) - 1, 0, -1):
            val = ''
            for k in range(i):
                if token:
                    val += str(int(words[k]))+".0" + " "
                else:
                    val += words[k] + ' '
            val = val[:-1]
            c_up = 1
            if val in self.count_words:
                c_up += self.count_words[val]
            c_down = 1
            if words[i] in self.count_words:
                c_down += self.count_words[words[i]]
            add_sum += np.log(c_up / c_down)
        down_val = np.exp(add_sum)
        perplexity = gmpy2.root(1 / down_val, len(words))
        print(f"Perplexity: {perplexity}")
    
    def predict(self, context, verbose=0, token=False):
        cleaned_words = []
        if(len(context) != self.context_len):
            context = context[:6]
        if token:
            cleaned_words = [self.word_to_embedding[int(w)] for w in context]
        if not token:
            cleaned_words = [clean_word(word) for word in context]
            cleaned_words = [self.word_to_embedding[w] for w in context]
        # print(cleaned_words, type(cleaned_words))
        other_context = np.array(cleaned_words).reshape(1, self.context_len, self.embedding_size)
        word_one_hot = self.nn_model.predict(other_context, verbose=verbose)
        predicted_index = np.argmax(word_one_hot, axis=-1)[0]
        predicted_word = self.unique_words[predicted_index]

        perplexity_words = context.copy()
        perplexity_words.append(predicted_word)
        self.computing_perplexity(perplexity_words, token)
        return predicted_word

    def load_embeddings(self, f_dir, token=False):
        with open(f_dir) as f:
            embedding_list = json.loads(f.read())
        if token:
            self.word_to_embedding = {int(float(word)): np.array(embedding) for word, embedding in embedding_list.items()}
        else:
            self.word_to_embedding = {word : np.array(embedding) for word, embedding in embedding_list.items()}
        self.unique_words = list(self.word_to_embedding.keys())
        self.embedding_size = len(self.word_to_embedding[self.unique_words[0]])
        print("Embeddings loaded in NN successfully!")

    def load_word_count(self, f_dir):
        with open(f_dir) as f:
            words = json.loads(f.read())
        self.count_words = {words: counts for words, counts in words.items()}
        print("Words loaded successfully!")

    def convert_input_to_embedding(self, input):
        return np.array([self.word_to_embedding[i] for i in input])

    def load_model(self,  f_dir):
        self.nn_model = tf.keras.models.load_model(f_dir)
        print("Model loaded successfully!")
    
    def save_model(self, f_dir):
        self.nn_model.save(f_dir)
        print("Model saved successfully!")
        

def generate_text(context, model, size, token=False):
    result = context.copy()[:6]
    context = context[:6]
    for i in range(size):
        predict = model.predict(context, token=token)
        context.pop(0)
        result.append(predict)
        context.append(predict)
        # print(len(context))
    return result
