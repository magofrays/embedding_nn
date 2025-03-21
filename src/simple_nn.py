import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from process_data import clean_word
import json

class SimpleNN:
    def __init__(self, context_len = 6, subsample_size = 5000):
        self.nn_model = Sequential()
        self.train_data = ()
        self.data = np.array([])
        self.unique_words = np.array([])
        self.subsample_size = subsample_size
        self.context_len = context_len
        self.embedding_size = 0
    
    def index_to_one_hot(self, index, length):
        one_hot = np.zeros(length)
        one_hot[index] = 1
        return one_hot
    
    def add_data(self, new_data):
        self.data = np.concatenate((self.data, new_data))
    
    def make_train_data(self, start, end):
        splits = []
        for i in range(start, end - self.context_len - 1):
            splits.append(self.data[i : i + self.context_len + 1])
        splits = np.array(splits)
        X_train = np.array([self.convert_input_to_embedding(s[:-1]) for s in splits])
        Y_train = np.array([self.index_to_one_hot(self.unique_words.tolist().index(s[-1]), len(self.unique_words)) for s in splits])
        print(f"Created train data size of: {self.subsample_size}")
        self.train_data = (X_train, Y_train)
    
    def compile(self):
        self.nn_model.add(Flatten(input_shape=(self.context_len, self.embedding_size)))
        self.nn_model.add(Dense(500, activation='relu'))
        self.nn_model.add(Dense(len(self.unique_words), activation='softmax'))
        self.nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def train(self, epochs=10, first_subsample_only = False):
        subsample_number = len(self.data)//self.subsample_size
        if first_subsample_only:
            self.make_train_data(0*self.subsample_size, (0+1)*self.subsample_size)
            self.nn_model.fit(self.train_data[0], self.train_data[1], epochs=epochs, batch_size=32)
            return
    
        for i in range(subsample_number):
            print(f"Iteration {i} out of {subsample_number}")
            self.make_train_data(i*self.subsample_size, (i+1)*self.subsample_size)
            self.nn_model.fit(self.train_data[0], self.train_data[1], epochs=epochs, batch_size=32)
        print("Training completed!")
    
    def predict(self, str_context, verbose=0):
        cleaned_words = [clean_word(word) for word in str_context]
        if(len(cleaned_words) != self.context_len):
            raise ValueError(f"Context must be length of: {self.context_len}")
        context = [self.word_to_embedding[w] for w in cleaned_words]
        context = np.array(context).reshape(1, self.context_len, self.embedding_size)
        word_one_hot = self.nn_model.predict(context, verbose=verbose)
        predicted_index = np.argmax(word_one_hot, axis=-1)[0]
        predicted_word = self.unique_words[predicted_index]
        return predicted_word

    def load_embeddings(self, f_name):
        with open("../src_data/"+f_name) as f:
            embedding_list = json.loads(f.read())
        self.word_to_embedding = {word: np.array(embedding) for word, embedding in embedding_list.items()}
        self.unique_words = np.array(list(self.word_to_embedding.keys()))
        self.embedding_size = len(self.word_to_embedding[self.unique_words[0]])
        print("Embeddings loaded successfully!")

    def convert_input_to_embedding(self, input):
        return np.array([self.word_to_embedding[str(i)] for i in input])

    def load_model(self,  f_name="model"):
        self.nn_model = tf.keras.models.load_model(f"../src_data/{f_name}.keras")
        print("Model loaded successfully!")
    
    def save_model(self, f_name="model"):
        self.nn_model.save(f"../src_data/{f_name}.keras")
        print("Model saved successfully!")
        

def generate_text(context, model, size):
    context = context.split()
    result = context.copy()
    for i in range(size):
        predict = str(model.predict(context))
        context.pop(0)
        result.append(predict)
        context.append(predict)
    return " ".join(result)
