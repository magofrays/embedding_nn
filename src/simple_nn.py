import numpy as np
import os
import logging
import gmpy2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

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
        self.count_words = {}
    
    def index_to_one_hot(self, index, length):
        one_hot = np.zeros(length)
        one_hot[index] = 1
        return one_hot
    
    def add_data(self, new_data):
        self.data = np.concatenate((self.data, new_data))
    
    def train_data_generator(self):
        def generator():
            # Проходим по всем возможным последовательностям
            for i in range(0, len(self.data) - self.context_len - 1):
                # Получаем последовательность
                sequence = self.data[i:i + self.context_len + 1]
                
                # Создаем X (эмбеддинги)
                x = self.convert_input_to_embedding(sequence[:-1])
                
                # Создаем Y (one-hot encoding)
                y = self.index_to_one_hot(
                    self.unique_words.tolist().index(sequence[-1]),
                    len(self.unique_words)
                )
                
                yield x, y
                
        return generator
    
    # def make_train_data(self, start, end):
    #     splits = []
    #     for i in range(start, end - self.context_len - 1):
    #         splits.append(self.data[i : i + self.context_len + 1])
    #     splits = np.array(splits)
    #     X_train = np.array([self.convert_input_to_embedding(s[:-1]) for s in splits])
    #     Y_train = np.array([self.index_to_one_hot(self.unique_words.tolist().index(s[-1]), len(self.unique_words)) for s in splits])
    #     print(f"Created train data size of: {self.subsample_size}")
    #     self.train_data = (X_train, Y_train)
    
    def compile(self):
        self.nn_model.add(Flatten(input_shape=(self.context_len, self.embedding_size)))
        self.nn_model.add(Dense(500, activation='relu'))
        self.nn_model.add(Dense(1000, activation='relu'))
        self.nn_model.add(Dense(len(self.unique_words), activation='softmax'))
        self.nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def create_dataset(self):
        return tf.data.Dataset.from_generator(
            self.data_generator(),
            output_signature=(
                tf.TensorSpec(shape=(self.context_len, embedding_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(len(self.unique_words),), dtype=tf.float32)
            )
        ).batch(32).prefetch(tf.data.AUTOTUNE)
    
    
    def train(self, epochs=10, first_subsample_only=False):
        # Создаем dataset
        dataset = self.create_dataset()
        
        # Если нужно обучать только на части данных
        if first_subsample_only:
            dataset = dataset.take(self.subsample_size // 32)  # 32 - batch size
            
        # Вычисляем steps per epoch
        steps_per_epoch = len(self.data) // self.subsample_size if not first_subsample_only else None
        
        # Запускаем обучение
        self.nn_model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
    # def train(self, epochs=10, first_subsample_only = False):
    #     subsample_number = len(self.data)//self.subsample_size
    #     if first_subsample_only:
    #         self.make_train_data(0*self.subsample_size, (0+1)*self.subsample_size)
    #         self.nn_model.fit(self.train_data[0], self.train_data[1], epochs=epochs, batch_size=32)
    #         return
    
    #     for i in range(subsample_number):
    #         print(f"Iteration {i+1} out of {subsample_number}")
    #         self.make_train_data(i*self.subsample_size, (i+1)*self.subsample_size)
    #         self.nn_model.fit(self.train_data[0], self.train_data[1], epochs=epochs, batch_size=32)
    #     print("Training completed!")

    def computing_perplexity(self, words):
        add_sum = 0
        for i in range(len(words) - 1, 0, -1):
            val = ''
            for k in range(i):
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
    
    def predict(self, str_context, verbose=0):
        cleaned_words = [clean_word(word) for word in str_context]
        if(len(cleaned_words) != self.context_len):
            raise ValueError(f"Context must be length of: {self.context_len}")
        context = [self.word_to_embedding[w] for w in cleaned_words]
        context = np.array(context).reshape(1, self.context_len, self.embedding_size)
        word_one_hot = self.nn_model.predict(context, verbose=verbose)
        predicted_index = np.argmax(word_one_hot, axis=-1)[0]
        predicted_word = self.unique_words[predicted_index]
        cleaned_words.append(predicted_word)
        self.computing_perplexity(cleaned_words)
        return predicted_word

    def load_embeddings(self, f_dir):
        with open(f_dir) as f:
            embedding_list = json.loads(f.read())
        self.word_to_embedding = {word: np.array(embedding) for word, embedding in embedding_list.items()}
        self.unique_words = np.array(list(self.word_to_embedding.keys()))
        self.embedding_size = len(self.word_to_embedding[self.unique_words[0]])
        print("Embeddings loaded in NN successfully!")

    def load_word_count(self, f_dir):
        with open(f_dir) as f:
            words = json.loads(f.read())
        self.count_words = {words: counts for words, counts in words.items()}
        print("Words loaded successfully!")

    def convert_input_to_embedding(self, input):
        return np.array([self.word_to_embedding[str(i)] for i in input])

    def load_model(self,  f_dir):
        self.nn_model = tf.keras.models.load_model(f_dir)
        print("Model loaded successfully!")
    
    def save_model(self, f_dir):
        self.nn_model.save(f_dir)
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
