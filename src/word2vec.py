import numpy as np
from collections import Counter
import json


class word2vec:
    def __init__(self, embedding_size=100, context_len=6, neg_context_mul=2):
        self.embedding_size = embedding_size
        self.context_len = context_len
        self.neg_context_mul = neg_context_mul
        self.word_to_embedding = {}
        self.data = np.array([])
        self.count_to_words = {}
        self.unique_words = np.array([])
        self.train_data = []
        
        self.subsample_size = 5000
        
    def similarity(self, c, w):
        c = c.reshape(1, -1)
        return np.dot(c, w)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def add_data(self, new_data, count_words=True):
        self.data = np.concatenate((self.data, new_data))
        if(count_words):
            self.count_words_data()

    def count_words_data(self):
        word_counts = Counter(self.data)
        self.count_to_words = {}
        for word, count in word_counts.items():
            if count not in self.count_to_words:
                self.count_to_words[count] = set()
            self.count_to_words[count].add(word)
        
    def find_neg_context(self, i, pos_context):
        word = self.data[i]
        neg_context_len = self.context_len*self.neg_context_mul*2
        word_count = np.count_nonzero(self.data==word)
        neg_context = set()
        specific_counter = 0
        specific_sign = -1

        while len(neg_context) != neg_context_len:
            word_count += specific_sign*specific_counter
            specific_counter += 1
            specific_sign = -specific_sign
            if word_count not in self.count_to_words:
                continue
            for i in self.count_to_words[word_count]:
                if i not in pos_context and i != word:
                    neg_context.add(i)
                    if(len(neg_context) == neg_context_len):
                        break
            
        return np.array([self.word_to_embedding[i] for i in neg_context])
            
    def find_pos_context(self, i):
        mask = (np.arange(len(self.data)) >= i - self.context_len) & (np.arange(len(self.data)) <= i + self.context_len)
        mask &= (np.arange(len(self.data)) != i)
        pos_context = np.array([self.word_to_embedding[i] for i in self.data[mask]])
        return pos_context
    
    def gradient_descent_iter(self, cur_weights, pos_context, neg_context, eta):
        k = 0
        for pos in pos_context:
            k += 1
            neg_grad_part = np.sum([self.sigmoid(self.similarity(neg, cur_weights))*neg for neg in neg_context], axis=0)
            cur_weights -= eta*((self.sigmoid(self.similarity(pos, cur_weights))-1)*pos + neg_grad_part)
        return cur_weights
    
    def logistic_regression(self, i, number_iterations, epsilon):
        word = self.data[i]
        pos_context = self.find_pos_context(i)
        neg_context = self.find_neg_context(i, pos_context)
        cur_weights = self.word_to_embedding[word]
        for i in range(number_iterations):
            new_weights = self.gradient_descent_iter(cur_weights, pos_context, neg_context, 0.1)
            if np.mean(np.abs(cur_weights-new_weights)) < epsilon:
                break
        return new_weights
        
    def convert_input_to_embedding(self, input):
        return np.array([self.word_to_embedding[str(i)] for i in input])
    
    def make_random_embeddings(self):
        for i in self.data:
            self.word_to_embedding[i] = np.random.random(self.embedding_size)
        self.unique_words = np.array(list(self.word_to_embedding.keys()))
    
    def learn(self):
        self.make_random_embeddings()
        for i in range(len(self.data)):
            if(i % 1000 == 0):
                print(f"created embeddings: {i}")
            weights = self.logistic_regression(i, 500, 0.01)
            self.word_to_embedding[self.data[i]] = weights   
    
    def save_embeddings(self, f_name="embeddings.json"):
        embeddings_list = {word: embedding.tolist() for word, embedding in self.word_to_embedding.items()}
        with open(f_name, "w") as f:
            json.dump(embeddings_list, f, indent=4)
        print("Embeddings saved successfully!")
    
    def load_embeddings(self, f_name):
        with open(f_name) as f:
            embedding_list = json.loads(f.read())
        self.word_to_embedding = {word: np.array(embedding) for word, embedding in embedding_list.items()}
        self.unique_words = np.array(list(self.word_to_embedding.keys()))
        print("Embeddings loaded successfully!")
    
