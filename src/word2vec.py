import numpy as np
from collections import Counter
import json

#context_len == l
class word2vec:
    def __init__(self, embedding_size = 100, context_len = 6, neg_context_mul = 2):
        self.embedding_size = embedding_size
        self.context_len = context_len
        self.neg_context_mul = neg_context_mul
        self.word_to_embedding = {}
        self.data = np.array([])
        self.count_to_words = {}
        self.unique_words = np.array([])
        
        self.subsample_size = 5000
        
    def similarity(self, c, w):
        c = c.reshape(1, -1)
        return np.dot(c, w)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def add_data(self, new_data, count_words = True):
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
        
    def find_neg_context(self, idx, pos_context):
        word = self.data[idx]
        neg_context_len = self.context_len * self.neg_context_mul
        word_count = np.count_nonzero(self.data == word)
        neg_context = set()
        specific_counter = 0
        specific_sign = -1

        while len(neg_context) != neg_context_len:
            word_count += specific_sign * specific_counter
            specific_counter += 1
            specific_sign = -specific_sign
            if word_count not in self.count_to_words:
                continue
            for i in self.count_to_words[word_count]:
                if i not in pos_context and i != word:
                    neg_context.add(i)
                    if len(neg_context) == neg_context_len:
                        break

        return {word : self.word_to_embedding[word] for word in neg_context}
            
    def find_pos_context(self, idx):
        mask = [x for x in range (max(0, idx - self.context_len), min(len(self.data), idx + self.context_len))]
        mask.remove(idx)
        mask = np.array(mask)
        pos_context = {word: self.word_to_embedding[word] for word in self.data[mask]}
        return pos_context
    
    def gradient_descent_iter(self, word_vec, pos_context, neg_context, eta):
        d_dword = np.array([0.0 for _ in range(self.embedding_size)])

        for pos_word, vec_pos in pos_context.items():
            d_dpos = (self.sigmoid(self.similarity(vec_pos, word_vec)) - 1) * word_vec
            d_dword += (self.sigmoid(self.similarity(vec_pos, word_vec)) - 1) * vec_pos
            pos_context[pos_word] -= eta * d_dpos

        for neg_word, vec_neg in neg_context.items():
            d_dneg = self.sigmoid(self.similarity(vec_neg, word_vec)) * word_vec
            d_dword += self.sigmoid(self.similarity(vec_neg, word_vec)) * vec_neg
            neg_context[neg_word] -= eta * d_dneg

        word_vec -= eta * d_dword

        return word_vec, pos_context, neg_context

    def change_weight(self, context):
        for word, vec_word in context.items():
            self.word_to_embedding[word] = vec_word

    def logistic_regression(self, idx, number_iterations, epsilon):
        word = self.data[idx]
        pos_context = self.find_pos_context(idx)
        neg_context = self.find_neg_context(idx, pos_context)
        cur_weights = self.word_to_embedding[word]
        for i in range(number_iterations):
            new_weights, pos_context, neg_context = self.gradient_descent_iter(cur_weights, pos_context, neg_context, 0.1)
            if np.abs(np.linalg.norm(cur_weights - new_weights)) < epsilon:
                break
        self.change_weight(neg_context)
        self.change_weight(pos_context)
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
        with open("../src_data/"+f_name, "w") as f:
            json.dump(embeddings_list, f, indent=4)
        print("Embeddings saved successfully!")
    
    def load_embeddings(self, f_name):
        with open("../src_data/"+f_name) as f:
            embedding_list = json.loads(f.read())
        self.word_to_embedding = {word: np.array(embedding) for word, embedding in embedding_list.items()}
        self.unique_words = np.array(list(self.word_to_embedding.keys()))
        print("Embeddings loaded successfully!")
    
