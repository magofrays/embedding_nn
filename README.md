# this is simple nlp program based on books of ozon671games3 books and 1984
## this is educational project
### there are two ways of using it:
#### - use without tokenizer:
1. tests/save_embeddings.py - to create embeddings for model
2. tests/save_nn.py - to create nn that uses your embeddings
3. tests/load_nn.py - to test model answers and get perplexity
#### - use with tokenizer:
1. tests/token/create_tokens.py - to train tokenizer and split data to tokens
2. tests/token/save_embeddings.py - to create from tokens embeddings
3. tests/token/save_nn.py - to create nn that predicts words from embeddings
4. tests/token/load_nn.py - to test model answers and get perplexity
##### unfortunately this model has only dense layers. So it's accuracy is only 0.05 without tokens and 0.40 with tokens.
