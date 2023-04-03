from transformers import AutoTokenizer
from keras.preprocessing.text import Tokenizer as KerasTokenizer
import contractions
import pickle
import re
import os

class Tokenizer:
    def __init__(self, model_name = "gpt2"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
    
    def fit(self, sentences, path = "Weights/keras_tokenizer.pickle"):
        self.keras_tokenizer = KerasTokenizer()
        tokenized_sentences = self.tokenize(sentences, get_token_ids = False)["tokens"]
        self.keras_tokenizer.fit_on_texts(tokenized_sentences)

        directory = "/".join(path.split("/")[:-1])
        os.makedirs(directory, exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.keras_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, path = "Weights/keras_tokenizer.pickle"):
        with open(path, 'rb') as handle:
            self.keras_tokenizer = pickle.load(handle)


    def tokenize(self, sentences, get_token_ids = True):
        if(isinstance(sentences, list) == False):
            sentences = [sentences]
        sentences = [contractions.fix(s) for s in sentences]
        input_ids = self.tok(sentences).input_ids

        tokens = []
        for sent in input_ids:
            tokens.append(["<s>"] + [
                    re.sub(r'[\s]', '_', self.tok.decode(i)) # replace whitespaces with underscores. Word2Vec has trouble embedding words with spaces
                    for i in sent
                ] + ["</s>"])
            
        ret = {"tokens": tokens}
        # print(tokens)
        if get_token_ids:
            ret["token_ids"] = self.keras_tokenizer.texts_to_sequences(tokens)
        
        return ret