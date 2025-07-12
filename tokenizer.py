from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def encode(text):
    return tokenizer.encode(text).ids

def decode(token_ids):
    return tokenizer.decode(token_ids)