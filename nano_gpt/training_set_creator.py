import torch
import random

def create_training_set(words, block_size, string_to_token_id, training_set_percentage):
    X, Y = [], []
    context = [0] * block_size
    random.shuffle(words)
    for w in words:
        for ch in list(w) + ['.']:
            target_label = string_to_token_id[ch]
            X.append(context)
            Y.append(target_label)
            context = context[1:] + [target_label] 
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    training_set_max_index = int(X.shape[0] * training_set_percentage)
    return X[:training_set_max_index], Y[:training_set_max_index],  \
        X[training_set_max_index:], Y[training_set_max_index:]