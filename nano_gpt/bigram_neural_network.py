import torch
import torch.nn.functional as F

class BigramNeuralNetwork:

    '''
        Model being trained
    '''
    def __init__(self, training_set):
        self.W = torch.randn((27, 1), requires_grad=True)
        with(open(training_set)) as ts:
            training_set_text = ts.read()
            words = training_set_text.split(" ")
        self.string_to_token_id, self.token_id_to_string = self.__tokenize(words)
        self.__train(words, 10, self.string_to_token_id)
    
    '''
        Already trained model
    '''
    def __init__(self, W, string_to_token_id, token_id_to_string):
        self.W = W
        self.string_to_token_id = string_to_token_id
        self.token_id_to_string = token_id_to_string
    
    def __init__(self, path):
        # TODO
        None

    def serialize(self, output_path):
        # TODO
        None

    def generate(self, token):
        token_id = self.string_to_token_id[token]
        values = F.one_hot(token_id, num_classes=27).float() @ self.W
        counts = torch.exp(values)
        probs = counts / torch.sum(counts, keepdims=True)
        max_value_index = torch.argmax(probs)
        return self.token_id_to_string[max_value_index]
    
    def __train(self, words, num_epochs, string_to_token_id):
        x, y, x_one_hot_encoded = self.__create_training_set(words, string_to_token_id)
        for i in range(num_epochs):
            probs = self.__forward_pass(x_one_hot_encoded, self.W)
            loss = self.__calculate_loss(probs, y)
            self.__backward(self.W, loss)
    
    def __tokenize(self, words):
        chars = sorted(list(set(''.join(words))))
        string_to_token_id = {ch:i+1 for i, ch in enumerate(chars)}
        string_to_token_id['<S>'] = 0
        token_id_to_string = {i:ch for i, ch in enumerate(chars)}
        token_id_to_string[0] = '<S>'
        return string_to_token_id, token_id_to_string

    def __create_training_set(self, words, string_to_token_id):
        x, y = [], []
        for w in words:
            chs = ['<S>'] + list(w) + ['<S>']
            for ch1, ch2 in zip(chs, chs[1:]):
                row_index = string_to_token_id[ch1]
                column_index = string_to_token_id[ch2]
                x.append(row_index)
                y.append(column_index) 
        x = torch.tensor(x)
        y = torch.tensor(y)
        xenc = F.one_hot(x,num_classes=27).float()
        return x, y, xenc
    
    def __forward_pass(self, one_hot_encoded_tokens, W):
        logits = one_hot_encoded_tokens @ W
        counts = logits.exp()
        return counts / counts.sum(1, keepdims=True)
    
    def __calculate_loss(self, X_probs_per_char, ys):
        loss = -X_probs_per_char[torch.arange(ys.nelement()), ys].log().mean() 
        # regularisation, pushes W to be 0: smoothing. It will avoid weights of growing
        regularisation_strengh = 0.01
        loss += regularisation_strengh * (W**2).mean() 
        return loss

    def __backward(self, W, loss):
        W.grad = None
        loss.backward()
        W.data += 0.1 * W.grad


