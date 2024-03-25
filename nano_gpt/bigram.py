import torch

# Problem of the BigramModel is that it ignores what came before 
# On evaluation: high property of a bigram means that the model learnt something
# A model running on random should get around 4 percent for each bigram (1/27)
# A good model should have probability near one per line and the rest near 0
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood
class BigramModel:

    def __init__(self, training_set):
        with(open(training_set)) as ts:
            training_set_text = ts.read()
            words = training_set_text.split(" ")
            self.N, self.P, self.string_to_integer, self.integer_to_string = self.__train(words)

    def __train(self, words):
        
        # Using torch matrix
        N = torch.zeros((43,43), dtype=torch.int32)
        chars = sorted(list(set(''.join(words))))
        print(len(chars))
        string_to_integer = {ch:i+1 for i, ch in enumerate(chars)}
        string_to_integer['<S>'] = 0
        integer_to_string = {i:ch for i, ch in enumerate(chars)}
        integer_to_string[0] = '<S>'

        for w in words:
            chs = ['<S>'] + list(w) + ['<S>']
            for ch1, ch2 in zip(chs, chs[1:]):
                row_index = string_to_integer[ch1]
                column_index = string_to_integer[ch2]
                N[row_index, column_index] += 1
        # Model smoothing: Add + 1 to N so to avoid log likelihood of infinite in
        # case there's not a given bigram in the training set 
        P = (N+1).float()
        P /= P.sum(dim=1, keepdim=True)
        return N, P, string_to_integer, integer_to_string
    
    def __sample(self, current_char):
        line_current_char = self.string_to_integer[current_char]
        line = self.N[line_current_char].float()
        probabilities = line/self.P[line_current_char]
        index = torch.multinomial(probabilities, 1, replacement=True).item()
        return self.integer_to_string[index]
    

    def __evaluate_model(self, words):
        log_likelihood = 0
        number_of_bigrams = 0
        for w in words:
            chs = ['<S>'] + list(w) + ['<S>']
            for ch1, ch2 in zip(chs, chs[1:]):
                row_index = self.string_to_integer[ch1]
                column_index = self.string_to_integer[ch2]
                logprob = torch.log(self.P[row_index, column_index])
                log_likelihood = log_likelihood + logprob
                number_of_bigrams = number_of_bigrams + 1
        normalized_log_likelihood = -log_likelihood / number_of_bigrams
        return normalized_log_likelihood
        

    def sample_word(self):
        current_char = '<S>'
        next_char = ''
        word = ''
        while next_char != '<S>':
            next_char = self.__sample(current_char)
            word += next_char
            current_char = next_char
        return word

if __name__=='__main__':
    bg = BigramModel("/Users/wbengtson/Development/nanogpt/nano_gpt/training_set.txt")
    print("Sampling word")
    print(bg.sample_word())
