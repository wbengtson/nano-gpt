from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from nano_gpt.tokenizer import tokenize_from_chars
from nano_gpt.training_set_creator import create_training_set
from datetime import datetime
import json

class Model(ABC):
    
    @abstractmethod
    def serialize(self, output_location):
        pass

    
    def calculate_loss(self, X_probs_per_token, Y):
        # Cross entropy
        # Better forward / backward pass by merging operations (fused kernels)
        # Avoid infinity in case of exponentiation of big numbers
        return F.cross_entropy(X_probs_per_token, Y)
    
    @abstractmethod
    def calculate_validation_set_loss(self):
        pass
    
    @abstractmethod
    def generate(self, tokens):
        pass



class MultilayerNeuralNetworkProbabilisticLanguageModelBuilder():


    def with_training_set(self, training_set):
        self.training_set = training_set
        return self
    
    def with_block_size(self, block_size):
        self.block_size = block_size
        return self
    
    def with_embedding_size(self, embedding_size):
        self.embedding_size = embedding_size
        return self
    
    def with_hidden_layer_nb_of_neurons(self, hidden_layer_nb_of_neurons):
        self.hidden_layer_nb_of_neurons = hidden_layer_nb_of_neurons
        return self
    
    def with_number_of_training_steps(self, number_of_training_steps):
        self.number_of_training_steps = number_of_training_steps
        return self
    
    def with_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self        

    def with_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def with_training_set_percentage(self, training_set_percentage):
        self.training_set_percentage = training_set_percentage
        return self
    
    def build_and_train(self):
        if self.training_set and self.block_size and self.embedding_size and \
           self.hidden_layer_nb_of_neurons and self.number_of_training_steps and self.learning_rate and \
           self.batch_size and self.training_set_percentage:
            model = MultilayerNeuralNetworkProbabilisticLanguageModel(self.training_set, self.block_size, self.embedding_size, self.hidden_layer_nb_of_neurons, self.number_of_training_steps, self.learning_rate, self.batch_size, self.training_set_percentage)
            return model
        else:
            raise ValueError("One of the parameters has not been initialized")


class MultilayerNeuralNetworkProbabilisticLanguageModel(Model):

    '''
        Model being trained
    '''
    def __init__(self, training_set, block_size, embedding_size, hidden_layer_nb_of_neurons, number_of_training_steps, learning_rate, batch_size, training_set_percentage):
        with(open(training_set)) as ts:
            training_set_text = ts.read()
            words = training_set_text.split(" ")
            self.training_set = training_set.as_uri()
        
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.hidden_layer_nb_of_neurons = hidden_layer_nb_of_neurons

        self.string_to_token_id, self.token_id_to_string = tokenize_from_chars(words)
        
        self.training_set_percentage = training_set_percentage
        self.validation_set_percentage = 1 - training_set_percentage
        self.X_train, self.Y_train, self.X_val, self.Y_val = create_training_set(words, self.block_size, self.string_to_token_id, self.training_set_percentage)
        self.number_of_training_steps = number_of_training_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_losses = []
        self.training_start = datetime.now().isoformat()
        self.__train()
        self.training_end = datetime.now().isoformat()
        self.validation_loss = self.calculate_validation_set_loss()

    def serialize(self, output_location):
        output_dict = dict()
        output_dict["training_set_file"] = self.training_set
        output_dict["block_size"] = self.block_size
        output_dict["embedding_size"] = self.embedding_size
        output_dict["hidden_layer_nb_of_neurons"] = self.hidden_layer_nb_of_neurons
        output_dict["number_of_training_steps"] = self.number_of_training_steps
        output_dict["learning_rate"] = self.learning_rate
        output_dict["batch_size"] = self.batch_size
        output_dict["training_set_percentage"] = self.training_set_percentage
        output_dict["training_start"] = self.training_start
        output_dict["training_end"] = self.training_end
        output_dict["W1"] = self.W1.tolist()
        output_dict["b1"] = self.b1.tolist()
        output_dict["W2"] = self.W2.tolist()
        output_dict["b2"] = self.b2.tolist()
        output_dict["training_losses"] = list(map(lambda x: x.item(), self.training_losses))
        output_dict["validation_loss"] = self.validation_loss.item()
        with open(output_location, "w") as f:
            f.write(json.dumps(output_dict))


    def generate(self, tokens):
        token_ids = list(map(lambda token: self.string_to_token_id[token], tokens))
        embeddings = self.C[token_ids]
        h = torch.tanh(embeddings.view(1, embeddings.shape[0] * embeddings.shape[1]) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        print(logits.shape)
        probs = logits / torch.sum(logits)
        max_value_index = torch.argmax(probs)
        return self.token_id_to_string[max_value_index.item()]
    
    def __train(self):
        number_of_tokens = len(self.string_to_token_id)
        # Embedding layer -> Matrix of size: number of tokens by the dimension of embeddings
        self.C = torch.randn((number_of_tokens, self.embedding_size))
        # Weights hidden layer 1
        self.W1 = torch.randn((self.embedding_size * self.block_size, self.hidden_layer_nb_of_neurons))
        # Bias hidden layer 1
        self.b1 = torch.randn((self.hidden_layer_nb_of_neurons))

        # Weights hidden layer 2
        self.W2 = torch.randn((self.hidden_layer_nb_of_neurons, number_of_tokens))
        # Bias hidden layer 2
        self.b2 = torch.randn(number_of_tokens)

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2] 

        for p in parameters:
            p.requires_grad = True

        for i in range(self.number_of_training_steps):
            batch_indexes = torch.randint(0, self.X_train.shape[0], (self.batch_size, ))
            probs = self.__forward_pass(self.X_train[batch_indexes], self.C, self.W1, self.b1, self.W2, self.b2)
            loss = self.calculate_loss(probs, self.Y_train[batch_indexes])
            self.training_losses.append(loss)
            self.__backward(parameters, loss, self.learning_rate)
    
    def calculate_validation_set_loss(self):
        probs = self.__forward_pass(self.X_val, self.C, self.W1, self.b1, self.W2, self.b2)
        loss = self.calculate_loss(probs, self.Y_val)
        return loss

    
    
    def __forward_pass(self, X, C, W1, b1, W2, b2):
        # Get embedding matrix for each block
        # Size will be 
        # the number of n-grams in X 
        # n as the length of a n-gram (eg 2 for a bigram) 
        # by the dimension of the embedding
        embeddings = C[X]
        h = torch.tanh(embeddings.view(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]) @ W1 + b1)
        logits = h @ W2 + b2
        return logits
    
    def __backward(self, parameters, loss, learning_rate):
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -learning_rate * p.grad





