from abc import ABC, abstractmethod
from importlib.resources import files
import torch
import torch.nn.functional as F
from nano_gpt.tokenizer import tokenize_from_chars
from nano_gpt.training_set_creator import create_training_set
from datetime import datetime
import json
import nano_gpt

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

    def __init__(self) -> None:
        self.training_set = None
        self.block_size = None
        self.embedding_size = None
        self.hidden_layer_nb_of_neurons = None
        self.number_of_training_steps = None
        self.learning_rate = None
        self.batch_size = None
        self.training_set_percentage = None
        self.generator_seed = None

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
    
    def with_generator_seed(self, generator_seed):
        self.generator_seed = generator_seed
        return self

    def build_and_train(self):
        if self.training_set and self.block_size and self.embedding_size and \
           self.hidden_layer_nb_of_neurons and self.number_of_training_steps and self.learning_rate and \
           self.batch_size and self.training_set_percentage:
            generator = torch.Generator()
            if self.generator_seed:
                generator.manual_seed(self.generator_seed)
            model = MultilayerNeuralNetworkProbabilisticLanguageModel(self.training_set, self.block_size, self.embedding_size, self.hidden_layer_nb_of_neurons, self.number_of_training_steps, self.learning_rate, self.batch_size, self.training_set_percentage, generator)
            return model
        else:
            raise ValueError("One of the parameters has not been initialized")


class MultilayerNeuralNetworkProbabilisticLanguageModel(Model):

    '''
        Model being trained
    '''
    def __init__(self, training_set, block_size, embedding_size, hidden_layer_nb_of_neurons, number_of_training_steps, \
                 learning_rate, batch_size, training_set_percentage, generator, \
                 W1=None, W2=None, b1=None, b2=None, C=None, training_start=None, training_end=None, validation_loss=None,
                 string_to_token_id=None, token_id_to_string=None, training_losses=None):
        
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.hidden_layer_nb_of_neurons = hidden_layer_nb_of_neurons
        
        self.generator = generator
        self.training_set_percentage = training_set_percentage
        self.validation_set_percentage = 1 - training_set_percentage
        
        self.number_of_training_steps = number_of_training_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        

        if W1 is not None and W2 is not None and b1 is not None and b2 is not None and C is not None:
            self.W1= W1
            self.W2 = W2
            self.b1 = b1
            self.b2 = b2
            self.C = C
            self.training_start = training_start
            self.training_end = training_end
            self.validation_loss = torch.tensor(validation_loss)
            self.string_to_token_id = string_to_token_id
            self.token_id_to_string = token_id_to_string
            self.training_losses = training_losses
            self.training_set = training_set
        else:
            training_set_file = files(nano_gpt).joinpath(training_set)
            with(open(training_set_file)) as ts:
                training_set_text = ts.read()
                words = training_set_text.split(" ")
            self.training_set = training_set_file.as_uri()
            self.string_to_token_id, self.token_id_to_string = tokenize_from_chars(words)
            self.X_train, self.Y_train, self.X_val, self.Y_val = create_training_set(words, self.block_size, self.string_to_token_id, self.training_set_percentage)
            self.training_losses = []
            self.training_start = datetime.now().isoformat()
            self.__train()
            self.training_end = datetime.now().isoformat()
            self.validation_loss = self.calculate_validation_set_loss()


    @classmethod 
    def from_serialized(cls, serialized_model_location):
        with(open(serialized_model_location, "r")) as f:
            model_json = json.loads(f.read())
        generator = torch.Generator()
        generator.set_state(torch.ByteTensor(model_json["generator"]))
        return cls(W1=torch.tensor(model_json["W1"]), b1=torch.tensor(model_json["b1"]), W2=torch.tensor(model_json["W2"]), 
                   b2=torch.tensor(model_json["b2"]), C=torch.tensor(model_json["C"]), block_size=model_json["block_size"],
                   training_set=model_json["training_set_file"], embedding_size=model_json["embedding_size"],
                   hidden_layer_nb_of_neurons=model_json["hidden_layer_nb_of_neurons"],
                   number_of_training_steps=model_json["number_of_training_steps"],
                   learning_rate=model_json["learning_rate"], batch_size=model_json["batch_size"],
                   training_start=model_json["training_start"],training_end=model_json["training_end"],
                   validation_loss=model_json["validation_loss"], training_set_percentage=model_json["training_set_percentage"],
                   generator=generator, string_to_token_id=model_json["string_to_token_id"], token_id_to_string=model_json["token_id_to_string"],
                   training_losses=torch.tensor(model_json["training_losses"]))


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
        output_dict["C"] = self.C.tolist()
        output_dict["training_losses"] = list(map(lambda x: x.item(), self.training_losses))
        output_dict["validation_loss"] = self.validation_loss.item()
        output_dict["generator"] = list(map(lambda x: x.item(), self.generator.get_state()))
        output_dict["string_to_token_id"] = self.string_to_token_id
        output_dict["token_id_to_string"] = self.token_id_to_string
        with open(output_location, "w") as f:
            f.write(json.dumps(output_dict))


    def generate(self, tokens):
        token_ids = list(map(lambda token: self.string_to_token_id[token], tokens))
        embeddings = self.C[token_ids]
        h = torch.tanh(embeddings.view(1, embeddings.shape[0] * embeddings.shape[1]) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = logits / torch.sum(logits)
        max_value_index = torch.argmax(probs)
        return self.token_id_to_string[str(max_value_index.item())]
    
    def __train(self):
        number_of_tokens = len(self.string_to_token_id)
        # Embedding layer -> Matrix of size: number of tokens by the dimension of embeddings
        self.C = torch.randn((number_of_tokens, self.embedding_size), generator=self.generator)
        # Weights hidden layer 1
        self.W1 = torch.randn((self.embedding_size * self.block_size, self.hidden_layer_nb_of_neurons), generator=self.generator)
        # Bias hidden layer 1
        self.b1 = torch.randn((self.hidden_layer_nb_of_neurons), generator=self.generator)

        # Weights hidden layer 2
        self.W2 = torch.randn((self.hidden_layer_nb_of_neurons, number_of_tokens), generator=self.generator)
        # Bias hidden layer 2
        self.b2 = torch.randn(number_of_tokens, generator=self.generator)

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2] 

        for p in parameters:
            p.requires_grad = True

        for i in range(self.number_of_training_steps):
            batch_indexes = torch.randint(0, self.X_train.shape[0], (self.batch_size, ), generator=self.generator)
            probs = self.__forward_pass(self.X_train[batch_indexes], self.C, self.W1, self.b1, self.W2, self.b2)
            loss = self.calculate_loss(probs, self.Y_train[batch_indexes])
            self.training_losses.append(loss)
            self.__backward(parameters, loss, self.learning_rate)
    
    @torch.no_grad
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

class MultilayerNeuralNetworkProbabilisticLanguageModelV2(Model):

    '''
        Model being trained
        Changes compared with v1: have W2 near 0 b2 = 0 so initial loss is smaller. higher are W2 and b2, higher it is the
        initial loss. that's because some of the next chars will have very high probs, while others smaller. This will allow the 
        loss function to converge faster.
        Second problem:
        Values of activation of the hidden layer - Most values are -1 and 1
        - Input of activations are too high
        - Problem: Derivative will be close to 0. Gradient vanishing -> Algorithm will not learn.
        We will set W1 and b1 closer to 0
        We want NN to have similar activations - similar standard deviation thorughout the layers: Fixes issues of gradient vanishing or explosion
        Usage of Kaiming activation
        Gain in Kaiming is necessary because some activations are squashing the distribution

    '''
    def __init__(self, training_set, block_size, embedding_size, hidden_layer_nb_of_neurons, number_of_training_steps, \
                 learning_rate, batch_size, training_set_percentage, generator, \
                 W1=None, W2=None, b1=None, b2=None, C=None, training_start=None, training_end=None, validation_loss=None,
                 string_to_token_id=None, token_id_to_string=None, training_losses=None):
        
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.hidden_layer_nb_of_neurons = hidden_layer_nb_of_neurons
        
        self.generator = generator
        self.training_set_percentage = training_set_percentage
        self.validation_set_percentage = 1 - training_set_percentage
        
        self.number_of_training_steps = number_of_training_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        

        if W1 is not None and W2 is not None and b1 is not None and b2 is not None and C is not None:
            self.W1= W1
            self.W2 = W2
            self.b1 = b1
            self.b2 = b2
            self.C = C
            self.training_start = training_start
            self.training_end = training_end
            self.validation_loss = torch.tensor(validation_loss)
            self.string_to_token_id = string_to_token_id
            self.token_id_to_string = token_id_to_string
            self.training_losses = training_losses
            self.training_set = training_set
        else:
            training_set_file = files(nano_gpt).joinpath(training_set)
            with(open(training_set_file)) as ts:
                training_set_text = ts.read()
                words = training_set_text.split(" ")
            self.training_set = training_set_file.as_uri()
            self.string_to_token_id, self.token_id_to_string = tokenize_from_chars(words)
            self.X_train, self.Y_train, self.X_val, self.Y_val = create_training_set(words, self.block_size, self.string_to_token_id, self.training_set_percentage)
            self.training_losses = []
            self.training_start = datetime.now().isoformat()
            self.__train()
            self.training_end = datetime.now().isoformat()
            self.validation_loss = self.calculate_validation_set_loss()


    @classmethod 
    def from_serialized(cls, serialized_model_location):
        with(open(serialized_model_location, "r")) as f:
            model_json = json.loads(f.read())
        generator = torch.Generator()
        generator.set_state(torch.ByteTensor(model_json["generator"]))
        return cls(W1=torch.tensor(model_json["W1"]), b1=torch.tensor(model_json["b1"]), W2=torch.tensor(model_json["W2"]), 
                   b2=torch.tensor(model_json["b2"]), C=torch.tensor(model_json["C"]), block_size=model_json["block_size"],
                   training_set=model_json["training_set_file"], embedding_size=model_json["embedding_size"],
                   hidden_layer_nb_of_neurons=model_json["hidden_layer_nb_of_neurons"],
                   number_of_training_steps=model_json["number_of_training_steps"],
                   learning_rate=model_json["learning_rate"], batch_size=model_json["batch_size"],
                   training_start=model_json["training_start"],training_end=model_json["training_end"],
                   validation_loss=model_json["validation_loss"], training_set_percentage=model_json["training_set_percentage"],
                   generator=generator, string_to_token_id=model_json["string_to_token_id"], token_id_to_string=model_json["token_id_to_string"],
                   training_losses=torch.tensor(model_json["training_losses"]))


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
        output_dict["C"] = self.C.tolist()
        output_dict["training_losses"] = list(map(lambda x: x.item(), self.training_losses))
        output_dict["validation_loss"] = self.validation_loss.item()
        output_dict["generator"] = list(map(lambda x: x.item(), self.generator.get_state()))
        output_dict["string_to_token_id"] = self.string_to_token_id
        output_dict["token_id_to_string"] = self.token_id_to_string
        with open(output_location, "w") as f:
            f.write(json.dumps(output_dict))


    def generate(self, tokens):
        token_ids = list(map(lambda token: self.string_to_token_id[token], tokens))
        embeddings = self.C[token_ids]
        h = torch.tanh(embeddings.view(1, embeddings.shape[0] * embeddings.shape[1]) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = logits / torch.sum(logits)
        max_value_index = torch.argmax(probs)
        return self.token_id_to_string[str(max_value_index.item())]
    
    def __train(self):
        number_of_tokens = len(self.string_to_token_id)
        # Embedding layer -> Matrix of size: number of tokens by the dimension of embeddings
        self.C = torch.randn((number_of_tokens, self.embedding_size), generator=self.generator)
        # Weights hidden layer 1
        self.W1 = torch.randn((self.embedding_size * self.block_size, self.hidden_layer_nb_of_neurons), generator=self.generator) * 0.1
        # Bias hidden layer 1
        # Sets to a low value instead of 0 to get some entropy
        self.b1 = torch.randn((self.hidden_layer_nb_of_neurons), generator=self.generator) * 0.01

        # Weights hidden layer 2
        self.W2 = torch.randn((self.hidden_layer_nb_of_neurons, number_of_tokens), generator=self.generator) * 0.01
        # Bias hidden layer 2
        self.b2 = torch.randn(number_of_tokens, generator=self.generator) * 0

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2] 

        for p in parameters:
            p.requires_grad = True

        for i in range(self.number_of_training_steps):
            batch_indexes = torch.randint(0, self.X_train.shape[0], (self.batch_size, ), generator=self.generator)
            probs = self.__forward_pass(self.X_train[batch_indexes], self.C, self.W1, self.b1, self.W2, self.b2)
            loss = self.calculate_loss(probs, self.Y_train[batch_indexes])
            self.training_losses.append(loss)
            self.__backward(parameters, loss, self.learning_rate)
    
    @torch.no_grad
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



