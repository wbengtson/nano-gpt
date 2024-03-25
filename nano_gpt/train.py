import torch

class Train:

    def __init__(self, train_data):
        chars = self.get_chars(train_data)
        self.string_to_integer = {ch:i for ch, i in enumerate(chars)}
        self.integer_to_string = {i:ch for ch, i in enumerate(chars)}
        self.train_data = train_data
        # block size is the context size
        # we like to make transfomers to see context from size of 1 to block size
        self.block_size = 8
        self.batch_size = 4
        self.training_set, self.validation_set = self.create_train_validation(self.encode_train_data(self.train_data))

    def get_chars(self, value):
        return list(set(value))

    def encode(self, value: str) -> list:
        return list(map(lambda x: self.string_to_integer[x], value))
    
    def decode(self, encoded_list: list) -> str:
        return ''.join(list(map(lambda x: self.integer_to_string(x), encoded_list)))
    
    def encode_train_data(self):
        return torch.tensor(self.encode(self.train_data), dtype=torch.long)
    
    def create_train_validation(self, encoded_train_data, train_percent):
        size_training_set = int(train_percent * encoded_train_data)
        training_set = encoded_train_data[0:size_training_set]
        validation_set = encoded_train_data[size_training_set:]
        return training_set, validation_set

    def get_batch(self, split):
        data = self.training_set if split == "training" else self.validation_set
        index_x = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[x:x+self.block_size] for x in index_x])
        y = torch.stack([data[x+1:x+self.block_size+1] for x in index_x])
        return x, y



