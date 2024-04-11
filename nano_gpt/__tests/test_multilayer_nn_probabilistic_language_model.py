from importlib.resources import files
from nano_gpt.models import MultilayerNeuralNetworkProbabilisticLanguageModelBuilder, MultilayerNeuralNetworkProbabilisticLanguageModel

import nano_gpt
import json

def test_train():
    training_set_file = files(nano_gpt).joinpath('names.txt')
    model = MultilayerNeuralNetworkProbabilisticLanguageModelBuilder().with_batch_size(30) \
                .with_block_size(3).with_embedding_size(2).with_hidden_layer_nb_of_neurons(10) \
                .with_learning_rate(0.1).with_number_of_training_steps(1000).with_training_set(training_set_file) \
                .with_generator_seed(42) \
                .with_training_set_percentage(0.8).build_and_train()
    assert model.validation_loss.item() == 2.7444448471069336

def test_serialize():
    training_set_file = files(nano_gpt).joinpath('names.txt')
    model = MultilayerNeuralNetworkProbabilisticLanguageModelBuilder().with_batch_size(30) \
                .with_block_size(3).with_embedding_size(2).with_hidden_layer_nb_of_neurons(10) \
                .with_learning_rate(0.1).with_number_of_training_steps(1000).with_training_set(training_set_file) \
                .with_training_set_percentage(0.8).with_generator_seed(42).build_and_train()
    model.serialize("output.json")
    new_model = MultilayerNeuralNetworkProbabilisticLanguageModel.from_serialized("output.json")
    next_char = new_model.generate(["e", "m", "m"])
    assert(next_char=="a")
    new_model.serialize("output_bis.json")
    with open("output.json") as m1:
        with open("output_bis.json") as m2:
            assert json.loads(m1.read()) == json.loads(m2.read())


def test_generate():
    training_set_file = files(nano_gpt).joinpath('names.txt')
    model = MultilayerNeuralNetworkProbabilisticLanguageModelBuilder().with_batch_size(30) \
                .with_block_size(3).with_embedding_size(2).with_hidden_layer_nb_of_neurons(10) \
                .with_learning_rate(0.1).with_number_of_training_steps(1000).with_training_set(training_set_file) \
                .with_generator_seed(42) \
                .with_training_set_percentage(0.8).build_and_train()
    next_char = model.generate(["e", "m", "m"])
    assert(next_char=="a")
