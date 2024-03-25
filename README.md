# nano-gpt

Project to learn how to create language models from scratch.

Based on Andrej's Karpathy course: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

## Model architectures:

nano_gpt/bigram.py Bigram model
nano_gpt/models.py#MultilayerNeuralNetworkProbabilisticLanguageModel Model based on https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

## How to run the unit test
1/ Run poetry shell
2/ Run poetry install
3/ Run poetry run pytest
