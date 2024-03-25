
from nano_gpt.models import MultilayerNeuralNetworkProbabilisticLanguageModel
import plotly.express as px
import torch

df = px.data.iris()


'''
    It only supports 2D embeddings for now
'''
def get_display_embeddings_fig(model: MultilayerNeuralNetworkProbabilisticLanguageModel, index_to_character: torch.tensor) -> px.Figure:
    trained_embeddings = model.C
    first_dim = trained_embeddings[:, 0].numpy()
    second_dim = trained_embeddings[:, 1].numpy()
    fig = px.scatter(x=first_dim, y=second_dim, text=index_to_character)
    return fig

def get_loss_fig(model: MultilayerNeuralNetworkProbabilisticLanguageModel) -> px.Figure:
    losses = model.training_losses
     

    
