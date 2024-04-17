
from nano_gpt.models import MultilayerNeuralNetworkProbabilisticLanguageModel
import plotly.express as px


'''
    It only supports 2D embeddings for now
'''
def get_display_embeddings_fig(model: MultilayerNeuralNetworkProbabilisticLanguageModel):
    trained_embeddings = model.C.detach()
    first_dim = trained_embeddings[:, 0]
    second_dim = trained_embeddings[:, 1]
    fig = px.scatter(x=first_dim, y=second_dim, text=model.token_id_to_string, size_max=20)
    fig.update_traces(marker={'size': 20, 'color': 'yellow'})
    return fig

def get_loss_fig(model: MultilayerNeuralNetworkProbabilisticLanguageModel):
    losses = list(map(lambda x: x.item(), model.training_losses))
    fig = px.line(y=losses, x=range(1, len(losses) + 1))
    return fig
     

    
