import torch

class Config:
    def __init__(self, *, path_to_dataset, path_to_embeddings, batch_size, window_size=2, message_passing_layers=2,
                 hidden=64, penultimate=64, dropout=0.5, lr=0.001, epochs=200, patience=20, directed=True,
                 normalize=True, use_master_node=True, weights=False):
        self.no_cuda = False
        self.path_to_dataset = path_to_dataset
        self.path_to_embeddings = path_to_embeddings
        self.window_size = window_size
        self.directed = directed
        self.normalize = normalize
        self.use_master_node = use_master_node
        self.batch_size = batch_size
        self.message_passing_layers = message_passing_layers
        self.hidden = hidden
        self.penultimate = penultimate
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.cuda = not self.no_cuda and torch.cuda.is_available()

        self.weights = weights
