import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_uniform_


class PositionalEncoding(nn.Module):
    """
    2D positional encoding if 4 dimentional input (multiple demos)
    1D positional encoding if 3 dimensional input (agent history)
    """
    def __init__(self, d_model, dropout=0, max_len=200): # dropout of 0
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # 1D positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TrajectoryEncoder(nn.Module):
    def __init__(self, obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers):
        super(TrajectoryEncoder, self).__init__()

        # Positional encoder
        self.positional_encoder = PositionalEncoding(d_model, max_len=demo_max_length)

        # State encoder
        self.obs_encoder = nn.Linear(obs_shape[0], d_model)

        # Encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._reset_parameters()


    def forward(self, traj, traj_mask):
        """
        traj dim: Batch * Nstep * Obsdim
        traj_mask: Batch * Nstep (True if the timestep is masked, False otherwise)
        """
        # Initial linear proj
        x = self.obs_encoder(traj)

        # Transformer require batch dim second
        x = x.transpose(0, 1)

        # Positional encoding demo states
        x = self.positional_encoder(x)

        # Transformer encoding
        x_seq_embedding = self.encoder(x, mask=None, src_key_padding_mask=traj_mask)

        # Replace batch in the first dimention
        x_seq_embedding = x_seq_embedding.transpose(0, 1)

        # Average on Nstep dimension (without taking masked pos into account)
        x_seq_embedding[traj_mask] = 0
        x_embedding = torch.sum(x_seq_embedding, dim=1) / (~traj_mask).sum(dim=-1, keepdim=True)

        # x_seq_embedding[traj_mask] = -float('Inf')
        # x_embedding = torch.max(x_seq_embedding, dim=1)[0]

        return x_embedding

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TrajectoryEncoder2(nn.Module):
    def __init__(self, obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers):
        super(TrajectoryEncoder2, self).__init__()

        # Positional encoder
        self.positional_encoder = PositionalEncoding(d_model, max_len=demo_max_length)

        # State encoder
        self.obs_encoder = nn.Linear(obs_shape[0], d_model)

        # Encoder 
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_layers)])
        self.encoder_layers2 = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_layers)])

        self.pooling_layer = nn.Linear(5 * d_model, d_model)

        self._reset_parameters()


    def forward(self, traj, traj_mask):
        """
        traj dim: Batch * Nstep * Obsdim
        traj_mask: Batch * Nstep (True if the timestep is masked, False otherwise)
        """
        # Initial linear proj
        x = self.obs_encoder(traj)

        # Transformer require batch dim second
        x = x.transpose(0, 1)

        # Positional encoding demo states
        x_seq_embedding = self.positional_encoder(x)
 
        # Transformer encoding
        for i, encoder_layer in enumerate(self.encoder_layers):
            x_seq_embedding = encoder_layer(x_seq_embedding, src_mask=None, src_key_padding_mask=traj_mask)

            # if i  == 0:
            #     # print(x_seq_embedding.shape)
            #     x_seq_embedding = x_seq_embedding.transpose(0, 1)
            #     x_seq_embedding = x_seq_embedding.reshape(x_seq_embedding.shape[0], 25, 2 * x_seq_embedding.shape[-1])
            #     x_seq_embedding = self.pooling_layer(x_seq_embedding)
            #     x_seq_embedding = x_seq_embedding.transpose(0, 1)
            #     # print(x_seq_embedding.shape)
            #     # print(traj_mask.shape)
            #     traj_mask = traj_mask.view(traj_mask.shape[0], 25, 2)
            #     traj_mask = traj_mask.all(dim=-1)
                # print(traj_mask.shape)


        # print(x_seq_embedding.shape)
        x_seq_embedding = x_seq_embedding.transpose(0, 1)
        x_seq_embedding = x_seq_embedding.reshape(x_seq_embedding.shape[0], 10, 5 * x_seq_embedding.shape[-1])
        x_seq_embedding = self.pooling_layer(x_seq_embedding)
        x_seq_embedding = x_seq_embedding.transpose(0, 1)
        # print(x_seq_embedding.shape)
        # print(traj_mask.shape)
        # print(traj_mask[:10])
        traj_mask = traj_mask.view(traj_mask.shape[0], 10, 5)
        # print(traj_mask[:10])
        traj_mask = traj_mask.all(dim=-1)
        # print(traj_mask[:10])
        # input()
        # print(traj_mask.shape)

        # for i, encoder_layer in enumerate(self.encoder_layers2):
        #     x_seq_embedding = encoder_layer(x_seq_embedding, src_mask=None, src_key_padding_mask=traj_mask)

        # Replace batch in the first dimention
        x_seq_embedding = x_seq_embedding.transpose(0, 1)

        # Average on Nstep dimension (without taking masked pos into account)
        x_seq_embedding[traj_mask] = 0
        x_embedding = torch.sum(x_seq_embedding, dim=1) / (~traj_mask).sum(dim=-1, keepdim=True)

        return x_embedding



    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TrajectoryDecoder(nn.Module):
    def __init__(self, obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers):
        super(TrajectoryDecoder, self).__init__()

        # Positional encoder
        self.positional_encoder = PositionalEncoding(d_model, max_len=demo_max_length)

        # State encoder
        self.obs_decoder = nn.Linear(d_model, obs_shape[0])

        # Encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.demo_max_length = demo_max_length

        self._reset_parameters()


    def forward(self, embedding, traj_mask):
        """
        traj dim: Batch * Nstep * Obsdim
        traj_mask: Batch * Nstep (True if the timestep is masked, False otherwise)
        """
        # State encoder
        x = embedding.unsqueeze(1).expand(-1, traj_mask.shape[1], -1)

        # Transformer require batch dim second
        x = x.transpose(0, 1)

        # Positional encoding of the embedding
        x = self.positional_encoder(x)

        # Transformer encoding
        x_seq_embedding = self.encoder(x, mask=None, src_key_padding_mask=traj_mask)

        # Replace batch in the first dimention
        x_seq_embedding = x_seq_embedding.transpose(0, 1)

        # Final linear proj
        x_seq = self.obs_decoder(x_seq_embedding)

        return x_seq

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

if __name__ == '__main__':
    obs_shape = (3,)
    d_model = 8
    nhead = 4
    dim_feedforward = 64
    dropout = 0
    activation = 'relu'
    demo_max_length = 50
    batch_size = 64
    num_layers = 3

    fake_traj = torch.zeros((batch_size, demo_max_length, obs_shape[0]))
    fake_traj_mask = torch.zeros((batch_size, demo_max_length), dtype=torch.bool)

    encoder = TrajectoryEncoder(obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers)
    decoder = TrajectoryDecoder(obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers)

    out = encoder(fake_traj, fake_traj_mask)
    # print(out)
    print(fake_traj.shape)
    print(out.shape)

    out = decoder(out, fake_traj_mask)
    print(out.shape)


"""
Simple three layers MLP for policy and value heads
"""

class ActorMLP(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorMLP, self).__init__()
        
        self.actor = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, action_dim)) 

        self.output_activation = nn.Tanh()


    def forward(self, obs):
        return self.output_activation(self.actor(obs))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)