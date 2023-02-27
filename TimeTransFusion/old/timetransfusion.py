import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
import random

from typing import List, Optional

# TODO:
# 1. Add pre-processing of data
# 2. Randomize order in batch?
# 3. Make a custom decoder where I can feed a matrix containing both targets and their noisy version?
# 4. Change positional encoding to use timestamp information
# 5. Make sure I am doing .to(device) everywhere necessary
# 6. Need to feed "n" the diffusion step to the network
# 7. Maybe add some encoding network after positional encoding (add timeseries number and whether value is observed or diffused)

class TTFModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, encoding_length: int, dropout: float = 0.5):
        super().__init__()

        self.model_type = 'Transformer'

        self.positional_encoder = PositionalEncoding(
            encoding_length=encoding_length
        )

        self.transformer = nn.Transformer(
            d_model=d_model + encoding_length,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=d_hid,
            dropout=dropout
        )

        self.linear = nn.Linear(d_model + encoding_length, 1)

        # self.model_type = 'Transformer'
        # self.d_model = d_model
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        # encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        # self.linear = nn.Linear(d_model, 1)
        # TODO: Find out what encoder should look like for time series
        #self.encoder = nn.Embedding(ntoken, d_model)


    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # TODO: Check if I need explicitly set device for function beneath
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0])
        
        # Redo these positional encoders
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # TODO: Think the mask used here is wrong as we need -inf on diagonal
        output = self.transformer(src,tgt,tgt_mask=tgt_mask)
        output = self.linear(output)
        
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, encoding_length: int, max_len: Optional[int] = 5000):
        super().__init__()

        assert encoding_length % 2 == 0, "Length of positional encoding needs to be a multiple of 2"

        position = torch.arange(max_len).unsqueeze(1)
        div_term = 2**torch.arange(1, encoding_length / 2 + 1, 1,dtype=torch.float)
        pe = torch.zeros(max_len, 1, encoding_length)
        pe[:, 0, 0::2] = torch.sin(math.pi*position / div_term)
        pe[:, 0, 1::2] = torch.cos(math.pi*position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        return torch.cat((x,self.pe[:x.size(0),:].expand(-1,x.shape[1],-1)),dim=2)



def create_inout_sequences(data: Tensor, context_length: int, prediction_length: int) -> List[Tuple[Tensor,Tensor]]:

    inout_seq = []
    data_length = len(data)

    for i in range(context_length, data_length - prediction_length):
        context = data[i - context_length:i]
        target = data[i:i+prediction_length]
        inout_seq.append((context,target))

    return inout_seq


# def batchify(data: List[Tuple[Tensor,Tensor]], bsz: int, device: torch.device) -> Tensor:
#     """Divides the data into bsz separate sequences, removing extra elements
#     that wouldn't cleanly fit.

#     Args:
#         data: List[Tuple[Tensor,Tensor]], list of N context, target pairs
#         bsz: int, batch size

#     Returns:
#         Tensor of shape [N // bsz, bsz]
#     """
#     seq_len = data.size(0) // bsz
#     data = data[:seq_len * bsz]
#     data = data.view(bsz, seq_len).t().contiguous()
#     return data.to(device)

# def get_batch(data: List[Tuple[Tensor,Tensor]], i: int, batch_size: int):

#     assert i < len(data) // batch_size , f"Requested batch number {i}, but only {len(data) // batch_size} available"

#     context = torch.transpose(torch.stack([element[0] for element in data[i*batch_size:(i+1)*batch_size]]),0,1).unsqueeze(2)
#     target = torch.transpose(torch.stack([element[1] for element in data[i*batch_size:(i+1)*batch_size]]),0,1).unsqueeze(2)

#     return context, target

def get_batch(data: List[Tuple[Tensor,Tensor]], batch_size: int):

    batch_data = random.choices(data,k=batch_size)

    context = torch.transpose(torch.stack([element[0] for element in batch_data]),0,1).unsqueeze(2)
    target = torch.transpose(torch.stack([element[1] for element in batch_data]),0,1).unsqueeze(2)

    return context, target





