import torch 
import torchinfo
import torch.nn as nn
from config import model_config
from torch.nn import functional as F
import math
import torch.nn.functional as F
from torchinfo import summary

# HEREIN THIS FILE THE WHOLE ARCHITECTURE OF TRANSFORMERS AND ATTENTION IS DEFINED MANUALLY FOR BETTER UNDERDSTANDING OF THE FLOW OF 
# MATHS AND MATRICES

# Although it can be implemented by PyTorch F.scaled_dot_product_attention(q,k,v)
class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # freq formula = 1 / (10_000) ** (2i / dh),  base_freq = 10k, dh = head_size
        self.inv_freqs = 1 / (config.base_freq ** (torch.arange(0,config.head_size, 2) / config.head_size)) # (dh // 2)

    def forward(self,q,k):
        _,T,_,H = q.shape

        time_steps = torch.arange(0,T)
        freqs = time_steps.unsqueeze(dim = 1) @ self.inv_freqs.unsqueeze(dim = 0) # (T,1) @ (1,dh//2) = (T,dh//2), basically outer_product
        # this is done to get varying freqs for each time step, for each pair of head dims.

        cos = torch.cos(freqs).reshape(1,T,1,H//2) # broadcasting to all batches and all heads
        sin = torch.sin(freqs).reshape(1,T,1,H//2)

        # get the even and odd head_size dims for all heads at every time step
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        # rotate according to the 2d rotation matrix
        q_out = torch.empty_like(q)
        q_out[..., 0::2] = q_even * cos - q_odd * sin
        q_out[..., 1::2] = q_even * sin + q_odd * cos

        k_out = torch.empty_like(k)
        k_out[..., 0::2] = k_even * cos - k_odd * sin
        k_out[..., 1::2] = k_even * sin + k_odd * cos

        return q_out, k_out


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.l1 = nn.LayerNorm(config.n_emb)
    
        self.attn = AttentionBlock(config)

        self.l2 = nn.LayerNorm(config.n_emb)

        self.fnn = FeedForward(config)

    def forward(self , x):
        x = x + self.attn(self.l1(x))
        x = x + self.fnn(self.l2(x))
        return x

class AttentionBlock(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_emb%config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_emb = config.n_emb

        # Now, defining the qkv matrix in a single large matrix
        self.attn = nn.Linear(config.n_emb , 3*config.n_emb ,bias=config.bias)

        #Final Layers for getting the outputs 
        self.c_proj = nn.Linear(config.n_emb , config.n_emb , bias =config.bias )

        # REGULARIZATION 
        self.drop = nn.Dropout(config.dropout)

        # THE CAUSAL MASK
        # Creates a (512, 512) grid of 1s and 0s. 
        # register_buffer means PyTorch saves it, but the optimizer doesn't train it.
        self.rope = RoPE(config)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )


    def forward(self,x):
        # B = 4 ( Batch Size) , T = 512 ( block size ) , C = 384
        B,T,C = x.size()

        qkv = self.attn(x) #Here in this step, out input matrix is multilplied with the weights in nn.Linear layers attn to get 
                           # a whole matrix of qkv 


        # now splitting the qkv matrix into three different q , k, v matrices
        q,k,v = qkv.split(self.n_emb ,dim = 2) # each matrix has a dim of (4,512,384)

        # NOW DOING THE MULTIHEAD SLICING WITH HEADS = 6
        # HERE TRANSPOSE(1,2) IS USED TO SWAP 6 AND 512 SO THAT PYTORCH CAN PROCESS THOSE SPLITTED MATRIX PARALLEY
        # FINAL SHAPE OF q,k,v AFTERD DOING THIS WILL BE (4,6,512,64)

        q = q.view(B, T ,self.n_heads , C//self.n_heads).transpose(1,2)
        k = k.view(B, T ,self.n_heads , C//self.n_heads).transpose(1,2)
        v = v.view(B, T ,self.n_heads , C//self.n_heads).transpose(1,2)

        q,k = self.rope(q,k)
        # NOW APPLYING THE ACTUAL TRANSFORMER MATH
        att = (q @ k.transpose(-2,-1)) * (1/(math.sqrt(k.size(-1))))

        # MASKING 
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # APPLYING SOFTMAX
        att = F.softmax(att , dim=-1)
        att = self.drop(att)
        # Multiplying with the VALUE(V) Matrix 
        y = att@v

        # NOW REASSEMBLE THE HEADS
        # SWAP 6 and 512 by transpose and then combine those splited matrix 6 and 64 back into 384
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # NOW MULTIPLYING WITH THE FINAL WEIGHT MATRIX c_proj
        y = self.c_proj(y)
        y = self.drop(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.feedNN = nn.Sequential(

            nn.Linear(config.n_emb , config.n_emb*4 , bias=config.bias),
            nn.GELU(),

            nn.Dropout(config.dropout),
            nn.Linear(config.n_emb*4 , config.n_emb , bias=config.bias)
        )
    
    def forward(self , x):
        return self.feedNN(x)




class SummarizationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        #wte :- Word Token Embeddings, (this matrix basically contains the token vectors)
        self.wte = nn.Embedding(config.vocab_size, config.n_emb)

        #wpe :- Word Postional Embeddings, (this matrix contians the information about the position of a word in a sentence, not about the word)
        self.wpe = nn.Embedding(config.block_size,config.n_emb)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        #Final normalization and output projection head
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.f_head = nn.Linear(config.n_emb , config.vocab_size , bias=False)

    def forward(self,x):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B,T = x.size() #(4,512)

        # extracting the token embeddings from wte
        embeddings = self.wte(x) #(4,512,384)

        #passing through each block
        for block in self.blocks:
            embeddings = block(embeddings)

        # Final Normalizing
        embeddings = self.ln_f(embeddings)

        # Final layer to get the probability distribution over the vocab size
        logits = self.f_head(embeddings)

        return logits


if __name__ =="__main__":
    config = model_config()
    model = SummarizationModel(config)
    summary(model)