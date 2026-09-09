from dataclasses import dataclass
@dataclass
class model_config:
    
    vocab_size :int = 50_432
    base_freq : int = 10_000
    n_emb:int = 384
    n_heads: int = 6 #Number of Heads for Multi Head Attention
    head_size : int = n_emb // n_heads
    bias : bool = False
    dropout : float = 0.3
    n_layers : int = 6

    
    block_size : int = 512
    batch_size: int = 4
    num_workers: int = 2


@dataclass
class train_config:
    ...

@dataclass
class data_config:
    num_workers : int = ...
    pin_memory : bool = ...
    persistent_workers : bool = ...
    prefetch_factor : int = ...
    in_order : bool = ...
