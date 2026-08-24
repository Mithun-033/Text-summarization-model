from dataclasses import dataclass

@dataclass
class model_config:


    # dataset parameters
    block_size : int = 512
    batch_size: int = 8
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
