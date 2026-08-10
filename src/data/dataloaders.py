from torch.utils.data import DataLoader, Dataset
from src.model.config import data_config

class data(Dataset):
    def __init__(self, path):
        super().__init__()

    def __len__(self): ...

    def __getitem__(self, idx): ...


class dataloader(Dataset):
    def __init__(self, data_config, train_path, val_path):
        super().__init__()
        assert train_path is not None, "train_path cannot be None"
        assert val_path is not None, "val_path cannot be None"
        self.train = data(train_path)
        self.val = data(val_path)

        assert data_config is not None, "data_config cannot be None"
        self.data_config = data_config

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor,
            pin_memory=self.data_config.pin_memory,
            in_order=self.data_config.in_order,
            shuffle = True
        )

    def val_dataloader(self): 
        return DataLoader(
            self.val,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor,
            pin_memory=self.data_config.pin_memory,
            in_order=self.data_config.in_order,
            shuffle = False
        )
