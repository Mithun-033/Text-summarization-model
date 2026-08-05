from torch.utils.data import Dataloader, Dataset

class data(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self,idx):
        ...

class dataloader(Dataset):
    def __init__(self):
        super().__init__()

    def train_dataloader(self):
        ...

    def val_dataloader(self):
        ...