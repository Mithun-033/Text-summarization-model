import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self,path,block_size):
        super().__init__()
        self.path = path
        self.block_size = block_size

        # now as every shard_file has different number of tokens, this creates a risk of index error
        self.blocks_per_shard = []

        for p in self.path:
            file = np.load(p,mmap_mode='r')
            num_blocks = (len(file)-1)//block_size
            self.blocks_per_shard.append(num_blocks)
        self.cumsum = np.cumsum(self.blocks_per_shard)

    def __len__(self):
        return int(self.cumsum[-1])

    
    def __getitem__(self,index):
        idx = 0
        for i in range(len(self.cumsum)):
            if self.cumsum[i] > index:
                idx = i
                break
        
        local_idx = 0
        if idx ==0 :
            local_idx = index
        else:
            local_idx = index - int(self.cumsum[idx-1])

        shard = np.load(self.path[idx] , mmap_mode='r')# this mmap_mode = 'r' do not loads the whole file into ram, just the sliced array
        start = local_idx*self.block_size

        x = shard[start : start + self.block_size]
        y = shard[start+1 : start+self.block_size+1]

        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    
class get_dataloader:
    def __init__(self,train_path:list ,val_path :list ,block_size :int, batch_size :int , num_workers:int):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        
    def train_dataloader(self):
        dataset = CustomDataset(self.train_path,self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        dataset = CustomDataset(self.val_path,self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False
        )


# for testing purpose only
if __name__ =="__main__":
    train_path = ['corpus/shard_0.npy']
    val_path = ['corpus/shard_1.npy']

    ds = get_dataloader(train_path,val_path,block_size=512,batch_size=8,num_workers=2)

    train_loader = ds.train_dataloader()

    for i in range(1):
        sample_x,sample_y = next(iter(train_loader))

        print(sample_x.shape)
        print(sample_y.shape)