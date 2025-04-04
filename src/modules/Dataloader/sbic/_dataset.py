import torch 
from torch.utils.data import Dataset 

class sbic_dataset(Dataset):
    def __init__(self,data,training=True,w_aug=False):
        self.data = data
        self.training = training
        self.w_aug = w_aug
    def __getitem__(self, index):
        item = {}
        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]
        return item
    def __len__(self):
        return len(self.data["label"])
    
    