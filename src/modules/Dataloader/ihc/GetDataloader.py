from ._collate import collate_fn_w_aug,collate_fn
from ._dataset import data_pointer  # Ensure this is your Dataset class
import torch 
import pickle 

def get_dataloader(train_batch_size,
                   eval_batch_size,
                   dataset='ihc',  # Default to string 'ihc'
                   seed=None,
                   w_aug="imp",
                   base_data_path = "./dataset/preprocessed"
                   ):
    if w_aug in ["imp", "aug"]:
        file_path = f'{base_data_path}/ihc_{w_aug}_preprocessed_bert.pkl'
    else:
        file_path = f'{base_data_path}/ihc_preprocessed_bert.pkl'
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if "ihc" in dataset:
        # Use the data_pointer class correctly to create datasets
        train_dataset = data_pointer(data["train"], training=True, w_aug=w_aug)
        valid_dataset = data_pointer(data["valid"], training=False, w_aug=w_aug)
        test_dataset = data_pointer(data["test"], training=False, w_aug=w_aug)

    else:
        raise NotImplementedError
    # Create DataLoaders
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, 
        collate_fn=collate_fn_w_aug, num_workers=0
    )
    valid_iter = torch.utils.data.DataLoader(
        valid_dataset, batch_size=eval_batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=0
    )
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=0
    )
    
    return train_iter, valid_iter, test_iter