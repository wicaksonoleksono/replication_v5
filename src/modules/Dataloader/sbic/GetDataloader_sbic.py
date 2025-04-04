from ._collate import collate_fn_sbic,collate_fn_w_aug_sbic_imp_con
from ._dataset import sbic_dataset
import torch 
import pickle 
def get_dataloader_sbic(train_batch_size,
                   eval_batch_size,
                   dataset='sbic',
                   seed=None,
                   w_aug="imp",
                   base_data_path = "./dataset/preprocessed"
):
    if w_aug in ["imp", "aug"]:
        file_path = f'{base_data_path}/sbic_{w_aug}_preprocessed_bert.pkl'
    else:
        file_path = f'{base_data_path}/sbic_preprocessed_bert.pkl'
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if "sbic" in dataset:
        train_dataset = sbic_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = sbic_dataset(data["dev"],training=False,w_aug=w_aug)
        test_dataset = sbic_dataset(data["test"],training=False,w_aug=w_aug)
        
    else:
        raise NotImplementedError
    
    # Create DataLoaders
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, 
        collate_fn=collate_fn_w_aug_sbic_imp_con, num_workers=0
    )
    valid_iter = torch.utils.data.DataLoader(
        valid_dataset, batch_size=eval_batch_size, shuffle=False, 
        collate_fn=collate_fn_sbic, num_workers=0
    )
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, 
        collate_fn=collate_fn_sbic, num_workers=0
    )
    return train_iter, valid_iter, test_iter