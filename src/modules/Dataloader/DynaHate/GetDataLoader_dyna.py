import pickle
import torch
from torch.utils.data import DataLoader
# Adjust these imports to your actual paths/names
from ._collate import collate_fn_dynahate
from ._dataset import dynahate_dataset


# Assuming you already have these in your codebase:
# from ._collate import collate_fn_dynahate
# from ._dataset import dynahate_dataset

def get_dataloader_dynahate(
    train_batch_size=32,
    eval_batch_size=32,
    base_data_path = "./dataset/preprocessed",
):
    """
    Builds and returns train, valid, and test dataloaders for the DynaHate dataset,
    without augmentation (w_aug=False), following the style of the reference code.

    :param train_batch_size: Batch size for training loader
    :param eval_batch_size:  Batch size for validation/test loader
    :param data_file:        Path to the 'dynahate_preprocessed_bert.pkl' file
    :return:                 (train_loader, valid_loader, test_loader)
    """
    file_path = f"{base_data_path}/dynahate_preprocessed_bert.pkl"

    # --- 1. Load the preprocessed pickle file ---
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # --- 2. Create Dataset objects ---
    #     The reference sets w_aug=False for DynaHate (and asserts that we do not consider w_aug).
    train_dataset = dynahate_dataset(data["train"], training=True,  w_aug=False)
    valid_dataset = dynahate_dataset(data["dev"],   training=False, w_aug=False)
    test_dataset  = dynahate_dataset(data["test"],  training=False, w_aug=False)
    # --- 3. Select the collate function (same for all splits in DynaHate) ---
    collate_fn = collate_fn_dynahate
    # --- 4. Build DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --- 5. Return the loaders ---
    return train_loader, valid_loader, test_loader
