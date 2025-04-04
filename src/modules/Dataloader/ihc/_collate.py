import torch
import itertools

def collate_fn_w_aug(data):
    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]
        if N == None: 
            N = 128
        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()
        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()
        return padded_seqs, attention_mask,lengths
    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
        flat = itertools.chain.from_iterable(item_info[key])
        original_posts = []
        augmented_posts = []
        for i, one_post in enumerate(flat):
            if i % 2 == 0:
                original_posts.append(one_post)
            else:
                augmented_posts.append(one_post)
        original_n_augmented_posts = original_posts + augmented_posts

        item_info[key] = original_n_augmented_posts
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])
    d={}
    d["label"] = item_info["label"]
    d["post"] = post_batch
    d["post_attn_mask"] = post_attn_mask
    return d


# Credits https://github.com/varsha33/LCL_loss
def collate_fn(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}
    d["label"] = item_info["label"]
    # d["post"] = post_batch
    d["post"] = post_batch.cuda()
    # d["post_attn_mask"] = post_attn_mask
    d["post_attn_mask"] = post_attn_mask.cuda()



    return d
