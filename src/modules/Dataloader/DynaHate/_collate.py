from torch import nn
import torch
def collate_fn_dynahate(data):
    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]
        if N == None: 
            N = 128
        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq) # Toxigen 오류 방지용 (int64)
            # end = lengths[i]
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
