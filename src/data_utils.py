from joblib import Parallel, delayed
import torch
import os
from math import ceil
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score


def senti_prompt(doc, label_id, tokenizer, max_len = 512):
    left_ids = tokenizer(doc + " It was", truncation=True, max_length=max_len-10, add_special_tokens=False)['input_ids']
    right_ids = tokenizer(".", truncation=True, max_length=500, add_special_tokens=False)['input_ids']
    ids = [tokenizer.cls_token_id] + left_ids + label_id + right_ids + [tokenizer.sep_token_id]
    to_pad = max_len - len(ids)
    return ids, ids + [0]*to_pad, [1]*len(ids) + [0]*to_pad, \
            (len(left_ids) + 1, len(left_ids) + 1 + len(label_id))

def topic_prompt(doc, label_id, tokenizer, max_len = 512, masked=False):
    doc_ids = tokenizer('News: ' + doc, truncation=True, max_length=max_len-10, add_special_tokens=False)['input_ids']
    ids = [tokenizer.cls_token_id] + label_id + doc_ids + [tokenizer.sep_token_id]
    to_pad = max_len - len(ids)
    return ids, ids + [0]*to_pad, [1]*len(ids) + [0]*to_pad, (1, 1 + len(label_id))


def acc(pred, gt):
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def encode(docs, tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, 
                                               max_length=max_len, padding='max_length',
                                                return_attention_mask=True, truncation=True, 
                                               return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


def create_dataset(dataset_dir, text_file, loader_name, tokenizer, label_file=None, max_len=512, num_cpus=20):
    loader_file = os.path.join(dataset_dir, loader_name)
    if os.path.exists(loader_file):
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)
    else:
        print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
        corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
        docs = [doc.strip() for doc in corpus.readlines()]
        corpus.close()
        print(f"Converting texts into tensors.")
        chunk_size = ceil(len(docs) / num_cpus)
        chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
        results = Parallel(n_jobs=num_cpus)(delayed(encode)(docs=chunk, tokenizer=tokenizer, max_len=max_len) for chunk in chunks)
        input_ids = torch.cat([result[0] for result in results])
        attention_masks = torch.cat([result[1] for result in results])
        if label_file is not None:
            print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
            truth = open(os.path.join(dataset_dir, label_file))
            labels = [int(label.strip()) for label in truth.readlines()]
            labels = torch.tensor(labels)
            data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
        else:
            data = {"input_ids": input_ids, "attention_masks": attention_masks}
        print(f"Saving encoded texts into {loader_file}")
        torch.save(data, loader_file)
    return data


def create_prompt_dataset(dataset_dir, text_file, id2labels_id, tokenizer, prompt, loader_name, max_len=512):
    loader_file = os.path.join(dataset_dir, loader_name)
    if os.path.exists(loader_file):
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)
    else:
        with open(os.path.join(dataset_dir, text_file), encoding="utf-8") as f:
            docs = [doc.strip() for doc in f]
        input_ids = []
        attension_mask = []
        positions = []
        labels = []
        doc_ids = []
        for di, doc in tqdm(enumerate(docs), total=len(docs)):
            for li, l in enumerate(id2labels_id):
                ids, padded_ids, att, pos = prompt(doc, l, tokenizer, max_len)
                input_ids.append(padded_ids)
                attension_mask.append(att)
                positions.append(pos)
                doc_ids.append(di)
                labels.append(li)
        input_ids = torch.tensor(input_ids)
        attension_mask = torch.tensor(attension_mask)
        positions = torch.LongTensor(positions)
        labels = torch.tensor(labels)
        doc_ids = torch.LongTensor(doc_ids)
        data = {"input_ids": input_ids, "attention_masks": attension_mask, 
                'positions': positions, "labels": labels, "doc_ids": doc_ids}
        print(f"Saving encoded texts into {loader_file}")
        torch.save(data, loader_file)
    return data


def load_label_names(dataset_dir, label_names, tokenizer):
    id2labels = []
    id2labels_id = []
    with open(os.path.join(dataset_dir, label_names)) as f:
        for line in f:
            id2labels.append(line.strip())
            id2labels_id.append(tokenizer(line.strip(), add_special_tokens=False)['input_ids'])
    return id2labels, id2labels_id


def make_dataloader(data_dict, batch_size, shuffle=False):
    if "labels" in data_dict:
        dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
    else:
        dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader


def make_cls_dataloader(data_dict, partial_pred, partial_ids, batch_size=32):
    dataset = TensorDataset(data_dict["input_ids"][partial_ids], 
                            data_dict["attention_masks"][partial_ids], 
                            torch.tensor(np.array(partial_pred)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_prompt_dataloader(data_dict, partial_pred, partial_ids, num_label, batch_size=32):
    input_ids = []
    attension_masks = []
    positions = []
    labels = []
    id2pred = {}
    for doc_id, pred in zip(partial_ids, partial_pred):
        id2pred[doc_id] = pred
    selected_ids = []
    for i in range(len(data_dict["doc_ids"])):
        doc_id = data_dict["doc_ids"][i].long().item()
        if doc_id in id2pred:
            if data_dict["labels"][i].long().item() == id2pred[doc_id]:
                selected_ids.append(i)
                labels.append(0)
            else:
                selected_ids.append(i)
                labels.append(1)
    dataset = TensorDataset(data_dict["input_ids"][selected_ids], 
                            data_dict["attention_masks"][selected_ids], 
                            data_dict["positions"][selected_ids],
                            torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
