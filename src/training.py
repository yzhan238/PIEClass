from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter, defaultdict
from data_utils import *


def get_pseudo_label(pred_scores, pred_conf, num_class, top_k, thres=0.9, labels=None, imbalanced=False):
    partial_label = []
    partial_hard = []
    partial_ids = []
    pred = np.argmax(pred_scores, axis=1)
    count = {i:0 for i in range(num_class)}
    for i in np.argsort(-np.array(pred_conf)):
        if not imbalanced and pred_conf[i] < thres: break
        count[pred[i]] += 1
    if imbalanced:
        top_each = [int(count[l] * top_k / len(pred_conf)) + 1 for l in range(num_class)]
    else:
        top_each = min(c for c in count.values())
        top_each = [min(top_each, top_k // num_class)] * num_class
    num_added = [0] * num_class
    for i in np.argsort(-np.array(pred_conf)):
        if pred_conf[i] < thres: break
        pred = np.argmax(pred_scores[i])
        if num_added[pred] < top_each[pred]:
            partial_ids.append(i)
            partial_hard.append(np.argmax(pred_scores[i]))
            if labels:
                partial_label.append(labels[i])
            num_added[pred] += 1
    print(f'total {len(partial_ids)}: {num_added}')
    if labels:
        print(acc(partial_hard, partial_label))
    
    return np.array(partial_hard), partial_ids


def random_sample(ids, hard, ratio=0.1):
    num_labels = len(set(hard))
    ret = [[], []]
    for i in range(num_labels):
        i_ids = [j for j,h in enumerate(hard) if h==i]
        indices = np.random.choice(i_ids, size=int(len(i_ids)*ratio)+1, replace=False)
        ret[0].extend([ids[j] for j in indices])
        ret[1].extend([hard[j] for j in indices])
    return ret


def up_sample(pred_hard, pred_ids):
    split = defaultdict(list)
    for i, l in enumerate(pred_hard):
        split[l].append(i)
    max_len = max([len(ids) for ids in split.values()])
    new_ids = []
    for ids in split.values():
        copy = max_len // len(ids)
        for _ in range(copy):
            new_ids.extend(ids)
        new_ids.extend(np.random.choice(ids, size=max_len%len(ids), replace=False).tolist())
    new_hard = [pred_hard[i] for i in new_ids]
    new_ids = [pred_ids[i] for i in new_ids]
    return new_hard, new_ids


def get_cls(model, test_loader, gpu, return_pred=False):
    print('classifier inference')
    class_scores = []
    class_conf = []
    class_pred = []
    labels = []
    for j, batch in enumerate(tqdm(test_loader)):
        if j == 0:
            with_label = len(batch) == 3
        batch_max_len = batch[1].sum(dim=1).max()
        input_ids = batch[0][:, :batch_max_len].to(f'cuda:{gpu}')
        input_mask = batch[1][:, :batch_max_len].to(f'cuda:{gpu}')
        with torch.no_grad():
            logits = model(input_ids, 
                       pred_mode="cls",
                       token_type_ids=None, 
                       attention_mask=input_mask)
            logits = logits[:, 0, :]
            preds = logits.argmax(dim=-1)
            class_pred.extend(preds.cpu().numpy().tolist())
            prob = softmax(logits.cpu().numpy(), axis=-1)
            class_scores.extend(prob.tolist())
            class_conf.extend(np.amax(prob, axis=-1).tolist())
            if with_label:
                labels.extend(batch[2].numpy().tolist())
    if with_label:
        print(acc(class_pred, labels))
    if return_pred:
        return class_pred
    return class_scores, class_conf


def train_cls(model, train_loader, max_epochs, num_labels, gpu, loss_fn=None, test_loader=None, warmup = 0.1, lr=2e-5, eps=1e-8, verbose=True):
    total_steps = len(train_loader) * max_epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup*total_steps, 
                                                num_training_steps=total_steps)
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()
    
    print('head token fine-tuning')
    model.zero_grad()
    ite1 = range(max_epochs) if verbose else tqdm(range(max_epochs))
    for e in ite1:
        total_train_loss = 0
        ite2 = tqdm(train_loader) if verbose else train_loader
        for j, batch in enumerate(ite2):
            input_ids = batch[0].to(f'cuda:{gpu}')
            input_mask = batch[1].to(f'cuda:{gpu}')
            labels = batch[2].to(f'cuda:{gpu}')
            logits = model(input_ids, 
                           pred_mode="cls",
                           token_type_ids=None, 
                           attention_mask=input_mask)
            logits = logits[:, 0, :]
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        if verbose:
            print(f'epoch {e}', total_train_loss / j)
        if test_loader:
            _ = get_cls(model, test_loader, gpu)


def train_prompt(model, train_loader, max_epochs, gpu, warmup = 0.1, lr=2e-5, eps=1e-8, verbose=True):
    total_steps = len(train_loader) * max_epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup*total_steps, 
                                                num_training_steps=total_steps)

    loss_fn = nn.BCELoss()
    m = nn.Sigmoid()
    
    print('prompt-based fine-tuning')
    model.zero_grad()
    ite1 = range(max_epochs) if verbose else tqdm(range(max_epochs))
    for e in ite1:
        total_train_loss = 0
        ite2 = tqdm(train_loader) if verbose else train_loader
        for j, batch in enumerate(ite2):
            input_ids = batch[0].to(f'cuda:{gpu}')
            input_mask = batch[1].to(f'cuda:{gpu}')
            positions = batch[2]
            labels = batch[3].to(f'cuda:{gpu}').float()
            logits = model(input_ids, 
                           pred_mode="prompt",
                           token_type_ids=None, 
                           attention_mask=input_mask)
            loss = 0
            for i in range(len(positions)):
                loss += loss_fn(m((logits[i][positions[i][0]:positions[i][1]]).mean()), labels[i])
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        if verbose:
            print(f'epoch {e}', total_train_loss / j)


def get_prompting_batch(model, data_dict, gpu, num_labels, batch_size=4, labels=None, selected_ids=None):
    print('prompting PLM')
    if not selected_ids:
        selected_ids = list(range(data_dict["input_ids"].size(0) // num_labels))
        dataset = TensorDataset(data_dict["input_ids"], 
                                data_dict["attention_masks"], 
                                data_dict["positions"])
    else:
        data_ids = []
        for i in selected_ids:
            data_ids.extend([i*num_labels+j for j in range(num_labels)])
        dataset = TensorDataset(data_dict["input_ids"][data_ids], 
                                data_dict["attention_masks"][data_ids], 
                                data_dict["positions"][data_ids])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_max_len = batch[1].sum(dim=1).max()
            input_ids = batch[0][:, :batch_max_len].to(f'cuda:{gpu}')
            input_mask = batch[1][:, :batch_max_len].to(f'cuda:{gpu}')
            positions = batch[2]
            logits = model(input_ids, 
                           pred_mode="prompt",
                           token_type_ids=None, 
                           attention_mask=input_mask)

            for i in range(len(positions)):
                all_scores.append(-(logits[i][positions[i][0]:positions[i][1]]).mean().item())
    
    pred_dis = []
    pred_conf = []
    pred_scores = []
    all_scores = np.array(all_scores).reshape((len(selected_ids), -1))
    for scores in all_scores:
        pred_dis.append(np.argmax(scores))
        pred_scores.append(softmax(scores))
        pred_conf.append(pred_scores[-1][pred_dis[-1]])
    if labels:
        print(acc(pred_dis, [labels[i] for i in selected_ids]))
    return pred_scores, pred_conf


def get_prompting_batch_multi(models, data_dict, gpu, num_labels, freeze_layers=None, batch_size=4, labels=None, selected_ids=None):
    print('prompting PLM')
    if not selected_ids:
        selected_ids = list(range(data_dict["input_ids"].size(0) // num_labels))
        dataset = TensorDataset(data_dict["input_ids"], 
                                data_dict["attention_masks"], 
                                data_dict["positions"])
    else:
        data_ids = []
        for i in selected_ids:
            data_ids.extend([i*num_labels+j for j in range(num_labels)])
        dataset = TensorDataset(data_dict["input_ids"][data_ids], 
                                data_dict["attention_masks"][data_ids], 
                                data_dict["positions"][data_ids])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = [[] for _ in models]
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_max_len = batch[1].sum(dim=1).max()
            input_ids = batch[0][:, :batch_max_len].to(f'cuda:{gpu}')
            input_mask = batch[1][:, :batch_max_len].to(f'cuda:{gpu}')
            positions = batch[2]
            all_logits = []
            if freeze_layers:
                input_shape = input_ids.size()
                input_ids = models[0](input_ids, 
                               pred_mode="inner",
                               start_at=freeze_layers,
                               token_type_ids=None, 
                               attention_mask=input_mask)
                extended_mask = models[0].get_extended_attention_mask(input_mask, input_shape, device=f'cuda:{gpu}')
                for model in models:
                    logits = model(input_ids, 
                                   pred_mode="prompt",
                                   start_at=freeze_layers,
                                   token_type_ids=None, 
                                   attention_mask=extended_mask)
                    all_logits.append(logits)
            else:
                for model in models:
                    logits = model(input_ids, 
                                   pred_mode="prompt",
                                   token_type_ids=None, 
                                   attention_mask=input_mask)
                    all_logits.append(logits)
            for j, logits in enumerate(all_logits):
                for i in range(len(positions)):
                    all_scores[j].append(-(logits[i][positions[i][0]:positions[i][1]]).mean().item())
    pred_conf = [[] for _ in models]
    pred_scores = [[] for _ in models]
    for i, scores in enumerate(all_scores):
        pred_dis = []
        scores = np.array(scores).reshape((len(selected_ids), -1))
        scores = softmax(scores, axis=1)
        for s in scores:
            pred_dis.append(np.argmax(s))
            pred_scores[i].append(s)
            pred_conf[i].append(s[pred_dis[-1]])
        if labels:
            print(acc(pred_dis, [labels[i] for i in selected_ids]))
    return pred_scores, pred_conf
