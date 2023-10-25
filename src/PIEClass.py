from transformers import ElectraTokenizerFast
import argparse
from data_utils import *
from model import *
from training import *

def PIEClass(args):

    if args.prompt == 'senti':
        prompt = senti_prompt
    elif args.prompt == 'topic':
        prompt = topic_prompt
    else:
        raise
    gpu = args.gpu
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

    train_data = create_dataset(args.data_dir, args.train, 'train.pt', tokenizer, max_len=args.max_len)
    num_docs = train_data['input_ids'].size(0)
    loader = make_dataloader(train_data, args.eval_batch_size)

    id2labels, id2labels_id = load_label_names(args.data_dir, args.label_names, tokenizer)
    num_labels = len(id2labels)
    train_data_prompt = create_prompt_dataset(args.data_dir, args.train, id2labels_id, tokenizer, prompt, 'train_prompt.pt', max_len=args.max_len)

    if args.test != '':
        if args.test_labels == '':
            test_data = create_dataset(args.data_dir, args.test, 'test.pt', tokenizer, max_len=args.max_len)
        else:
            test_data = create_dataset(args.data_dir, args.test, 'test.pt', tokenizer, label_file=args.test_labels, max_len=args.max_len)
        test_loader = make_dataloader(test_data, args.eval_batch_size)
    else:
        test_loader = loader

    # initial pseudo labels
    model = ClassModel.from_pretrained("google/electra-base-discriminator",
                                        num_labels=num_labels).to(f'cuda:{gpu}')
    pred_scores, pred_conf = get_prompting_batch(model, train_data_prompt, gpu, num_labels, 
                                                batch_size=args.eval_batch_size)
    partial_hard, partial_ids = get_pseudo_label(pred_scores, pred_conf, num_labels, top_k = ceil(num_docs*0.1), thres=0.5, imbalanced=args.imbalanced)


    # iterative classifier trianing
    for ite in range(args.num_iter):
        print(f'start iteration {ite}')
        if args.imbalanced:
            partial_hard, partial_ids = up_sample(partial_hard, partial_ids)
        train_loader = make_cls_dataloader(train_data, partial_hard, partial_ids, args.train_batch_size)
        model = ClassModel.from_pretrained("google/electra-base-discriminator",
                                       num_labels=num_labels).to(f'cuda:{gpu}')
        model.freeze_layers(args.freeze_layers)
        train_cls(model, train_loader, 5, num_labels, gpu, verbose=False, lr=args.cls_lr)
        class_scores, class_conf = get_cls(model, loader, gpu)
        if args.test != '' and args.test_labels != '':
            _ = get_cls(model, test_loader, gpu)
        partial_hard, partial_ids = get_pseudo_label(class_scores, class_conf, num_labels, imbalanced=args.imbalanced,
                                                    top_k=ceil(num_docs * args.thres_t * (ite+1)), thres=args.thres_p)
        for i in range(args.prompt_num):
            sampled_ids, sampled_hard = random_sample(partial_ids, partial_hard, ratio=0.01)
            model = ClassModel.from_pretrained("google/electra-base-discriminator",
                                               num_labels=len(id2labels)).to(f'cuda:{gpu}')
            model.freeze_layers(args.freeze_layers)
            if args.imbalanced:
                sampled_hard, sampled_ids = up_sample(sampled_hard, sampled_ids)
            train_loader = make_prompt_dataloader(train_data_prompt, sampled_hard, sampled_ids, num_labels, 
                                                    batch_size=args.train_batch_size)
            train_prompt(model, train_loader, 10, gpu, lr=args.prompt_lr, verbose=False)
            torch.save(model.state_dict(), f'{args.data_dir}/prompt_temp_{i}.pt')
        models = []
        for i in range(args.prompt_num):
            model = ClassModel.from_pretrained("google/electra-base-discriminator",
                                               output_hidden_states=True,
                                               num_labels=len(id2labels))
            model.load_state_dict(torch.load(f'{args.data_dir}/prompt_temp_{i}.pt'))
            model.to(f'cuda:{gpu}')
            models.append(model)
        all_pred_scores, all_pred_conf = get_prompting_batch_multi(models, train_data_prompt, gpu, num_labels, 
                                                                    batch_size=args.eval_batch_size, 
                                                                    freeze_layers=args.freeze_layers)
        all_res = [(s, c) for s, c in zip(all_pred_scores, all_pred_conf)]
        all_res.append((class_scores, class_conf))

        all_pred = [np.argmax(scores, axis=1) for scores, _ in all_res]
        agreed_ids = [i for i in range(num_docs) if all(all_pred[j][i] == all_pred[-1][i] for j in range(len(all_pred)-1))]

        final_ids = set(agreed_ids)
        for s, c in all_res:
            _, ids = get_pseudo_label(s, c, num_labels, imbalanced=args.imbalanced, thres=args.thres_p, 
                                        top_k=ceil(num_docs * args.thres_t*(ite+1)))
            final_ids = final_ids.intersection(set(ids))
        partial_ids = list(final_ids)
        partial_hard = all_pred[0][partial_ids]
    for i in range(args.prompt_num):
        if os.path.isfile(f'{args.data_dir}/prompt_temp_{i}.pt'):
            os.remove(f'{args.data_dir}/prompt_temp_{i}.pt')

    # final classifier training
    model = ClassModel.from_pretrained("google/electra-base-discriminator",
                                       num_labels=num_labels).to(f'cuda:{gpu}')
    train_loader = make_cls_dataloader(train_data, partial_hard, partial_ids, args.train_batch_size)
    train_cls(model, train_loader, 5, num_labels, gpu)


    pred = get_cls(model, test_loader, gpu, True)
    with open(os.path.join(args.data_dir, 'predictions.txt'), 'w') as f:
        for l in pred:
            print(l, file=f)

    # save the final model
    if args.save_model:
        torch.save(model.state_dict(), f'{args.data_dir}/PIEClass_ELECTRA.pt')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--train', default='train.txt', type=str, help='training corpus')
    parser.add_argument('--label_names', default='label_names.txt', type=str, help='label names')
    parser.add_argument('--test', default='', type=str, help='testing corpus, will use training corpus if testing is not available')
    parser.add_argument('--test_labels', default='', type=str, help='testing labels, leave black if not available')
    parser.add_argument('--prompt', default='senti', type=str, help='type of prompt')
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--freeze_layers', default=None, type=int)
    parser.add_argument('--num_iter', default=5, type=int)
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--thres_t', default=0.2, type=float)
    parser.add_argument('--thres_p', default=0.95, type=float)
    parser.add_argument('--cls_lr', default=2e-5, type=float)
    parser.add_argument('--prompt_lr', default=2e-5, type=float)
    parser.add_argument('--prompt_num', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--imbalanced', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    args = parser.parse_args()

    PIEClass(args)