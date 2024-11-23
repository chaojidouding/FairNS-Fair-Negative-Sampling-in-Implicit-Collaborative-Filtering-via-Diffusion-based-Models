import os
import random
from pdb import post_mortem

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

from modules.Diffnet import *
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end,item_type, n_negs=1):
    def sampling(user_item, train_set, n, item_type):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            pos_item = int(_)
            negitems = []
            for i in range(n):  # sample n times
                # 负采样方式
                # 1.随机
                if args.rand_type == 1:
                    while True:
                        negitem = random.choice(range(n_items))
                        if negitem not in train_set[user]:
                            break
                # 2.概率
                # 生成随机数，285/503概率为0，218/503概率为1
                # 生成1-503的随机数
                elif args.rand_type == 2:
                    if args.dataset == "ml-2-types":
                        rand = random.randint(1, 503)
                        if rand <= 285:
                            rand_type = 0
                        else:
                            rand_type = 1
                    else:
                        rand = random.randint(1, 1543)
                        if rand <= 343:
                            rand_type = 0
                        else:
                            rand_type = 1
                    while True:
                        negitem = random.choice(range(n_items))
                        # print(negitem,rand_type, item_type[negitem])
                        if negitem not in train_set[user] and item_type[negitem][0] == rand_type:
                            break
                # 3.同类型
                elif args.rand_type == 3:
                    pos_type = item_type[pos_item]
                    cnt = 0
                    while True:
                        cnt += 1
                        negitem = random.choice(range(n_items))
                        # print(pos_type[0],item_type[negitem][0],negitem)
                        # print(pos_type, negitem,item_type[negitem][0])
                        if cnt > 1000 and negitem not in train_set[user]:
                            break
                        if negitem not in train_set[user] and item_type[negitem][0] == pos_type[0]:
                            break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    def sampling2(user_item, train_set, n,item_type):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            pos_item = int(_)
            negitems = []
            for i in range(n):  # sample n times
                #负采样方式
                #3.同类型
                pos_type = item_type[pos_item]
                while True:
                    negitem = random.choice(range(n_items))
                    # print(pos_type, negitem,item_type[negitem][0])
                    if negitem not in train_set[user] and item_type[negitem][0] == pos_type[0]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = sampling(entity_pairs, train_pos_set, n_negs,item_type)
    # feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
    #                                                    train_pos_set,
    #                                                    n_negs*K)).to(device)
    if args.run_type == 8 or args.run_type == 9:
        feed_dict['neg_items2'] = sampling2(entity_pairs, train_pos_set, n_negs,item_type)

    return feed_dict


def merge_tensors(tensors, values):
    # 检查输入是否有效
    if not tensors or not values or len(tensors) != len(values):
        raise ValueError("tensors 和 values 必须是非空且长度相等的列表")

    # 将 values 转换为 tensor
    weights = torch.tensor(values)
    weights = weights.to(device)
    # 确保 weights 可以广播到 tensors 中的任意张量的形状
    # 这里假设所有 tensors 的形状相同
    shape = tensors[0].shape
    weights = weights.view(-1, *([1] * len(shape)))

    # 使用加权求和来合并张量
    merged_tensor = sum(w * t for w, t in zip(weights, tensors))

    return merged_tensor

def get_feed_dict_by_diffution(batch,model,diffution):
    users = batch['users']
    pos_items = batch['pos_items']
    neg_items = batch['neg_items']

    user_embeddings, item_embeddings = model.get_embedding(batch)
    # print("())",user_embeddings.shape, item_embeddings.shape)
    # print(user_embeddings.shape)
    users_emb = user_embeddings[users]
    pos_items_emb = item_embeddings[pos_items]
    neg_items_emb = item_embeddings[neg_items]
    # print(pos_items.shape)
    pos_items_squeeze = pos_items_emb.squeeze(1)
    # print(pos_items_squeeze.shape)
    # print("***")
    neg_diffution_emb_list = diffution.sample(pos_items_squeeze.shape,pos_items_squeeze)
    # print(neg_items_emb.shape)

    #neg_emb_list:len=K shape=[batch_size,channel]

    # [K, batch_size, channel]
    neg_emb_tensor = torch.stack(neg_diffution_emb_list)

    # [batch_size, K, channel]
    neg_emb_tensor = neg_emb_tensor.permute(1, 0, 2)

    # [batch_size, K, 1, channel]
    neg_emb_tensor = neg_emb_tensor.unsqueeze(2)

    #merge neg_items_emb and neg_emb_tensor
    # print(neg_items_emb.shape, neg_emb_tensor.shape)
    merged_tensor = torch.cat((neg_items_emb, neg_emb_tensor), dim=1)
    if args.run_type == 8 or args.run_type == 9:
        neg_items2 = batch['neg_items2']
        neg_items_emb2 = item_embeddings[neg_items2]
    # print(merged_tensor.shape)
    # print("merged_tensor.shap e", merged_tensor.shape)
    split_tensors = torch.split(neg_emb_tensor, 1, dim=1)
    # 分别获取四个张量
    tensor1 = split_tensors[0]  # 第一个张量
    tensor2 = split_tensors[1]  # 第二个张量
    tensor3 = split_tensors[2]  # 第三个张量
    tensor4 = split_tensors[3]  # 第四个张量

    feed_dict = {}
    feed_dict['users'] = batch['users']
    feed_dict['users_emb'] = users_emb
    feed_dict['pos_items_emb'] = pos_items_emb
    if args.run_type == 1:
        tensor_list = [tensor1, tensor2, tensor3, tensor4, neg_items_emb]
        value_list = [0.05,0.05,0.05,0.05,0.8]
        # value_list = [0,0,0,0,1]
        feed_dict['neg_items_emb'] = merge_tensors(tensor_list, value_list)
        # feed_dict['neg_items_emb'] = merged_tensor
    elif args.run_type == 2:
        feed_dict['neg_items_emb'] = neg_items_emb
    elif args.run_type == 3:
        feed_dict['neg_items_emb'] = neg_emb_tensor
    elif args.run_type == 4:
        # print(tensor1)
        feed_dict['neg_items_emb'] = torch.cat((tensor1, neg_items_emb), dim=1)
    elif args.run_type == 5:
        feed_dict['neg_items_emb'] = torch.cat((tensor2, neg_items_emb), dim=1)
    elif args.run_type == 6:
        feed_dict['neg_items_emb'] = torch.cat((tensor3, neg_items_emb), dim=1)
    elif args.run_type == 7:
        feed_dict['neg_items_emb'] = torch.cat((tensor4, neg_items_emb), dim=1)
    elif args.run_type == 8:
        # print(neg_items_emb.shape,neg_items_emb2.shape)
        feed_dict['neg_items_emb'] = torch.cat((neg_items_emb, neg_items_emb2), dim=1)

    elif args.run_type == 9:
        neg_gcn_emb_dns = []
        # print(tensor1)
        for k in range(1):
            neg_gcn_emb_dns.append(model.dynamic_negative_sampling(user_embeddings, item_embeddings,
                                                                       users,
                                                                       torch.tensor(neg_items).to(device),
                                                                       pos_items))
        neg_gcn_emb_dns = torch.stack(neg_gcn_emb_dns, dim=1)

        tensor_list = [tensor1, tensor2, tensor3, tensor4, neg_gcn_emb_dns]
        # print(tensor1)
        # print(neg_gcn_emb_dns)
        value_list = [0.05,0.05,0.05,0.05,0.8]
        # print(merge_tensors(tensor_list, value_list))
        # value_list = [0,0,0,0,1]
        feed_dict['neg_items_emb'] = merge_tensors(tensor_list, value_list)
        # feed_dict['neg_items_emb'] = torch.cat((neg_gcn_emb_dns, neg_items_emb2), dim=1)
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, item_type,total_types,type_num_list = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""


    # MF can be implemented by setting the number of layers of LightGCN to 0.
    from modules.LightGCN import LightGCN
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        raise NotImplementedError("model %s not supported" % args.gnn)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    diffusion = Diffusion_Cond(args.nhid, args.nhid, args, args.nhid)
    d_optimizer = torch.optim.Adam(diffusion.parameters(), \
                                   lr=args.d_lr, weight_decay=args.decay)

    model = model.to(device)
    diffusion = diffusion.to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    cur_best_pre_fair = 100
    cur_best_pre_loss = 100000
    stopping_step = 0
    should_stop = False
    logger.info("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)
        """training"""
        model.train()
        patience_cnt = 0
        loss, s = 0, 0
        hits = 0
        train_s_t = time.time()
        num_iterations = (len(train_cf) - 1) // args.batch_size + 1
        for _ in range(num_iterations):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,item_type,
                                  n_negs)
            #通过model获取embedding
            user_embeddings,item_embeddings = model.get_embedding(batch)
            user_embeddings = user_embeddings.to(device)
            item_embeddings = item_embeddings.to(device)


            def train_diffusion(train_batch):
                pos_item_embeddings_output = item_embeddings[train_batch['pos_items']].detach()
                neg_item_embeddings_output = item_embeddings[train_batch['neg_items']].detach()
                user_embeddings_output = user_embeddings[train_batch['users']].detach()
                user_embeddings_output = user_embeddings_output.squeeze(1)
                pos_item_embeddings_output = pos_item_embeddings_output.squeeze(1)
                neg_item_embeddings_output = neg_item_embeddings_output.squeeze(1)
                for d_epoch in range(args.d_epoch):
                    d_optimizer.zero_grad()

                    dif_loss = diffusion(pos_item_embeddings_output, neg_item_embeddings_output,user_embeddings_output ,device)
                    dif_loss.backward(retain_graph=True)
                    d_optimizer.step()
                return
            train_diffusion(batch)
            #generate neg_item by diffution model
            batch_diffusion = get_feed_dict_by_diffution(batch,model,diffusion)

            batch_loss, _, _ = model(epoch, batch_diffusion)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
            s = min(s, len(train_cf))
        train_e_t = time.time()
        """testing"""
        if (epoch + 1) % 2 == 0:
            model.eval()
            test_s_t = time.time()
            test_ret = test(model, user_dict, n_params, item_type,total_types,type_num_list, mode='test')
            test_e_t = time.time()
            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time.time()
                valid_ret = test(model, user_dict, n_params, item_type,total_types,type_num_list, mode='valid')
                test_e_t = time.time()
            if (epoch + 1) % 10 == 0:
                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                     "precision", "hit_ratio", "item_fairness","item_fairness_std"]
                train_res.add_row(
                [epoch + 1, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'],
                 valid_ret['ndcg'],
                 valid_ret['precision'], valid_ret['hit_ratio'], valid_ret['item_fairness'],valid_ret['item_fairness_std']])
                print(valid_ret)
                print(train_res)
            # best results
            if valid_ret['recall'][0] > cur_best_pre_0:
                cur_best_pre_0 = test_ret['recall'][0]
                torch.save(model.state_dict(), args.out_dir + 'model_best.ckpt')  # save weight
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt == args.patience:
                break
    logger.info("start load best model ...")
    # load best model
    best_model_path = args.out_dir + 'model_best.ckpt'
    model.load_state_dict(torch.load(best_model_path))

    # testing
    model.eval()
    final_test_ret = test(model, user_dict, n_params,item_type,total_types,type_num_list,mode='test')

    train_res = PrettyTable()
    train_res.field_names = ["recall", "ndcg", "precision", "hit_ratio", "item_fairness","item_fairness_std"]
    print("final test result")
    print(final_test_ret)
    train_res.add_row(
        [final_test_ret['recall'], final_test_ret['ndcg'], final_test_ret['precision'], final_test_ret['hit_ratio'],
         final_test_ret['item_fairness'],final_test_ret['item_fairness_std']])
    print(train_res)

