import argparse
import pickle
import time
import dgl

from pandas import DataFrame
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
from Task11model_MI_mol import FusionLayer,GNN1,GNN2
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")
torch.cuda.set_device(device)


print(device)

parser = argparse.ArgumentParser(description='CompactDDI_Existing Drug-Existing Drug_Deng et al. Dataset')
parser.add_argument("--epoches",type=int,choices=[100,500,1000,2000],default=120)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=1*1e-2) #4*1e-3
parser.add_argument("--event_num",type=int,default=65)

parser.add_argument("--n_drug",type=int,default=572)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--dropout",type=float,default=0.3)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=128)
args = parser.parse_args()
def prepare2( df_drug, feature_list, mechanism, action):
    d_label = {}
    d_feature = {}

    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
    for i in feature_list:
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
    return d_feature

def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(
            np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix
def focal_loss(preds, labels):
    labels = labels.view(-1, 1)
    preds = preds.view(-1, preds.size(-1))

    preds_logsoft = F.log_softmax(preds, dim=1)
    preds_softmax = torch.exp(preds_logsoft)

    preds_softmax = preds_softmax.gather(1, labels)
    preds_logsoft = preds_logsoft.gather(1, labels)

    loss = -torch.mul(torch.pow((1 - preds_softmax), 2),
                          preds_logsoft)

    loss = loss.mean()

    return loss

def setup_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def read_network(path):

    edge_index = []
    rel_index = []

    flag = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            else:
                flag += 1
                head, tail, rel = line.strip().split(" ")[:3]
                edge_index.append([int(head), int(tail)])
                rel_index.append(int(rel))

        f.close()
    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1

    return num_node, edge_index, rel_index, num_rel
def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = "../Deng Dataset/Deng Dataset"+str(num)+".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations)


def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label

def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type,classes= np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task1'+'.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x,train_y,test_x,test_y,net):
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.clone()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    train_x_total = torch.LongTensor(np.concatenate([
        train_x1.cpu().numpy(),
        train_x.cpu().numpy()
    ], axis=0)).to(device)
    train_y = torch.LongTensor(np.concatenate([
        train_y.cpu().numpy(),
        train_y.cpu().numpy()
    ])).to(device)
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,65),dtype=torch.float).to(device)
    for epoch in range(args.epoches):
        t = time.time()
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)

            opti.zero_grad()
            train_acc = 0
            train_label = y
            x = x
            output, loss_mi1, smooth_loss, AE_loss1 = net(x)
            loss_ori=loss_function(output, train_label)
            l = loss_ori + loss_mi1 + 10 * smooth_loss + AE_loss1
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output,dim=1).cpu(), train_label.cpu())
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            output2, loss_mi2,smooth_loss2, AE_loss2  = net(test_x)
            test_output = F.softmax(output2, dim=1)
            test_label = test_y
            loss = loss_function(test_output, test_label) + loss_mi2 + 10 * smooth_loss2 + AE_loss2
            test_loss = loss.item()
            test_output_cpu = torch.argmax(test_output, dim=1).cpu().numpy()
            test_label_cpu = test_label.cpu().numpy()

            test_score = f1_score(test_output_cpu, test_label_cpu, average='macro')
            test_output_cpu = torch.argmax(test_output, dim=1).cpu().numpy()
            test_label_cpu = test_label.cpu().numpy()

            test_acc = accuracy_score(test_output_cpu, test_label_cpu)
            test_list.append(test_score)
            if test_score==max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:", test_score,"time:", time.time() - t)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output



def main():
    conn = sqlite3.connect("../Deng Dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    feature_list = ["smile", "target", "enzyme"]

    print("Preparing DDI features, this may take a while...")
    d_feature_dict = prepare2(df_drug, feature_list, mechanism, action)

    new_label = prepare(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)
    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]

    drug_names = df_drug['name'].tolist()

    feature_matrix = [d_feature_dict[name] for name in drug_names]

    bio_feature_tensor = torch.from_numpy(np.array(feature_matrix)).float()


    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()


    train_sum, test_sum = 0, 0

    y_true = np.array([])
    y_score = np.zeros((0, 65), dtype=float)
    y_pred = np.array([])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    x_datasets = np.array(x_datasets)
    new_label = np.array(new_label)

    with open('drug_with_description.pkl', 'rb') as file:
        doc_embeddings = pickle.load(file)
    description = doc_embeddings

    graphs, labels = dgl.load_graphs("kg_data.bin")
    kg_g = graphs[0]

    smiles_list = []
    with open('id_smiles_indexed.tsv') as file:
        for line in file.readlines():
            smiles = line.strip().split('\t')[1]
            smiles_list.append(smiles)

    kfold = kf.split(x_datasets, new_label)
    for i, (train_idx, test_idx) in enumerate(kfold):
        if i == 0:
            t = time.time()
            net = nn.Sequential(GNN1(kg_g, args),
                                GNN2(smiles_list, args),
                                FusionLayer(args, description, bio_feature_tensor)).to(device)
            train_x = torch.tensor(x_datasets[train_idx], dtype=torch.long).to(device)
            train_y = torch.tensor(new_label[train_idx], dtype=torch.long).to(device)
            test_x = torch.tensor(x_datasets[test_idx], dtype=torch.long).to(device)
            test_y = torch.tensor(new_label[test_idx], dtype=torch.long).to(device)

            test_loss, test_acc, train_loss, train_acc, test_list, test_output = train(train_x, train_y, test_x, test_y,
                                                                                       net)
            train_sum += train_acc
            test_sum += test_acc
            pred_type = torch.argmax(test_output, dim=1).cpu().numpy()
            y_pred = np.hstack((y_pred, pred_type))
            y_score = np.row_stack((y_score, test_output.cpu().numpy()))
            y_true = np.hstack((y_true, test_y.cpu().numpy()))
            print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f, time %f' % (
                i, test_loss, test_acc, train_loss, train_acc, time.time() - t))
            break
    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("../result/", "all", result_all)
    save_result("../result/", "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (i, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    setup_seed()
    main()