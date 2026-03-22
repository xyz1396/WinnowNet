import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn import metrics
import numpy as np
import glob
import pickle
import sys
import getopt
import os
from pkl_utils import (
    expand_pickle_inputs,
    get_entry_label,
    get_entry_label_confidence,
    get_entry_model_input,
    load_feature_pickle,
    normalize_long_flag_aliases,
)

threshold=0.9
def _label_matches_expected(entry):
    label = get_entry_label(entry)
    if label is None:
        return True
    return label == 1


def _load_checkpoint_weights(model_path):
    return torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )


def _load_feature_records(feature_paths, force_label=None, dataset_name="dataset"):
    L = []
    Yweight = []
    total_rows = 0
    kept_rows = 0
    skipped_non_one = 0
    unlabeled_rows = 0

    for feature_path in feature_paths:
        _, entries = load_feature_pickle(feature_path)
        for _, entry in entries.items():
            total_rows += 1
            model_input = get_entry_model_input(entry)
            if model_input is None:
                continue

            if force_label is None:
                label = get_entry_label(entry)
                confidence = get_entry_label_confidence(entry)
                if label is None:
                    continue
                if confidence is None:
                    confidence = 1.0 if label == 1 else 0.0
            else:
                if get_entry_label(entry) is None:
                    unlabeled_rows += 1
                elif not _label_matches_expected(entry):
                    skipped_non_one += 1
                    continue
                label = force_label
                confidence = 1.0 if force_label == 1 else 0.0

            if label == 1:
                if confidence > threshold:
                    L.append(model_input[0])
                    Yweight.append([1, 1])
                    kept_rows += 1
            else:
                L.append(model_input[0])
                Yweight.append([0, confidence])
                kept_rows += 1

    print(
        f"[{dataset_name}] total_rows={total_rows} kept_rows={kept_rows} "
        f"unlabeled_rows={unlabeled_rows} skipped_label_not_1={skipped_non_one}"
    )
    return L, Yweight


def _collect_pickles_from_directory(directory):
    return sorted(glob.glob(os.path.join(directory, "*.pkl")))


def _resolve_training_inputs(input_directory, explicit_target, explicit_decoy):
    target_pickles = expand_pickle_inputs(explicit_target)
    decoy_pickles = expand_pickle_inputs(explicit_decoy)
    if target_pickles or decoy_pickles:
        if not target_pickles or not decoy_pickles:
            raise ValueError("Both -target and -decoy must be provided together.")
        return target_pickles, decoy_pickles, True

    pct1_dir = os.path.join(input_directory, "pct1")
    pct2_dir = os.path.join(input_directory, "pct2")
    if os.path.isdir(pct1_dir) and os.path.isdir(pct2_dir):
        return _collect_pickles_from_directory(pct2_dir), _collect_pickles_from_directory(pct1_dir), True

    return _collect_pickles_from_directory(input_directory), [], False


def split_list(X, Yweight, val_ratio=0.1, test_ratio=0.1, seed=10):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    test_n = max(1, int(n * test_ratio)) if n >= 10 else max(0, n // 5)
    val_n = max(1, int(n * val_ratio)) if n - test_n >= 10 else max(0, (n - test_n) // 5)
    test_idx = idx[:test_n]
    val_idx = idx[test_n:test_n + val_n]
    train_idx = idx[test_n + val_n:]

    def _pick(indices):
        return [X[i] for i in indices], [Yweight[i] for i in indices]

    return _pick(train_idx), _pick(val_idx), _pick(test_idx)


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight):
        self.X = X
        self.yweight = yweight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mz = self.X[idx][0]
        exp = self.X[idx][1]
        theory = self.X[idx][2]
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]

        xFeatures=[]
        for i in range(len(mz)):
            xFeatures.append([mz[i],exp[i],theory[i]])

        xFeatures=np.asarray(xFeatures,dtype=float)
        xFeatures = xFeatures.transpose()
        xFeatures = torch.FloatTensor(xFeatures)

        return xFeatures, y, weight



class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum

class T_Net(nn.Module):
    def __init__(self, k):
        super(T_Net, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = T_Net(k=3)
        self.fstn = T_Net(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        pointfeat=x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, trans, trans_feat



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, matrix3x3, matrix64x64 = self.transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.fc3(x)
        return output

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data,batch_size=32)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data1, label, weight in data_loader:
        data1, label, weight = Variable(data1), Variable(label), Variable(weight)
        data1, label, weight = data1.to(device),label.to(device), weight.to(device)

        output = model(data1)
        losses = loss(output, label, weight)

        total_loss += losses.data.item()
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data.cpu().numpy().tolist())

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    Pos_prec = 0
    Neg_prec = 0

    if y_pred.count(1) == 0:
        Pos_prec = 0
    elif y_pred.count(0) == 0:
        Neg_prec = 0
    else:
        for idx in range(len(y_pred)):
            if y_pred[idx] == 1:
                if y_true[idx] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if y_true[idx] == 1:
                    TN += 1
                else:
                    FN += 1

        Pos_prec = TP / (TP + FP)
        Neg_prec = FN / (TN + FN)

    return acc / data_len, total_loss / data_len, Pos_prec, Neg_prec



def test_model(model, test_data, device):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=32)

    model.load_state_dict(_load_checkpoint_weights('cnn_pytorch.pt'))

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,label, weight in test_loader:
        y_true.extend(label.data)
        data1,label, weight = Variable(data1), Variable(label), Variable(weight)
        data1,label, weight = data1.to(device),label.to(device), weight.to(device)

        output = model(data1)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred.extend(pred)
        y_pred_prob.extend(pred_prob[:, 1].tolist())

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print(
        "Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(
        y_true, y_pred, target_names=['T', 'D']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test,model_name,pretrained_model):
    LR = 1e-3
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    device = torch.device("cuda")
    model = Net()
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    if len(pretrained_model)>0:
        print("loading pretrained_model")
        model.load_state_dict(_load_checkpoint_weights(pretrained_model))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    #model.load_state_dict(
    #    torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    train_loader = Data.DataLoader(train_data, batch_size=128, num_workers=8, shuffle=True, pin_memory=True)
    for epoch in range(0, 50):
        start_time = time.time()
        best_epoch_loss = 10000
        # load the training data in batch
        batch_count = 0
        model.train()
        for x1_batch, y_batch, weight in train_loader:
            batch_count = batch_count + 1
            inputs, targets, weight = Variable(x1_batch),Variable(y_batch), Variable(
                weight)
            inputs, targets, weight = inputs.to(device), targets.to(device), weight.to(
                device)
            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets, weight)
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), 'checkpoints/epoch' + str(epoch) + '.pt')
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(val_data, model, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_name)

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Val_loss: {5:>6.2}, Val_acc {6:>6.2%},Val_Posprec {7:6.2%}, Val_Negprec {8:6.2%} " \
                             "Time: {9} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, val_loss, val_acc,
                         val_PosPrec, val_Negprec, time_dif))



if __name__ == "__main__":
    argv = normalize_long_flag_aliases(
        sys.argv[1:],
        {
            "-target": "--target",
            "-decoy": "--decoy",
        },
    )
    try:
        opts, args = getopt.getopt(argv, "hi:m:p:t:", ["target=", "decoy="])
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directory containing feature pickles with embedded labels\n")
        print("-m\t Output trained model name\n")
        print("-p\t Optional pretrained model name\n")
        print("-target\t Target feature pickle(s), comma-separated or repeated\n")
        print("-decoy\t Decoy feature pickle(s), comma-separated or repeated\n")
        sys.exit(1)
        start_time=time.time()
    input_directory=""
    model_name=""
    pretrained_model=""
    target_inputs = []
    decoy_inputs = []
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directory containing feature pickles with embedded labels\n")
            print("-m\t Output trained model name\n")
            print("-p\t Optional pretrained model name\n")
            print("-target\t Target feature pickle(s), comma-separated or repeated\n")
            print("-decoy\t Decoy feature pickle(s), comma-separated or repeated\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_directory=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-p"):
            pretrained_model=arg
        elif opt == "--target":
            target_inputs.append(arg)
        elif opt == "--decoy":
            decoy_inputs.append(arg)
    start = time.time()
    split_mode = bool(target_inputs or decoy_inputs)
    if not split_mode and len(input_directory) == 0:
        raise ValueError("Use -i for embedded-label training or provide both -target and -decoy.")

    target_pickles, decoy_pickles, split_mode = _resolve_training_inputs(
        input_directory,
        target_inputs,
        decoy_inputs,
    )

    if split_mode:
        if len(target_pickles) == 0 or len(decoy_pickles) == 0:
            raise ValueError("Missing target/decoy pickle inputs.")
        X_pos, y_pos = _load_feature_records(target_pickles, force_label=1, dataset_name="target")
        X_neg, y_neg = _load_feature_records(decoy_pickles, force_label=0, dataset_name="decoy")
        (X_train_pos, y_train_pos), (X_val_pos, y_val_pos), (X_test_pos, y_test_pos) = split_list(X_pos, y_pos)
        (X_train_neg, y_train_neg), (X_val_neg, y_val_neg), (X_test_neg, y_test_neg) = split_list(X_neg, y_neg)

        X_train = X_train_pos + X_train_neg
        yweight_train = y_train_pos + y_train_neg
        X_val = X_val_pos + X_val_neg
        yweight_val = y_val_pos + y_val_neg
        X_test = X_test_pos + X_test_neg
        yweight_test = y_test_pos + y_test_neg
    else:
        if len(target_pickles) == 0:
            raise ValueError(f"No feature pickles found under {input_directory}.")
        L, Yweight = _load_feature_records(target_pickles, dataset_name="embedded")
        (X_train, yweight_train), (X_val, yweight_val), (X_test, yweight_test) = split_list(L, Yweight)

    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test, model_name, pretrained_model)
    print('done')
