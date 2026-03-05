import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.utils.data as Data
import time
import sys
import getopt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn import metrics
import numpy as np
import glob
import pickle
import os
from components.encoders import MassEncoder, PeakEncoder, PositionalEncoder

threshold=0.9
def LabelToDict(fp):
    sample = fp.read().strip().split('\n')
    label_dic = dict()
    if len(sample) == 0 or len(sample[0].strip()) == 0:
        return label_dic

    header = sample[0].strip().split('\t')
    has_header = ("PSMId" in header or "SpecId" in header) and "Label" in header
    if has_header:
        idx_col = header.index("PSMId") if "PSMId" in header else header.index("SpecId")
        label_col = header.index("Label")
        qvalue_col = header.index("q-value") if "q-value" in header else -1
        for scan in sample[1:]:
            s = scan.strip().split('\t')
            if len(s) <= max(idx_col, label_col):
                continue
            idx = s[idx_col]
            qvalue = float(s[qvalue_col]) if qvalue_col >= 0 and len(s) > qvalue_col else 0.0
            label = 1 if s[label_col] in ("1", "True", "T", "true") else 0
            conf = max(0.0, min(1.0, 1.0 - qvalue))
            label_dic[idx] = [conf if label == 1 else 0, label]
    else:
        for scan in sample:
            s = scan.strip().split('\t')
            if len(s) < 3:
                continue
            idx = s[1]
            if s[0] == 'True':
                label = 1
                label_dic[idx] = [1, label]
            else:
                label = 0
                label_dic[idx] = [0, label]
    return label_dic


def pad_control(data,pairmaxlength):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    width = 2
    if len(data) > 0:
        width = max(len(x) for x in data)
    zero_peak = [0.0] * width
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append(zero_peak.copy())
    for i in range(len(data)):
        if len(data[i]) < width:
            data[i] = list(data[i]) + [0.0] * (width - len(data[i]))
        elif len(data[i]) > width:
            data[i] = list(data[i])[:width]
    data = sorted(data, key=lambda x: x[0])
    return np.asarray(data,dtype=float)


def readData(psms, features, force_label=None):
    L = []
    Yweight = []
    positive=0
    negative=0

    for i in range(len(psms)):
        with open(psms[i]) as f:
            D_Label=LabelToDict(f)
        with open(features[i],'rb') as f:
            D_features=pickle.load(f)

        for j, label_item in D_Label.items():
            if j not in D_features:
                continue
            if force_label is None:
                confidence = label_item[0]
                label = label_item[1]
            else:
                confidence = 1.0 if force_label == 1 else 0.0
                label = force_label

            if label == 1:
                if confidence > threshold:
                    L.append(D_features[j])
                    Y = 1
                    weight = 1
                    positive+=1
                    Yweight.append([Y, weight])
            else:
                L.append(D_features[j])
                Y = 0
                weight = confidence
                negative+=1
                Yweight.append([Y, weight])

        del D_features
    print(positive)
    print(negative)
    return L, Yweight


def pair_psm_and_feature_files(directory):
    psm_files = sorted(glob.glob(os.path.join(directory, "*.tsv")))
    feature_files = sorted(glob.glob(os.path.join(directory, "*.pkl")))
    if len(psm_files) == 0 or len(feature_files) == 0:
        return [], []

    pkl_by_stem = {
        os.path.splitext(os.path.basename(pkl))[0]: pkl for pkl in feature_files
    }
    matched_psm = []
    matched_pkl = []
    used = set()
    for psm in psm_files:
        stem = os.path.splitext(os.path.basename(psm))[0]
        candidates = [
            stem,
            stem.replace("_filtered_psms", ""),
            stem.replace("_psms", ""),
            stem + "_spectra_feature",
        ]
        match = None
        for cand in candidates:
            if cand in pkl_by_stem and pkl_by_stem[cand] not in used:
                match = pkl_by_stem[cand]
                break
        if match is None:
            for pkl_stem, pkl_path in pkl_by_stem.items():
                if pkl_path in used:
                    continue
                if pkl_stem.startswith(stem) or stem.startswith(pkl_stem):
                    match = pkl_path
                    break
        if match is not None:
            matched_psm.append(psm)
            matched_pkl.append(match)
            used.add(match)

    if len(matched_psm) == 0 and len(psm_files) == len(feature_files):
        return psm_files, feature_files
    return matched_psm, matched_pkl


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
        xspectra1 = pad_control(self.X[idx][0],200)
        xspectra2 = pad_control(self.X[idx][1],200)
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]
        xspectra1 = torch.FloatTensor(xspectra1)
        xspectra2 = torch.FloatTensor(xspectra2)

        return xspectra1, xspectra2, y, weight



class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum


class MS2Encoder(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_intensity: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 200
    ):
        super().__init__()
        self.peak_encoder = PeakEncoder(
            dim_model=dim_model,
            dim_intensity=dim_intensity,
            min_wavelength=0.001,
            max_wavelength=7000,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, spectra: torch.Tensor):
        B, P, _ = spectra.shape
        src_key_padding_mask = spectra.sum(dim=2) == 0
        peaks = self.peak_encoder(spectra)
        out = self.transformer(peaks, src_key_padding_mask=src_key_padding_mask)
        return out

class DualPeakClassifier(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        dim_intensity: int = 128,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        n_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_len: int = 200,
    ):
        super().__init__()
        self.encoder1 = MS2Encoder(
            dim_model, dim_intensity, n_heads, dim_feedforward, n_layers, dropout, max_len
        )
        self.encoder2 = MS2Encoder(
            dim_model, dim_intensity, n_heads, dim_feedforward, n_layers, dropout, max_len
        )
        self.classifier = nn.Linear(2 * dim_model, num_classes)

    def forward(
        self,
        spectra1: torch.Tensor,
        spectra2: torch.Tensor,
    ):
        out1 = self.encoder1(spectra1)
        out2 = self.encoder2(spectra2)
        rep1 = out1.mean(dim=1)
        rep2 = out2.mean(dim=1)
        joint = torch.cat([rep1, rep2], dim=-1)
        outputs = self.classifier(joint)
        return outputs

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):

    model.eval()
    data_loader = Data.DataLoader(data,batch_size=1024,num_workers=8, shuffle=True, pin_memory=True)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for input1, input2, label, weight in data_loader:
        input1, input2, label, weight = input1.to(device,non_blocking=True),input2.to(device,non_blocking=True),label.to(device,non_blocking=True), weight.to(device,non_blocking=True)

        output = model(input1,input2)
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

    if y_pred.count(1) == 0:
        Pos_prec = 0
        Neg_prec = FN / (TN + FN)
    elif y_pred.count(0) == 0:
        Pos_prec = TP / (TP + FP)
        Neg_prec = 0
    else:
        Pos_prec = TP / (TP + FP)
        Neg_prec = FN / (TN + FN)

    return acc / data_len, total_loss / data_len, Pos_prec, Neg_prec



def test_model(model, test_data, device,model_str):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=1024)

    model.load_state_dict(torch.load(model_str, map_location=lambda storage, loc: storage))

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,addfeat,label, weight in test_loader:
        y_true.extend(label.data)
        data1,addfeat,label, weight = Variable(data1), Variable(addfeat),Variable(label), Variable(weight)
        data1,addfeat,label, weight = data1.to(device),Variable(addfeat),label.to(device), weight.to(device)

        output = model(data1,addfeat)
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


def train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test,model_name, pretrained_model):
    LR = 1e-4
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    device = torch.device("cuda")
    model = DualPeakClassifier(dim_model=256,n_heads=4,dim_feedforward=512,n_layers=4,dim_intensity=None,num_classes=2,dropout=0.3,max_len=200)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    if len(pretrained_model)>0:
        print("loading pretrained_model")
        model.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    #model.load_state_dict(torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    scaler = GradScaler("cuda")
    train_loader = Data.DataLoader(train_data, batch_size=128, num_workers=8, shuffle=True, pin_memory=True)
    for epoch in range(0, 80):
        start_time = time.time()
        model.train()
        batch_idx=0
        for input1, input2, y_batch, weight in train_loader:
            input1, input2, targets, weight = input1.to(device,non_blocking=True), input2.to(device,non_blocking=True), y_batch.to(device,non_blocking=True), weight.to(device,non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(input1,input2)  # forward computation
                loss = criterion(outputs, targets, weight)
            # backward propagation and update parameters
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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

    for i in range(0,80):
        test_model(model, test_data, device,'checkpoints/epoch'+str(i)+'.pt')


if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:m:p:t:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directories for spectra features and Label\n")
        print("-m\t Pre-trained model name\n")
        print("-p\t Output trained model name\n")
        sys.exit(1)
        start_time=time.time()
    input_directory=""
    model_name=""
    pretrained_model=""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directories for spectra features\n")
            print("-m\t ms2 format spectrum information\n")
            print("-p\t Output trained model name\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_directory=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-p"):
            pretrained_model=arg
    start = time.time()
    pct1_dir = os.path.join(input_directory, "pct1")
    pct2_dir = os.path.join(input_directory, "pct2")

    if os.path.isdir(pct1_dir) and os.path.isdir(pct2_dir):
        pct2_psms, pct2_features = pair_psm_and_feature_files(pct2_dir)
        pct1_psms, pct1_features = pair_psm_and_feature_files(pct1_dir)
        if len(pct2_psms) == 0 or len(pct1_psms) == 0:
            raise ValueError("Missing .tsv/.pkl pairs under pct1/pct2 directories.")

        X_pos, y_pos = readData(pct2_psms, pct2_features, force_label=1)
        X_neg, y_neg = readData(pct1_psms, pct1_features, force_label=0)

        (X_train_pos, y_train_pos), (X_val_pos, y_val_pos), (X_test_pos, y_test_pos) = split_list(X_pos, y_pos)
        (X_train_neg, y_train_neg), (X_val_neg, y_val_neg), (X_test_neg, y_test_neg) = split_list(X_neg, y_neg)

        X_train = X_train_pos + X_train_neg
        yweight_train = y_train_pos + y_train_neg
        X_val = X_val_pos + X_val_neg
        yweight_val = y_val_pos + y_val_neg
        X_test = X_test_pos + X_test_neg
        yweight_test = y_test_pos + y_test_neg
    else:
        psms, features = pair_psm_and_feature_files(input_directory)
        if len(psms) == 0:
            psms = sorted(glob.glob(input_directory + '/*tsv'))
            features = sorted(glob.glob(input_directory + '/*pkl'))
        L, Yweight = readData(psms, features)
        (X_train, yweight_train), (X_val, yweight_val), (X_test, yweight_test) = split_list(L, Yweight)

    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test, model_name, pretrained_model)
    print('done')
