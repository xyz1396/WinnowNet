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
import sys
import getopt
import pickle
from WinnowNet_Att import pad_control, DualPeakClassifier

threshold=0.9

def readData(feature):
    L = []
    Yweight = []
    with open(feature,'rb') as f:
        D_features=pickle.load(f)
    for j in D_features.keys():
        L.append(D_features[j])

    del D_features
    return L


class DefineDataset(Data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xspectra1 = pad_control(self.X[idx][0],200)
        xspectra2 = pad_control(self.X[idx][1],200)
        xspectra1 = torch.FloatTensor(xspectra1)
        xspectra2 = torch.FloatTensor(xspectra2)

        return xspectra1, xspectra2


def get_time_diff(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def test_model(model, model_name, test_data, device):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=32)
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=True))
    y_pred_prob = []
    for data1, data2 in test_loader:
        data1,data2 = Variable(data1), Variable(data2)
        data1,data2 = data1.to(device),data2.to(device)
        output = model(data1,data2)
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred_prob.extend(pred_prob[:, 1].tolist())

    return y_pred_prob




if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t input representation file name\n")
        print("-m\t Pre-trained model name\n")
        print("-o\t Output for PSM rescoring\n")
        sys.exit(1)
        start_time=time.time()
    input_file=""
    model_name=""
    output_file=""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t input representation file name\n")
            print("-m\t Pre-trained model name\n")
            print("-o\t Output for PSM rescoring\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_file=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-o"):
            output_file=arg
    start = time.time()
    L  = readData(input_file)
    test_data = DefineDataset(L)
    device = torch.device("cuda")
    model = DualPeakClassifier(dim_model=256,n_heads=4,dim_feedforward=512,n_layers=4,dim_intensity=None,num_classes=2,dropout=0.3,max_len=200)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    y_pred_prob=test_model(model, model_name, test_data, device)
    with open(output_file,'w') as f:
        for line in y_pred_prob:
            f.write(str(line))
            f.write('\n')

    print('done')
