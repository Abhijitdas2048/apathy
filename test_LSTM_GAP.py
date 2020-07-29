import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as sio
import torchvision.models as models

sys.path.append('..');
from utils.database.Pixelmap import GAP_database_apathy_fold;

from utils.model.model import LSTM_FC_Apathy_one_layer, LSTM_FC_Apathy_two_layer;
from utils.model.model import LSTM_FC_Apathy_one_layer_bi, LSTM_FC_Apathy_two_layer_bi;

###############################################################################################
##################################  Settings ##################################################
###############################################################################################
batch_size_num = 100;   ### batch size
epoch_num = 5;      ### epochs
learning_rate = 0.001;  ### learning rate
test_batch_size = 20;   ### batch size for test
log_file = '../log/test_LSTM_GAP_topk.txt';  ### log file, output training loss
Is_First_time = True;   ### Is train from begining

GRU_mode = 4;  ### 1.for one layer gru;  2 for two layer gru; 3 for bi-direction one layer gru; 4 for bi-directional two layer gru
instance_num = 50;    ### how many instances (clips) are used for each video
feature_length = 79;  ### feature length
MIL_k = 50;           ### if you use multi-instance learning, how many instancea are used for average pooling.
fold_num = 3;          ### how many folds for experiments in total;

feature_path = '../data/feature_AU_Lip/';  ## feature location
result_path = '../result/test_fold/';            ## where the results are put
model_path = '../model/test/';              ## where the models are put
fold_data_path = '../data/fold.mat';        ## where the fold.mat are put
################################################################################################

def train(train_loader, epoch):
    net.train();
    train_loss = 0;

    f = open(log_file, 'a+');
    f.write("epoch: {}\n".format(epoch));
    for batch_idx, (data, hr, idx) in enumerate(train_loader):
        # data = Variable(data.view(-1,feature_channel*feature_size));
        data = Variable(data);
        # target = Variable(target);
        hr = Variable(hr.view(-1));
        data, hr = data.cuda(), hr.cuda();

        output = net(data);

        # print(output)
        # print(hr)
        # loss, loss1, loss2 = lossfunc(output, target, hr);
        loss = lossfunc(output, hr);

        train_loss += loss.data[0];

        optimizer.zero_grad()
        loss.backward()
        optimizer.step();

        f.write("train_loss: {}\n".format(loss.data[0]));

    print("=====================");
    print("epoch: {}".format(epoch));
    print('Train loss: {:.8f}'.format(train_loss));

    f.write("=====================");
    f.write("epoch: {}".format(epoch));
    f.write('Train loss: {:.8f}'.format(train_loss));

def test(test_loader, epoch, fold):
    net.eval()
    test_loss = 0;

    gt = np.array([]);
    result = np.array([]);
    idx_all = np.array([]);
    for data, hr, idx in test_loader:
        data = Variable(data);
        hr = Variable(hr.view(-1));

        data, hr = data.cuda(), hr.cuda();
        output = net(data)

        loss = lossfunc(output, hr);

        test_loss += loss.data[0];

        gt = np.append(gt, hr.data.cpu().numpy());
        result = np.append(result, output.data.cpu().numpy());
        idx_all = np.append(idx_all, idx.numpy());

    result_name = result_path + 'fold' + str(fold) + '_result' + str(epoch) + '.mat';
    sio.savemat(result_name, dict(idx = idx_all, result=result, gt=gt));
    best_result = test_loss;

    f = open(log_file, 'a+');
    print('Test set: Average loss: {:.8f}'.format(test_loss));
    f.write('Test set: Average loss: {:.8f}\n'.format(test_loss));

    return best_result;

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_path + 'model_best.pth.tar')

for fold in range(1, fold_num+1):
# Data Loader
    train_dataset = GAP_database_apathy_fold(root_dir = feature_path, feature_num = instance_num, feature_length = feature_length, Training = True,
                                             fold_path = fold_data_path, fold_num = fold_num, fold = fold);
    train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                              shuffle=True, num_workers=2);

    test_dataset = GAP_database_apathy_fold(root_dir = feature_path,  feature_num = instance_num, feature_length = feature_length, Training= False,
                                            fold_path = fold_data_path, fold_num = fold_num, fold = fold);
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=True, num_workers=2);

    if GRU_mode == 1:
        net = LSTM_FC_Apathy_one_layer(feature_length = feature_length, instance_num = instance_num, MIL_k = MIL_k);
    elif GRU_mode == 2:
        net = LSTM_FC_Apathy_two_layer(feature_length = feature_length, instance_num = instance_num, MIL_k = MIL_k);
    elif GRU_mode == 3:
        net = LSTM_FC_Apathy_one_layer_bi(feature_length = feature_length, instance_num = instance_num, MIL_k = MIL_k);
    elif GRU_mode == 4:
        net = LSTM_FC_Apathy_one_layer_bi(feature_length = feature_length, instance_num = instance_num, MIL_k = MIL_k);

    net.cuda();

    print(net);

    lossfunc = nn.CrossEntropyLoss();

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all cnn parameters

    if not Is_First_time:
        temp = torch.load(model_path);
        optimizer.load_state_dict(temp['optimizer']);

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5);

    best_prec1 = 9999999;
    f = open(log_file, 'a+');
    f.write('***********************************\n');
    if Is_First_time:
        begin_epoch = 1;
    else:
        temp = torch.load(model_path);
        begin_epoch = temp['epoch'];

    for epoch in range(begin_epoch, epoch_num + 1):
        # scheduler.step();
        train(train_loader, epoch)
        prec1 = test(test_loader, epoch, fold);

        is_best = prec1 < best_prec1;
        best_prec1 = min(prec1, best_prec1)

        if epoch% 2 == 0:
            save_name = model_path + '/fold' + str(fold) + '_checkpoint' + str(epoch) + '.pth.tar';
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_name);

    f.write('***********************************\n');
