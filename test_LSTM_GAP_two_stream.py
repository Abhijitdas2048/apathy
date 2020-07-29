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
from utils.database.Pixelmap import GAP_database_apathy_two_stream_fold;

from utils.model.model import LSTM_FC_Apathy_two_stream, LSTM_FC_Apathy_two_stream_bi;

###############################################################################################
##################################  Settings ##################################################
###############################################################################################
batch_size_num = 100;   ### batch size
epoch_num = 5;      ### epochs
learning_rate = 0.001;  ### learning rate
test_batch_size = 20;   ### batch size for test
log_file = '../log/test_LSTM_GAP_topk.txt';  ### log file, output training loss
Is_First_time = True;   ### Is train from begining

instance_num = 50;    ### how many instances (clips) are used for each video
MIL_k = 50;           ### if you use multi-instance learning, how many instancea are used for average pooling.
fold_num = 3;          ### how many folds for experiments in total;
Bi_direction = True;   #### whether bi direction

### first stream
feature_length1 = 79;  ### feature length
feature_path1 = '../data/feature_AU_Lip/';  ## feature location
#### second stream
feature_length2 = 117;  ### feature length
feature_path2 = '../data/feature_GAP/';  ## feature location

result_path = '../result/test/';            ## where the results are put
model_path = '../model/test/';              ## where the models are put
fold_data_path = '../data/fold.mat';        ## where the fold.mat are put
################################################################################################

def train(train_loader, epoch):
    net.train();
    train_loss = 0;

    f = open(log_file, 'a+');
    f.write("epoch: {}\n".format(epoch));
    for batch_idx, (data1, data2, hr, idx) in enumerate(train_loader):
        # data = Variable(data.view(-1,feature_channel*feature_size));
        data1 = Variable(data1);
        data2 = Variable(data2);
        # target = Variable(target);
        hr = Variable(hr.view(-1));
        data1, data2, hr = data1.cuda(), data2.cuda(), hr.cuda();

        output = net(data1, data2);


        # print(output);
        # print(hr)
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
    for data1, data2, hr, idx in test_loader:
        data1 = Variable(data1);
        data2 = Variable(data2);
        hr = Variable(hr.view(-1));

        data1, data2, hr = data1.cuda(), data2.cuda(), hr.cuda();
        output = net(data1, data2)

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
    train_dataset = GAP_database_apathy_two_stream_fold(root_dir1 = feature_path1, root_dir2 = feature_path2, feature_length1 = feature_length1, feature_length2 = feature_length2,
                                                        feature_num = instance_num, Training = True, fold_path = fold_data_path, fold_num = fold_num, fold = fold);
    train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                              shuffle=True, num_workers=2);

    test_dataset = GAP_database_apathy_two_stream_fold(root_dir1 = feature_path1, root_dir2 = feature_path2, feature_length1 = feature_length1, feature_length2 = feature_length2,
                                                        feature_num = instance_num, Training = False, fold_path = fold_data_path, fold_num = fold_num, fold = fold);
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=True, num_workers=2);

    if Bi_direction:
        net = LSTM_FC_Apathy_two_stream_bi(feature_length1=feature_length1, feature_length2=feature_length2,
                                        instance_num=instance_num, MIL_k=MIL_k);
    else:
        net = LSTM_FC_Apathy_two_stream(feature_length1 = feature_length1, feature_length2 = feature_length2,
                                        instance_num = instance_num, MIL_k = MIL_k);

    net.cuda();

    print(net);

    lossfunc = nn.CrossEntropyLoss();

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all cnn parameters

    if not Is_First_time:
        temp = torch.load(model_path);
        optimizer.load_state_dict(temp['optimizer']);

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

        if epoch % 2 == 0:
            save_name = model_path + '/fold' + str(fold) + '_checkpoint' + str(epoch) + '.pth.tar';
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_name);

    f.write('***********************************\n');
