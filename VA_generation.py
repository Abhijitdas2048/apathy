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
from utils.database.Pixelmap_VA import AffectNet_VA, Test_VA;



###############################################################################################
##################################  Settings ##################################################
###############################################################################################
test_batch_size = 5;   ### batch size for test
log_file = '../log/test_LSTM_GAP_topk.txt';  ### log file, output training loss

image_all_dir = '../data/video/' ### where the cropped face images are put
result_all_name = '../data/';         ## where the results are put
model_path = '../model/test_VA/checkpoint1.pth.tar';           ## where the VA models are put
################################################################################################

net = models.resnet50(pretrained = False);
net.fc = nn.Linear(2048, 2);

temp = torch.load(model_path);
net.load_state_dict(temp['state_dict']);

net.cuda();
print(net);

test_transforms = transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.55744, 0.42414907, 0.36356255), (0.26696834, 0.22972111, 0.21915098))
        ])

def test(test_loader, result_name):
    net.eval()

    result = np.array([]);
    idx_all = np.array([]);
    for data, idx in test_loader:
        data = Variable(data);
        data = data.cuda();
        output = net(data)
        result = np.append(result, output.data.cpu().numpy());
        idx_all = np.append(idx_all, idx.numpy());

    sio.savemat(result_name, dict(idx = idx_all, result=result));


for dirs in os.listdir(image_all_dir):
    image_dir = image_all_dir + dirs + '/';  ### where the cropped face images are put
    result_name = result_all_name + dirs + '.mat';  ## where the results are put
    print(image_dir);
    print(result_name);

    # Data Loader
    test_dataset = Test_VA(root = image_dir, transform=test_transforms);
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                              shuffle=False, num_workers=2);

    test(test_loader, result_name);
