# Train with labels from sensor
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse

import preprocessing as pp
from preprocessing import Meteo_DS as gMeteo_DS
from preprocessing import SlidingWindow as gSlidingWindow
import train_functions as tfunc
from model import GeneralLSTM

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-d',  type=str,help='dataset type ("similar","year","season")')
parser.add_argument('-pretrain',  type=bool,help='dataset type ("similar","year","season")',default=False)

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

### Hyperparameter ###

epochs = 150

### data set type ###
dataset_type = args.d
pretrain = args.pretrain

for i in range(2):
  exper_n = i+1
  predict_fname = 'me_'+dataset_type+'_exper_'+str(exper_n)+'.csv'
  meteo_path = 'data/pretrain/mendota_meteo.csv'
  predict_pb0_path = 'data/train_splitted/'+predict_fname
  mendota_depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
  ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
  train_dataset = gMeteo_DS(meteo_path,predict_pb0_path,mendota_depth_areas,ice_csv_path=ice_flags_path,transform=True)

  train_dl = DataLoader(gSlidingWindow(train_dataset.Xt,353,int(353/2),train_dataset.labels,phys_data=train_dataset.X, dates=train_dataset.dates ),shuffle=False)

  # init model

  input_size = train_dl.dataset.x.size()[-1]
  batch_size = train_dl.dataset.x.size()[0]
  model = GeneralLSTM(input_size, state_size,batch_size,device, num_layers=1)
  criterion = torch.nn.MSELoss()#
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  model.to(device)
  
  if pretrain:
    #load pre-trianed LSTM
    pretrain_model_path = 'model/model_pretrain_pgdl_ec01_400_til2009.model'
    pretrain_dict = torch.load(pretrain_model_path)['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)
  
  save_path = 'model'
  file_name = 'similar_'+dataset_type+'_exper_'+str(exper_n)+'.model'
  if not pretrain:
    file_name = file_name.split('.')[0]+'_no_pt'+file_name.split('.')[1]
    
  log_file_name = file_name.split('.')[0]+'.log'
  log_path = os.path.join(save_path,log_file_name)
  save_path = os.path.join(save_path,file_name)
  
  tfunc.train_ec(model, train_dl, optimizer, criterion, epochs, torch.Tensor(mendota_depth_areas.astype(np.float32)), \
               device, ec_lambda=0.1, dc_lambda=0.0, lambda1=0.0, ec_threshold=36, begin_loss_ind=50, grad_clip=1.0, save_path=save_path, verbose=True)
  
  

