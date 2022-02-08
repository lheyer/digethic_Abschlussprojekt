# for model evaluation
import torch
import numpy as np
import pandas as pd
import os
import preprocessing as pp
from preprocessing import Meteo_DS as gMeteo_DS
from model import GeneralLSTM

def eval(model_path, meteo_path, test_data_path, depth_areas, ice_flags_path, state_size=20, begin_loss_ind=50):


  
  test_dataset = gMeteo_DS(meteo_path,test_data_path,depth_areas,ice_csv_path=ice_flags_path,transform=True,testing=True)

  dates = test_dataset.XY.date

  # initialize model
  input_size = torch.Tensor(test_dataset.Xt).size()[-1]
  batch_size = torch.Tensor(test_dataset.Xt).size()[-1]
  model = GeneralLSTM(input_size, state_size,batch_size,device, num_layers=1)
  criterion = torch.nn.MSELoss()#
  #optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  model.to(device)

  #load trained model
  pretrain_dict = torch.load(model_path)['state_dict']
  model_dict = model.state_dict()
  pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
  model_dict.update(pretrain_dict)
  model.load_state_dict(pretrain_dict)

  model.eval()

  with torch.no_grad(): #disable gradient
    model.hidden = model.init_hidden(batch_size=torch.Tensor(test_dataset.Xt).size()[0])
    h_state = None
    preds,h_state = model(torch.Tensor(test_dataset.Xt))
    #print(preds.size())
    #print(torch.Tensor(train_dataset.labels).size())
    loss_preds = preds[:, begin_loss_ind:,0]
    loss_Y = torch.Tensor(test_dataset.labels)[:, begin_loss_ind:]
    d_loss = criterion(loss_preds[~torch.isnan(loss_Y)], loss_Y[~torch.isnan(loss_Y)])
    print("rmse=", np.sqrt(d_loss))

    depth_labels = ['temp_'+str(i) for i in test_dataset.XY.iloc[0].depths]
    preds_df = pd.DataFrame(preds.detach().numpy()[:,:,0].T,columns=depth_labels)

  return preds_df, np.sqrt(d_loss), dates
