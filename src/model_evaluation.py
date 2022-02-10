# 
import pandas as pd
import numpy as np
import preprocessing as pp
import visualize as vis
from eval import eval
from preprocessing import Meteo_DS as gMeteo_DS
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-d',  type=str,help='dataset type ("similar","year","season")')
parser.add_argument('-train_type',  type=str,help='dataset type ("train","no_pretrain","pretrain"',default='train')


args = parser.parse_args()

train_type = args.train_type

depth_areas = pp.lake_depth_areas_dict['Lake Mendota']

dataset_type = args.d
if train_type=='train':
  predictions_similar = list([])
  rsme_similar = list([])
  dates_similar = list([])
  for exper_n in range(2):
    exper_n +=1
    print(exper_n)

    meteo_path = 'data/pretrain/mendota_meteo.csv'
    test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_'+str(exper_n)+'.csv'
    depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
    ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
    model_path = 'model/model_train_'+dataset_type+'_exper_'+str(exper_n)+'.model'

    print(test_data_path)

    predictions, rsme, dates = eval(model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)
    predictions_similar.append([predictions])
    rsme_similar.append([rsme])
    dates_similar.append([dates])

  test_df = pd.DataFrame()
  for exper_n in range(2):
    exper_n +=1
    test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_'+str(exper_n)+'.csv'

    new_df = pd.read_csv(test_data_path,parse_dates=['date'])
    test_df = test_df.append(new_df)

  vis.plot_ts(predictions_similar, dates_similar, test_df=test_df,title='Predictions of model trained with '+dataset_type+' dataset on test dataset vs labels',labels=None,savepath='pictures',plotname='ts_'+dataset_type) 

elif train_type='no_pretrain':
  meteo_path = 'data/pretrain/mendota_meteo.csv'
  test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_1.csv'
  depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
  ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
  model_path = 'model/model_train_'+dataset_type+'_exper_1_no_pt.model'

  print(test_data_path)

  predictions, rsme, dates = eval(model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)
  
  print('RSME: ',rsme)
  
  test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_1.csv'

  test_df = pd.read_csv(test_data_path,parse_dates=['date'])
  
  vis.plot_ts(predictions_similar, dates_similar, test_df=test_df,title='Predictions of model (no pretraining) trained with '+dataset_type+' dataset on test dataset vs labels',labels=None,savepath='pictures',plotname='ts_'+dataset_type+'_no_pt_')
  
elif train_type='pretrain':
  meteo_path = 'data/pretrain/mendota_meteo.csv'
  test_data_path = 'data/predictions/me_predict_pb0.csv'
  depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
  ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
  model_path ='model/model_pretrain_pgdl_ec01_400_til2009.model'
  
  print(test_data_path)

  predictions, rsme, dates = eval(model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)
  
  print('RSME: ',rsme)
  
  test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_1.csv'

  new_df = pd.read_csv(test_data_path,parse_dates=['date'])
  
  vis.plot_depth_ts(prediction_df,dates,title='',test_data=False,savepath='pictures',plotname='dp_ts_pretrain')
  
  vis.plot_depth_ts(new_df,dates,title='',test_data=False,savepath='pictures',plotname='dp_ts_pretrain')
  
  
  
