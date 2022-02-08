# 
import preprocessing as pp
import visualize as vis
from eval import eval
from preprocessing import Meteo_DS as gMeteo_DS
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-d',  type=str,
                    help='dataset type ("similar","year","season"))


args = parser.parse_args()


depth_areas = pp.lake_depth_areas_dict['Lake Mendota']

dataset_type = args.d
predictions_similar = list([])
rsme_similar = list([])
dates_similar = list([])
for exper_n in range(2):
  exper_n +=1
  print(exper_n)

  meteo_path = '/content/drive/MyDrive/digethic_Project/data/data_ReadEtAl/pretrain/mendota_meteo.csv'
  test_data_path = '/content/drive/MyDrive/digethic_Project/data/data_ReadEtAl/test_splitted/me_test_'+dataset_type+'_exper_'+str(exper_n)+'.csv'
  depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
  ice_flags_path = '/content/drive/MyDrive/digethic_Project/data/example_input/raw_data/mendota_pretrainer_ice_flags.csv'
  model_path = '/content/drive/MyDrive/digethic_Project/data/tmp/train/model/model_train_'+dataset_type+'_exper_'+str(exper_n)+'.model'

  print(test_data_path)

  predictions, rsme, dates = eval(model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)
  predictions_similar.append([predictions])
  rsme_similar.append([rsme])
  dates_similar.append([dates])
                    
test_df = pd.DataFrame()
for exper_n in range(2):
  exper_n +=1
  test_data_path = '/content/drive/MyDrive/digethic_Project/data/data_ReadEtAl/test_splitted/me_test_'+dataset_type+'_exper_'+str(exper_n)+'.csv'

  new_df = pd.read_csv(test_data_path,parse_dates=['date'])
  test_df = test_df.append(new_df)
                    
vis.plot_ts(predictions_similar, dates_similar, test_df=test_df,title='Predictions of model trained with '+dataset_type+' dataset on test dataset vs labels',labels=None,savepath='pictures',plotname=dataset_type) 

