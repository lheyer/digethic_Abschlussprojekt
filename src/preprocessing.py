# Preprocessing of pretraining and training data

import datetime as dt

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

#########################
### Constants & Dicts ###
#########################

lake_id_dict= {'Lake Mendota': 'nhd_13293262',
               'Sparkling Lake': 'nhd_13344210'}
seconds_per_timestep = 86400

lake_area_dict = {'Lake Mendota': 39416000,
               'Sparkling Lake': 635356}
lake_depth_areas_dict = dict()
lake_depth_areas_dict['Lake Mendota'] = np.array([\
        39865825,38308175,38308175,35178625,35178625,33403850,31530150,31530150,30154150,30154150,29022000,\
        29022000,28063625,28063625,27501875,26744500,26744500,26084050,26084050,25310550,24685650,24685650,\
        23789125,23789125,22829450,22829450,21563875,21563875,20081675,18989925,18989925,17240525,17240525,\
        15659325,14100275,14100275,12271400,12271400,9962525,9962525,7777250,7777250,5956775,4039800,4039800,\
        2560125,2560125,820925,820925,216125])




#########################
### Dataset classes ###
#########################

class Meteo_DS(Dataset):
  """Meteorological Dataset
    for serving the meteorological
  """


  def __init__(self, meteo_csv_path: str, pred_csv_path: str, depth_areas: list, ice_csv_path =None, transform=None):
    """
    Args:
        meteo_csv_path (string): Path to the csv file with meteorological data
        pred_csv_path (string): Path to the csv file with temperature predictions
        ice_csv_path (string, optional): Path to the csv file with ice flags
        depth_areas (list): List with depth areas (float) for selected lake
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.X = pd.read_csv(meteo_csv_path,parse_dates=['date'])
    self.Xcols = self.X.columns
    self.Y = pd.read_csv(pred_csv_path,parse_dates=['date'])
    self.Ycols = self.Y.columns
    
    self.transform = transform

    self.depth_areas = np.array(depth_areas)
    self.n_depths = self.depth_areas.size
    self.lake_depths = np.array([i*0.5 for i in range(self.n_depths)])

    self.XY = pd.merge(self.X,self.Y,on='date')
    self.XY = self.XY.sort_values(by='date').reset_index(drop=True)

    self.n_steps = self.XY.shape[0]

    self.XY.loc[:,'tm_yday'] = self.XY.date.dt.dayofyear
    self.XY.loc[:,'depths'] = self.XY.tm_yday.apply(lambda x : self.lake_depths)

    self.phys_list = ['tm_yday','ShortWave','LongWave',\
                      'AirTemp','RelHum','WindSpeed','Rain','Snow']

    if ice_csv_path is not None:
      self.ice_csv_path = ice_csv_path
      self.XY = self.get_ice_mask()
      self.phys_list.append('ice')
      print(self.phys_list)

    
    self.X = self.XY.explode('depths').sort_values(['depths','date'])[self.phys_list]\
                                        .to_numpy().reshape(self.n_depths,-1,9)

    # date vector                                  
    helper = np.vectorize(lambda x: dt.date.toordinal(pd.Timestamp(x).to_pydatetime()))
    self.dates = helper(self.XY.date.values)


    if self.transform:
      self.Xt = self.scale(self.X[:,:,:-1])
      print(np.max(self.Xt))
      print(np.min(self.Xt))
   
    self.labels = self.get_labels(self.XY[self.Ycols])

  def scale(self,X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1,8))
    X = X.reshape(self.n_depths,-1,8)
    return X

  def get_labels(self,Y):
    
    Y_labels = Y.drop('date', axis=1).T #.values
    Y_labels.loc[:,'depth'] = Y_labels.index #.apply(lambda x: float(x.split('_')[-1]))
    Y_labels.depth = Y_labels.depth.apply(lambda x: float(x.split('_')[-1]))
    Y_labels = Y_labels.sort_values('depth')
    Y_labels = Y_labels.drop(columns=['depth']).iloc[:n_depths].to_numpy()
    return Y_labels

  def get_ice_mask(self):
    # Ice mask (so we don't penalize energy imbalance on days with ice)
    df = pd.read_csv(self.ice_csv_path,parse_dates=['date'])
    
    # bool to [0,1]
    df.ice = df.ice.apply(lambda x : 1 if x else 0)
    self.XY = pd.merge(self.XY, df,on='date')

    return self.XY


  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):

    X = self.X.astype('float') #.reshape(-1,7)
    X = X[index]

    return X
  
  
class SlidingWindow(Dataset):
  """Sliding Window Class for data sorted by (Variable,time)"""

  def __init__(self, features : np.array, window : int, step : int, labels : np.array , phys_data=None, dates=None ,label_window=1):
    """
    Args:
        features (numpy.array): (n_depths, time_steps,features) array including features 
        window (int): data window for feature data
        step (int): stride width
        labels (numpy.array): (n_depths, time_steps) array including labels
        phys_data (numpy.array): (n_depths, time_steps, features+mask) array including non-normalized meteorological data & ice mask
    """
    
    #assert len(x)==len(y)
    self.x = torch.Tensor(features.astype(np.float32))

    self.y = torch.Tensor(labels.astype(np.float32))#)

    if (phys_data is not None) and (dates is not None): #isinstance(phys_data, np.array):
      self.phys_data = torch.Tensor(phys_data.astype(np.float32))
      self.dates = torch.Tensor(dates.astype(int))
    else:
      self.phys_data = None

    self.window = int(window)
    self.step = step
    self.label_window = label_window
    #self.xlen = int(self.x.shape[1])
    self.xlen = self.x.size(1)
    self.start_index = 0



  def __getitem__(self,idx):
    x_start = idx*self.step
    x_end = x_start+self.window
    y_start = x_end
    y_end = x_end+self.label_window

    if isinstance(self.phys_data,torch.Tensor):
      #print(' found phys data')
      return self.x[:,x_start:x_end,:], self.phys_data[:,x_start:x_end,:], self.dates[x_start:x_end], self.y[:,x_start:x_end] #[:,y_start:y_end]
    else:
      #print('did not find phys data')
      return self.x[:,x_start:x_end,:], self.y[:,y_start:y_end]
      

  def __len__(self):
    #print(self.xlen)
    return int(np.floor((self.xlen- self.label_window)/self.step) -1 ) #-self.label_window)%self.step#(self.window + self.label_window + self.step)

  def __iter__(self):  

    return iter(range(self.start_index,self.__len__()))
  
  
  
