# functions for data visualisation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join


# def stack_depth_cols_test_labels(df):
#   new_df = pd.DataFrame(columns=['date','depth','temp'])
#   for col in df.columns:
#     if col in ['date','exper_n','exper_id']:
#       pass
#     else:
#       d = col.split('_')[1]
#       #print(d)

#       tmp_df = pd.DataFrame(columns=['date','depth','temp'])
#       tmp_df.temp = df[col]
#       #print(tmp_df.shape)
#       tmp_df.date = df.index
#       tmp_df.depth = float(col.split('_')[1])
#       new_df = new_df.append(tmp_df)
#   return new_df


def stack_depth_cols_preds(df,dates):
  new_df = pd.DataFrame(columns=['date','depth','temp'])
  for col in df.columns:
    d = col.split('_')[1]
    #print(d)
    #print(train_dataset.XY[train_dataset.XY.depth==d].index.shape)

    tmp_df = pd.DataFrame(columns=['date','depth','temp'])
    tmp_df.temp = df[col]
    #print(tmp_df.shape)
    tmp_df.date = dates
    tmp_df.depth = float(col.split('_')[1])
    new_df = new_df.append(tmp_df)

  return new_df

def plot_depth_ts(prediction_df,dates,title='',test_data=False,savepath=None,plotname=''):

  # new_df = pd.DataFrame(columns=['date','depth','temp'])
  # for col in prediction_df.columns:
  #   d = col.split('_')[1]
  #   #print(d)
  #   #print(train_dataset.XY[train_dataset.XY.depth==d].index.shape)

  #   tmp_df = pd.DataFrame(columns=['date','depth','temp'])
  #   tmp_df.temp = prediction_df[col]
  #   #print(tmp_df.shape)
  #   tmp_df.date = dates
  #   tmp_df.depth = float(col.split('_')[1])
  #   new_df = new_df.append(tmp_df)

  if test_data:
    new_df = prediction_df
  else:
    new_df = stack_depth_cols_preds(prediction_df,dates)

  plt.figure(figsize=(30,5))

  plt.title(title,fontsize=20)

  ax = sns.scatterplot(data=new_df.set_index('date'), x="date", y="depth",hue="temp",palette='viridis',s=60,marker='s',linewidth=0)
  ax.yaxis.label.set_size(20)
  ax.tick_params(axis='y', labelsize=16)
  ax.xaxis.label.set_size(20)
  ax.tick_params(axis='x', labelsize=16)
  plt.legend(loc='best', prop={'size': 16},bbox_to_anchor=(0.55, 0.5, 0.5, 0.5))
  
  if savepath is not None:
    plt_name = plotname + 'plot_depth_ts.png'
    savepath = join(savepath,plt_name)

    plt.savefig(savepath)
    
  return 
    
    



def plot_ts(prediction_df_list, pred_date_list, test_df=None,title=None,labels=None,savepath=None,plotname=''):

  plt.figure(figsize=(30,5))
  if title is None:
    plt.title('Predictions of pretrained model on training data and labels, all averaged over depth',fontsize=20)
  else:
    plt.title(title,fontsize=20)

  if labels is None:
    #print(len(prediction_df_list))
    labels = ['sampel '+str(i) for i in range(len(prediction_df_list))]
    print(labels)

  ind = 0
  for df,dates in zip(prediction_df_list,pred_date_list):
    
    new_df = stack_depth_cols_preds(df[0],dates[0])

    ax = new_df.groupby('date').mean().temp.plot(label=labels[ind],marker='.',ls='')
    ind+=1

  if test_df is not None:
    #new_df = stack_depth_cols_test_labels(test_df)
    test_df.groupby('date').mean().temp.plot(label='labels',marker='.',ls='')

  ax.yaxis.label.set_size(20)
  ax.tick_params(axis='y', labelsize=16)
  ax.xaxis.label.set_size(20)
  ax.tick_params(axis='x', labelsize=16)
  plt.legend(loc='best', prop={'size': 16})
  plt.ylabel('T [°C]')
  plt.xlabel('date')
  
  if savepath is not None:
    plt_name = plotname + 'plot_ts.png'
    savepath = join(savepath,plt_name)
   
    plt.savefig(savepath)
    
  return
