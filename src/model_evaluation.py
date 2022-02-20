#
import pandas as pd
import numpy as np
import preprocessing as pp
import visualize as vis
import seaborn as sns
import matplotlib.pyplot as plt
from eval import eval
from preprocessing import Meteo_DS as gMeteo_DS
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument(
    '-d',  type=str, help='dataset type ("similar","year","season")')
parser.add_argument('-train_type',  type=str,
                    help='dataset type ("train","no_pretrain","pretrain"', default='train')


args = parser.parse_args()

train_type = args.train_type

depth_areas = pp.lake_depth_areas_dict['Lake Mendota']

dataset_type = args.d
if train_type == 'train':
    predictions_similar = list([])
    rsme_similar = list([])
    dates_similar = list([])
    for exper_n in range(2):
        exper_n += 1
        print(exper_n)

        meteo_path = 'data/pretrain/mendota_meteo.csv'
        test_data_path = 'data/test_splitted/me_test_' + \
            dataset_type+'_exper_'+str(exper_n)+'.csv'
        depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
        ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
        model_path = 'model/model_train_' + \
            dataset_type+'_exper_'+str(exper_n)+'.model'

        print(test_data_path)

        predictions, rsme, dates = eval(
            model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)
        predictions_similar.append(predictions)
        rsme_similar.append(rsme)
        dates_similar.append(dates)

    test_df = pd.DataFrame()
    for exper_n in range(2):
        exper_n += 1
        test_data_path = 'data/test_splitted/me_test_' + \
            dataset_type+'_exper_'+str(exper_n)+'.csv'

        new_df = pd.read_csv(test_data_path, parse_dates=['date'])
        # test_df = test_df.append(new_df)
        test_df = pd.concat([test_df, new_df], axis=0)

    vis.plot_ts(predictions_similar, dates_similar, test_df=test_df, title='Predictions of model trained with ' +
                dataset_type+' dataset on test dataset vs labels', labels=None, savepath='pictures', plotname='ts_'+dataset_type)

elif train_type == 'no_pretrain':
    meteo_path = 'data/pretrain/mendota_meteo.csv'

    depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
    ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
    predictions_similar = list([])
    rsme_similar = list([])
    dates_similar = list([])
    test_df = pd.DataFrame()
    for exper_n in range(2):
        exper_n += 1
        print('exper_n: ', exper_n)
        model_path = 'model/model_train_'+dataset_type + \
            '_exper_'+str(exper_n)+'_no_pt.model'
        test_data_path = 'data/test_splitted/me_test_' + \
            dataset_type+'_exper_'+str(exper_n)+'.csv'

        print('test_data path: ', test_data_path)

        predictions, rsme, dates = eval(
            model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)

        predictions_similar.append(predictions)
        rsme_similar.append(rsme)
        dates_similar.append(dates)
        print('RSME no pretrain sample '+str(exper_n)+': ', rsme)

        #test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_1.csv'

        #test_df = pd.read_csv(test_data_path, parse_dates=['date'])
        new_df = pd.read_csv(test_data_path, parse_dates=['date'])
        # test_df = test_df.append(new_df)
        test_df = pd.concat([test_df, new_df], axis=0)

    vis.plot_ts(predictions_similar, dates_similar, test_df=test_df, title='Predictions of model (no pretraining) trained with ' +
                dataset_type+' dataset on test dataset vs labels', labels=None, savepath='pictures', plotname='ts_'+dataset_type+'_no_pt_')

elif train_type == 'pretrain':

    # Evaluate model on training data
    meteo_path = 'data/pretrain/mendota_meteo.csv'
    test_data_path = 'data/predictions/me_predict_pb0.csv'
    depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
    ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
    model_path = 'model/model_pretrain_pgdl_ec01_400_til2009.model'

    print(test_data_path)

    predictions, rsme, dates = eval(
        model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)

    print('RSME of '+model_path+' (on training data): ', rsme)

    #test_data_path = 'data/test_splitted/me_test_'+dataset_type+'_exper_1.csv'

    new_df = pd.read_csv(test_data_path, parse_dates=['date'])

    vis.plot_depth_ts(predictions, dates, title='', test_data=False,
                      savepath='pictures', plotname='pretrain')

    vis.plot_depth_ts(new_df, new_df.date.values, title='', test_data=False,
                      savepath='pictures', plotname='pretrain_labels')

    model_path = 'model/model_pretrain_pgdl_ec00_400_til2009.model'

    print(test_data_path)

    predictions, rsme, dates = eval(
        model_path, meteo_path, test_data_path, depth_areas, ice_flags_path)

    print('RSME of '+model_path+' (on training data): ', rsme)
    # Evaluate model on test data
    time_slice = ['2009-01-01', '2020-01-01']
    meteo_path = 'data/pretrain/mendota_meteo.csv'
    test_data_path = 'data/predictions/me_predict_pb0.csv'
    depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
    ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'
    model_path = 'model/model_pretrain_pgdl_ec01_400_til2009.model'

    print(test_data_path)

    pred_list = list([])
    rsme_list = list([])
    date_list = list([])

    predictions, rsme, dates = eval(
        model_path, meteo_path, test_data_path, depth_areas, ice_flags_path, time_slice=time_slice)

    print('RSME of \n'+model_path+' (on test data): ', rsme)
    pred_list.append(predictions)
    rsme_list.append(rsme)
    date_list.append(dates.values)

    # now rnn model (ec_lambda=0.0)

    model_path = 'model/model_pretrain_pgdl_ec00_400_til2009.model'

    predictions, rsme, dates = eval(
        model_path, meteo_path, test_data_path, depth_areas, ice_flags_path, time_slice=time_slice)

    print('RSME of \n'+model_path+' (on test data): ', rsme)
    pred_list.append(predictions)
    rsme_list.append(rsme)
    date_list.append(dates.values)

    test_df = pd.read_csv(test_data_path, parse_dates=['date'])
    test_df = test_df[(test_df.date > time_slice[0]) &
                      (test_df.date < time_slice[1])]

    # print(pred_list)
    # print(test_df)
    print(date_list)

    vis.plot_ts(pred_list, date_list, test_df=test_df, title='Predictions of model trained with and without energy conservation loss on test dataset vs labels',
                savepath='pictures', plotname='ts_pretraining_on_test', labels=['ec_lambda=0.1', 'ec_lambda=0.0'])

    sns.set_theme()

    plt.figure(figsize=(30, 20))

    pgdl = vis.stack_depth_cols_preds(pred_list[0], date_list[0])
    pgdl = pgdl.groupby('date').mean()
    pgdl.loc[:, 'model/label'] = 'lambda_ec=0.1'
    pgdl.loc[:, 'date'] = pgdl.index

    rnn = vis.stack_depth_cols_preds(pred_list[1], date_list[1])
    rnn = rnn.groupby('date').mean()
    rnn.loc[:, 'model/label'] = 'lambda_ec=0.0'
    rnn.loc[:, 'date'] = rnn.index

    labels = vis.stack_depth_cols_preds(test_df, test_df.date.values)
    labels = labels.groupby('date').mean()
    labels.loc[:, 'model/label'] = 'labels'
    labels.loc[:, 'date'] = labels.index

    sns.displot(pd.concat([pgdl, rnn, labels], ignore_index=True),
                x='temp', hue='model/label', stat='density', aspect=1.5)

    plt.xlabel('Temperature [°C]')
    plt.savefig('pictures/pretrain_model_dist.png')
