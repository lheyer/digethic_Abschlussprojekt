from unicodedata import decimal
import numpy as np


try:
    from preprocessing import Meteo_DS, SlidingWindow, lake_depth_areas_dict
except ImportError:
    import sys
    sys.path.insert(0, '../src')
    sys.path.insert(1, 'src')
    from preprocessing import Meteo_DS, SlidingWindow, lake_depth_areas_dict


##############
# Data paths #
##############

mendota_meteo_path = '../data/pretrain/mendota_meteo.csv'
predict_pb0_path = '../data/predictions/me_predict_pb0.csv'
ice_flags_path = '../data/pretrain/mendota_pretrainer_ice_flags.csv'

mendota_depth_areas = lake_depth_areas_dict['Lake Mendota']


##################
# Test functions #
##################

def test_MeteoDS():

    df = Meteo_DS(mendota_meteo_path,
                  predict_pb0_path,
                  mendota_depth_areas,
                  ice_csv_path=ice_flags_path,
                  time_slice=['2008-01-01', '2008-12-31'],
                  transform=True)

    print('test scaled input:')
    assert df.Xt.min() >= 0
    print('minimum of tranformed array: ', df.Xt.min())
    print('maximum of tranformed array: ', df.Xt.max().round(decimals=5))
    assert df.Xt.max().round(decimals=5) <= 1

    assert df.Xt.shape[0] == 50
    assert df.Xt.shape[2] == 9
    assert len(df.Xt.shape) == 3

    print('test unscaled physical input for energy conservation calculation:')

    assert set(np.unique(df.X[:, :, 9])) == {0, 1}

    assert df.X.shape[0] == 50
    assert df.X.shape[2] == 10

    print('test labels')
    assert df.labels.shape[0] == 50
    assert len(df.labels.shape) == 2
    assert df.labels.min() > -20
    assert df.labels.max() < 50


def test_SlidingWindow():

    df = Meteo_DS(mendota_meteo_path,
                  predict_pb0_path,
                  mendota_depth_areas,
                  ice_csv_path=ice_flags_path,
                  time_slice=['2008-01-01', '2008-12-31'],
                  transform=True)

    window = 353
    stride = int(window/2)

    df = SlidingWindow(df.Xt, window, stride, df.labels,
                       phys_data=df.X, dates=df.dates)

    last_index = len(df) - 1

    assert df[last_index][0].shape[0] == 50
    assert df[last_index][0].shape[1] == window
    assert df[last_index][2].shape[0] == window
    assert len(df[last_index][2].shape) == 1
    assert df[last_index][3].shape[0] == 50
    assert df[last_index][3].shape[1] == window
    assert len(df[last_index][3].shape) == 2
