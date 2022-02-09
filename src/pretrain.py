# Pretraining with Labels simulated with GLM model
import phys_functions as pf
import train_funtions as tfunc
import preprossesing as pp
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import GeneralLSTM


# Save model somewhere?
save_path = None

##################
### Data paths ###
##################

#### change paths to something interactive (--> if __name__ == '__main__' block needed)
mendota_meteo_path = 'data/pretrain/mendota_meteo.csv'
predict_pb0_path = 'data/predictions/me_predict_pb0.csv'
ice_flags_path = 'data/pretrain/mendota_pretrainer_ice_flags.csv'

###################
### Import data ###
###################

#window size with sliding window gap of 176 (half window size)
window = 353
stride = int(window/2)

mendota_depth_areas = pp.lake_depth_areas_dict['Lake Mendota']
train_dataset = pp.Meteo_DS(mendota_meteo_path,predict_pb0_path,mendota_depth_areas,ice_csv_path=ice_flags_path,transform=True)

train_dl = DataLoader(pp.SlidingWindow(train_dataset.Xt,window,stride,train_dataset.labels,phys_data=train_dataset.X, dates=train_dataset.dates ),shuffle=False)

########################################
### Declare constant hyperparameters ###
########################################

learning_rate = 0.005
epochs = 400
state_size = 20 # number of hidden features

# bt_sz: n_depths*N_sec
N_sec = 19
# value describing balance between data-driven loss and energy driven Loss
elam = 0.1
# value above enery difference will be penalized
ec_threshold = 24


###############################
### Init model & optimizer  ###
###############################

input_size = train_dl.dataset.x.size()[-1]
batch_size = train_dl.dataset.x.size()[0]
model = GeneralLSTM(input_size, state_size,batch_size, num_layers=1)
criterion = torch.nn.MSELoss()#
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
model.to(device)

save_path = 'model/model_pretrain_pgdl_ec01_400_til2009.model'

tfunc.train_ec(model, train_dl, optimizer, criterion, epochs, torch.Tensor(mendota_depth_areas.astype(np.float32)),\
              device, ec_lambda = 0.1, dc_lambda = 0.,lambda1 = 0.0, ec_threshold = 36, \
             begin_loss_ind = 50, grad_clip = 1.0, save_path = save_path, verbose=False)
