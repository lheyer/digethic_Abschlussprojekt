###
import phys_functions as pf
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


def train_ec(model, train_loader, optimizer, criterion, num_epochs, depth_areas, device, \
             ec_lambda = 0.1, dc_lambda = 0.,lambda1 = 0.0, ec_threshold = 36, \
             begin_loss_ind = 50, grad_clip = 1.0, save_path=None, verbose=False):
  """
  grad_clip: how much to clip the gradient 2-norm in training
  lambda1: magnitude hyperparameter of regularization loss (L1-Norm)
  ec_lambda: magnitude hyperparameter of ec loss
  ec_threshold: anything above this far off of energy budget closing is penalized
  dc_lambda: magnitude hyperparameter of depth-density constraint (dc) loss
  begin_loss_ind = 50#index in sequence where we begin to calculate error or predict
  """
  
  reg1_loss = 0
  ec_loss = 0

  model.train()
  for epoch in range(num_epochs):
    
    
    avg_d_loss = 0
    avg_ec_loss = 0
    avg_reg1_loss = 0
    avg_loss = 0

    for batch_num, batch in enumerate(tqdm(train_loader)):

      optimizer.zero_grad()

      # forward pass
      X= batch[0].to(device)  #[0][0].to(device)
      X_phys = batch[1].to(device) #[1][0].to(device)
      dates = batch[2].to(device) #[2][0].to(device)
      Y= batch[3][:,:,np.newaxis].to(device) #[3][0][:,:,np.newaxis].to(device)
      #print(X.size())
      #print(Y.size())
      #Y.to(device)
      

      
      model.hidden = model.init_hidden(batch_size=X.size()[0])
      h_state = None

      #make prediction
      preds,h_state = model(X)
      
      h_state = None
      #print('prediction size: ',preds.size())
      #print('Ŷ size: ',Y.size())


      loss_preds = preds[:, begin_loss_ind:]
      loss_Y = Y[:, begin_loss_ind:]
      
      # [~torch.isnan(loss_Y)] for not including nan-values in labels used for calculating MSE-loss 
      # though not important for the other loss metrics
      #print('epoch: ',epoch)
      #print('batch_num:', batch_num)
      #print('shape loss_preds', loss_preds.shape)
      #print('shape loss_Y', loss_Y.shape)
      #print('shape torch.isnan(loss_Y)', torch.isnan(loss_Y).shape)
      
      d_loss = criterion(loss_preds[~torch.isnan(loss_Y)], loss_Y[~torch.isnan(loss_Y)])


      if ec_lambda > 0:
        ec_loss = pf.calculate_ec_loss(X[:, begin_loss_ind:, :], loss_preds, \
                                    X_phys[:, begin_loss_ind:, :], \
                                    loss_Y, dates[begin_loss_ind:], \
                                    depth_areas, len(depth_areas), 24, \
                                    use_gpu=False, combine_days=1)
      
      if lambda1 > 0:
        l1_parameters = []
        for name, parameter in model.named_parameters():
          if 'bias' in name:
            continue
          else:
            l1_parameters.append(parameter.view(-1))
        reg1_loss = torch.abs(torch.cat(l1_parameters)).sum()
      
      
      loss = d_loss + ec_lambda * ec_loss + lambda1*reg1_loss 

      
      loss.backward(retain_graph=False)
      

      # faster convergence of loss (no shoulder)
      if grad_clip > 0:
        clip_grad_norm_(model.parameters(), grad_clip, norm_type=2)


      optimizer.step()

      avg_d_loss += d_loss.item()
      avg_loss += loss.item()
      if ec_lambda > 0:
        avg_ec_loss += ec_loss.item()
        
      if lambda1 > 0:
        avg_reg1_loss += reg1_loss.item()
   
    if verbose:
      print('Training  sum of losses: {} \n'.format(avg_loss/len(train_loader)))
      print('Training  data loss: {} \n'.format(avg_d_loss/len(train_loader)))
      print('Training  EC loss: {} \n'.format(avg_ec_loss/len(train_loader)))

  print('training finished')
  
  if save_path:
    print('model save to: ',str(save_path))
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, str(save_path))


  return epoch,loss

