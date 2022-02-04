# Code taken from 
# https://github.com/jdwillard19/lake_modeling/blob/01c4231b26f338da4e52513535b89cc9a6f30278/src/data/phys_operations.py
# and 
# https://github.com/jdwillard19/lake_modeling/blob/01c4231b26f338da4e52513535b89cc9a6f30278/src/data/pytorch_data_operations.py
# Parts changed or inactivated marked with
# #### Changed/Taken out >### ...code... ####< Changed/Taken out ###

import math
import torch
import numpy as np


def calculate_air_density(air_temp, rh):
    # returns air density in kg / m^3
    # equation from page 13 GLM/GLEON paper(et al Hipsey)

    # Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    # atmospheric pressure
    p = 1013.  # mb

    # water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)

    # water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure / (p - vapPressure)
    # print( 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15)))
    # print("vs")
    # print(1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))
    # sys.exit()
    # return 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15))
    return (1.0 / c_gas * (1 + r) / (1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100  #


def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 22 in GLM/GLEON paper(et al Hipsey)
    # GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    # calculate air density
    rho_a = calculate_air_density(air_temp, rel_hum)

    # specific heat capacity of air in J/(kg*C)
    c_a = 1005.

    # bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    # wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    # U_10 = wind_speed
    return -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)


def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 23 in GLM/GLEON paper(et al Hipsey)
    # GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    # air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    # bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    # latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    # wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
    #
    # ratio of molecular weight of water to that of dry air
    omega = 0.622

    # air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    #print('e_s: ',e_s.size())
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    #print('e_a: ',e_a.size())

    #print('rel_hum: ',rel_hum.size())
    #print('surf_temp: ',surf_temp.size())
    return -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)


def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)


def calculate_vapour_pressure_saturated(temp):
    # returns in miilibars
    # print(torch.pow(10, (9.28603523 - (2332.37885/(temp+273.15)))))

    # Converted pow function to exp function workaround pytorch not having autograd implemented for pow
    exponent = torch.tensor((9.28603523 - (2332.37885 / (temp + 273.15))) * math.log(10))
    return torch.exp(exponent)


def calculate_wind_speed_10m(ws, ref_height=2.):
    # from GLM code glm_surface.c
    c_z0 = 0.001  # default roughness
    return ws * (math.log(10.0 / c_z0) / math.log(ref_height / c_z0))
  
  
def transformTempToDensity(temp, use_gpu):
    # print(temp)
    # converts temperature to density
    # parameter:
    # @temp: single value or array of temperatures to be transformed
    densities = torch.empty_like(temp)
    if use_gpu:
        temp = temp.cuda()
        densities = densities.cuda()
    # return densities
    # print(densities.size()
    # print(temp.size())
    densities[:] = 1000 * (
                1 - ((temp[:] + 288.9414) * torch.pow(temp[:] - 3.9863, 2)) / (508929.2 * (temp[:] + 68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))
    # print("DENSITIES")
    # for i in range(10):
    #     print(densities[i,i])

    return densities


def calculate_dc_loss(outputs, n_depths, use_gpu):
    # calculates depth-density consistency loss
    # parameters:
    # @outputs: labels = temperature predictions, organized as depth (rows) by date (cols)
    # @n_depths: number of depths
    # @use_gpu: gpu flag

    assert outputs.size()[0] == n_depths

    densities = transformTempToDensity(outputs, use_gpu)

    # We could simply count the number of times that a shallower depth (densities[:-1])
    # has a higher density than the next depth below (densities[1:])
    # num_violations = (densities[:-1] - densities[1:] > 0).sum()

    # But instead, let's use sum(sum(ReLU)) of the density violations,
    # per Karpatne et al. 2018 (https://arxiv.org/pdf/1710.11431.pdf) eq 3.14
    sum_violations = (densities[:-1] - densities[1:]).clamp(min=0).sum()

    return sum_violations


# def calculate_ec_loss(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu,
#                       combine_days=1):
def calculate_ec_loss(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu,
                      combine_days=1):
    """
    description: calculates energy conservation loss
      parameters:
      @inputs: (N-depth*n_sets,n_days,8) features (standardized)
      @outputs: labels
      @phys: features(not standardized) of sw_radiation, lw_radiation, etc
      @labels modeled temp (will not used in loss, only for test)
      @depth_areas: cross-sectional area of each depth
      @n_depths: number of depths
      @use_gpu: gpu flag
      @combine_days: how many days to look back to see if energy is conserved (obsolete)
    """
    import numpy as np

    # ******************************************************
    # description: calculates energy conservation loss
    # parameters:
    # @inputs: features
    # @outputs: labels
    # @phys: features(not standardized) of sw_radiation, lw_radiation, etc
    # @labels modeled temp (will not used in loss, only for test)
    # @depth_areas: cross-sectional area of each depth
    # @n_depths: number of depths
    # @use_gpu: gpu flag
    # @combine_days: how many days to look back to see if energy is conserved (obsolete)
    # *********************************************************************************

    #print('n_depths: ',n_depths)
    n_sets = math.floor(inputs.size()[0] / n_depths)  # sets of depths in batch
    #print('n_sets: ',n_sets)
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]
    #print('n_dates: ',n_dates)

    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    #print("modeled temps: ", outputs.size())
    densities = transformTempToDensity(outputs, use_gpu)
    #print("modeled densities: ", densities.size())

    # for experiment
    if use_gpu:
        densities = densities.cuda()
    diff_per_set = torch.empty(n_sets)
    for i in range(n_sets):
        # loop through sets of n_depths

        # indices
        start_index = (i) * n_depths
        end_index = (i + 1) * n_depths

        # assert have all depths
        # assert torch.unique(inputs[:,0,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,100,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,200,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,0,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,100,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,200,1]).size()[0] == n_depths

        # calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index, :], densities[start_index:end_index, :],
                                              depth_areas)
        #print('lake_energies: ',lake_energies.size())
        # calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #print('lake_energy_deltas: ',lake_energy_deltas.size())
        # calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index, :, :], outputs[start_index, :], combine_days)
        
        #### Taken out >###
        
        ### can use this to plot energy delta and flux over time to see if they line up
        # doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min) \
        #                 .timetuple().tm_yday for x in dates[start_index, :]])
        
        # doy = doy[1:-1]
        
        ####< Taken out ###
        diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()

        # mendota og ice guesstimate
        # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

        # actual ice
        diff_vec = diff_vec[np.where((phys[0, 1:-1,8] == 0))[0]]# 8 instead of nicht 9] == 0))[0]]
        # #compute difference to be used as penalty
        if diff_vec.size() == torch.Size([0]):
            diff_per_set[i] = 0
        else:
            diff_per_set[i] = diff_vec.mean()
    if use_gpu:
        diff_per_set = diff_per_set.cuda()
    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    #print(diff_per_set.mean())
    return diff_per_set.mean()


def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            # take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


def calculate_lake_energy(temps, densities, depth_areas):
    # calculate the total energy of the lake for every timestep
    # sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    # then multiply by the specific heat of water
    dz = 0.5  # thickness for each layer, hardcoded for now
    cw = 4186  # specific heat of water
    energy = torch.empty_like(temps[0, :])
    n_depths = depth_areas.size()[0]
    depth_areas = depth_areas.view(n_depths, 1).expand(n_depths, temps.size()[1])
    energy = torch.sum(depth_areas * temps * densities * 0.5 * cw, 0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    # given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400  # seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1]) / (time * surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas


def calculate_energy_fluxes(phys, surf_temps, combine_days):
    # print("surface_depth = ", phys[0:5,1])
    fluxes = torch.empty_like(phys[:-combine_days - 1, 0])

    time = 86400  # seconds per day
    surface_area = 39865825

    e_s = 0.985  # emissivity of water, given by Jordan
    alpha_sw = 0.07  # shortwave albedo, given by Jordan Read
    alpha_lw = 0.03  # longwave, albeda, given by Jordan Read
    sigma = 5.67e-8  # Stefan-Baltzmann constant
    R_sw_arr = phys[:-1, 2] + (phys[1:, 2] - phys[:-1, 2]) / 2
    #print('R_sw_arr: ',R_sw_arr.size(),'\n',R_sw_arr[0])
    R_lw_arr = phys[:-1, 3] + (phys[1:, 3] - phys[:-1, 3]) / 2
    R_lw_out_arr = e_s * sigma * (torch.pow(surf_temps[:] + 273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:] - R_lw_out_arr[:-1]) / 2

    air_temp = phys[:-1, 4]
    air_temp2 = phys[1:, 4]
    rel_hum = phys[:-1, 5]
    rel_hum2 = phys[1:, 5]
    ws = phys[:-1, 6]
    ws2 = phys[1:, 6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2) / 2
    H = (H + H2) / 2

    # test
    fluxes = (R_sw_arr[:-1] * (1 - alpha_sw) + R_lw_arr[:-1] * (1 - alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])

    return fluxes
