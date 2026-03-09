import torch
import torch.nn as nn
import torch.autograd as autograd
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from datasets import inverse_data_transform, data_transform
from regularization.gauss_weight import gauss_weights, colorforward_difference_y_direction, colorbackward_difference_y_direction, colorforward_difference_x_direction, colorbackward_difference_x_direction
import random

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ddpg_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None):


    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.sampling.T_sampling
        n = x.size(0)
        img_shape = x.shape 
        device = x.device
        x0_preds = []
        xs = [x]
        total_timesteps = config.diffusion.num_diffusion_timesteps
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0: j = -1    
            if j < i: 
                t = (torch.ones(n) * i).to(device)
                next_t = (torch.ones(n) * j).to(device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(device)

                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=device) * class_num
                    et = model(xt, t, classes); et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
                if et.size(1) == 6: et = et[:, :3]

               
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    
                if sigma_y==0.:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = (at_next) ** args.gamma
                    weight_noise_t = delta_t


                eta_reg = max(1e-4, sigma_y**2 * args.eta_tilde )
                if args.eta_tilde < 0:
                    eta_reg = 1e-4 + args.xi * (sigma_y*255.0)**2

                scale_gLS = args.scale_ls  #e.g. <= 1/A_funcs.singulars().max()**2 

                guidance_BP = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1), eta_reg).reshape(*x0_t.size())
                guidance_LS = A_funcs.At(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)).reshape(*x0_t.size())
                
                if args.step_size_mode==0:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = 1
                elif args.step_size_mode==1:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = (1 - at_next)/(1 - at)
                elif args.step_size_mode==2:
                    step_size_LS = (1 - at_next)/(1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"
                    
                xt_next_tilde = x0_t  - step_size * ( step_size_BP * (1-delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS)
                
                
                reg_gamma =args.reg_gamma
                current_reg_strength = args.strength
                x0t_shape = x0_t.permute(0, 2, 3, 1) 
                if n > 1:
                     reshape_x0t = x0t_shape[0].view(img_shape[2], img_shape[3], img_shape[1]) 
                else:
                     reshape_x0t = x0t_shape.view(img_shape[2], img_shape[3], img_shape[1])

                reshape_x0t_numpy_x = colorbackward_difference_x_direction(colorforward_difference_x_direction(reshape_x0t.cpu()))
                x_tensor = torch.from_numpy(reshape_x0t_numpy_x)
                w_x = gauss_weights(x_tensor, reg_gamma)

                reshape_x0t_numpy_y = colorbackward_difference_y_direction(colorforward_difference_y_direction(reshape_x0t.cpu()))
                x_tensor = torch.from_numpy(reshape_x0t_numpy_y)
                w_y = gauss_weights(x_tensor, reg_gamma)

                new1 = colorbackward_difference_x_direction(w_x.cpu() * colorforward_difference_x_direction(reshape_x0t.cpu()))
                new2 = colorbackward_difference_y_direction(w_y.cpu() * colorforward_difference_y_direction(reshape_x0t.cpu()))
                add = torch.from_numpy(new1 + new2).float().to(device)
                add = add.permute(2, 0, 1) # C, H, W
                if n > 1:
                    add = add.unsqueeze(0).expand(n, -1, -1, -1)
                xt_next_tilde = xt_next_tilde + current_reg_strength * add

                et_hat = (xt - at.sqrt() * xt_next_tilde) / (1 - at).sqrt()
                c1 = 0; c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1 - zeta) * weight_noise_t
                xt_next = at_next.sqrt() * xt_next_tilde + c1 * torch.randn_like(x0_t) + c2 * et_hat

                x0_preds.append(x0_t.to('cpu')) 
                xs.append(xt_next.to('cpu'))
            else:
                assert 1, "error：j >= i"

        if sigma_y != 0.:
            xs.append(x0_t.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1
    t = T_sampling; ts = []
    while t >= 1:
        t = t - 1; ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1; ts.append(t)
    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0); assert t <= T_sampling, (t, T_sampling)
        

