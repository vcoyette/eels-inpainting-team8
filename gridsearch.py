from itertools import product
from inpainting_functions import *
from utils.common_utils import *
from models.skip import *
import csv
import torch
import time

torch.backends.cudnn.enabled = torch.cuda.is_available()
torch.backends.cudnn.benchmark = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

params_resnet = {
        'loss_name': ['mse', 'master_metric'],
        'pcat_th': [5, 10, 15],
        'num_blocks': [4, 8, 16, 32],
        'num_channels': [32, 64, 128],
        'path': ['data/HR-sample/spim4-2_ali.dm4']
        }

params_skip = {
        'loss_name': ['mse', 'master_metric'],
        'grad_clipping':[True, False],
        'pca_th': [5,10,15],
        'num_channels_down': [[8,8,8],[16,16,16],[64,32,16]],
        'filter_size_':[3,5,7],
        'path': ['data/HR-sample/spim4-2_ali.dm4'],
        'net_type':['skip6']
        }

num_iter = 1

def params_gen(params):
    if not params.items():
        yield {}
    else:
        keys, values = zip(*params.items())
        for v in product(*values):
            p = dict(zip(keys, v))
            yield p

def get_final_metrics(out_np,orig_np):
    filled_image, real_image = torch.tensor(out_np).unsqueeze(0).float(), torch.tensor(orig_np).unsqueeze(0).float()
    psnr = master_metric(real_image, filled_image, 1, 0, 0, 'sum')
    ssim = master_metric(real_image, filled_image, 0, 1, 0, 'sum')
    sad = master_metric(real_image, filled_image, 0, 0, 1, 'sum')

    return ssim.item(), psnr.item(), sad.item()

def test_params_skip(loss_name, grad_clipping, pca_th, num_channels_down, filter_size_, path, net_type):

    start_time = time.time()

    full_pca_img, partial_pca_img, mask, l1, l2, PCA1, PCA2 = load_and_process_fc(path,pca_th,0.2)
    img_var = np_to_torch(partial_pca_img).type(dtype)
    mask_var = np_to_torch(mask).type(dtype)

    LR = 0.01

    INPUT = 'noise'
    input_depth = partial_pca_img.shape[0]
    output_depth = partial_pca_img.shape[0]

    num_channels_up = num_channels_down.copy()
    num_channels_up.reverse()

    depth = int(net_type[-1])
    net = skip(input_depth, output_depth,
            num_channels_down = num_channels_down[:depth],
            num_channels_up =   num_channels_up[:depth],
            num_channels_skip =    [16, 16, 16][:depth],
            filter_size_up = filter_size_,filter_size_down = filter_size_,  filter_skip_size=1,
            upsample_mode='nearest', # downsample_mode='avg',
            need1x1_up=False,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)


    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, partial_pca_img.shape[1:],var=1).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    net_input = net_input_saved


    print('Starting optimization with ADAMW')
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=False, patience=100, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=5e-6)

    for j in range(num_iter):

        out = net(net_input)

        optimizer.zero_grad()

        if loss_name == 'mse':
            mse = torch.nn.MSELoss().type(dtype)
            total_loss = mse(out * mask_var, img_var * mask_var)
        elif loss_name == 'master_metric':
            total_loss = -master_metric((out * mask_var), (img_var * mask_var), 1, 1, 1, 'product')
        else:
            raise ValueError("Input a correct loss name (among 'mse' | 'master_metric'")

        total_loss.backward()

        if grad_clipping:
            for param in net.parameters():
                param.grad.data.clamp_(-1, 1)

        optimizer.step()
        scheduler.step(total_loss)

    out_np = torch_to_np(out)

    elapsed = time.time() - start_time

    return get_final_metrics(out_np,full_pca_img), elapsed


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

def test_parameters_resnet(loss_name, grad_clipping, pca_th, num_blocks, num_channels, path):

    start_time = time()

    full_pca_img, partial_pca_img, mask, l1, l2, PCA1, PCA2 = load_and_process_fc(path,pca_th,0.2)
    img_var = np_to_torch(partial_pca_img).type(dtype)
    mask_var = np_to_torch(mask).type(dtype)

    LR = 0.01

    INPUT = 'noise'
    input_depth = partial_pca_img.shape[0]
    output_depth = partial_pca_img.shape[0]

    net = ResNet(input_depth, output_depth, num_blocks, num_channels, act_fun='LeakyReLU')

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, partial_pca_img.shape[1:],var=1).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    net_input = net_input_saved

    print('Starting optimization with ADAMW')
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=False, patience=100, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=5e-6)

    for j in range(num_iter):

        out = net(net_input)

        optimizer.zero_grad()

        if loss_name == 'mse':
            mse = torch.nn.MSELoss().type(dtype)
            total_loss = mse(out * mask_var, img_var * mask_var)
        elif loss_name == 'master_metric':
            total_loss = -master_metric((out * mask_var), (img_var * mask_var), 1, 1, 1, 'product')
        else:
            raise ValueError("Input a correct loss name (among 'mse' | 'master_metric'")

        total_loss.backward()

        if grad_clipping:
            for param in net.parameters():
                param.grad.data.clamp_(-1, 1)

        optimizer.step()
        scheduler.step(total_loss)

    out_np = torch_to_np(out)

    elapsed = time.time() - start_time

    return get_final_metrics(out_np,full_pca_img), elapsed



# Hyperparam tuning skip
num_skip_config = sum(1 for i in params_gen(params_skip))
print("Number of skip config to test: {}".format(num_skip_config))
configs_skip = []

count=0
for param in params_gen(params_skip):
    config_skip_items = param.copy()
    (ssim, psnr, sad), duration = test_params_skip(**param)
    config_skip_items['ssim'] = ssim
    config_skip_items['psnr'] = psnr
    config_skip_items['sad'] = sad
    config_skip_items['duration'] = duration

    configs_skip.append(config_skip_items)
    count += 1
    print("Skip: {}/{}".format(count, num_skip_config))

# Hyperparam tuning skip
num_resnet_config = sum(1 for i in params_gen(params_resnet))
print("Number of resnet config to test: {}".format(num_resnet_config))
configs_resnet = {}
count = 0
for param in params_gen(params_resnet):
    config_resnet_items = param.copy()
    (ssim, psnr, sad), duration = test_params_resnet(**param)
    config_skip_items['ssim'] = ssim
    config_skip_items['psnr'] = psnr
    config_skip_items['sad'] = sad
    config_skip_items['duration'] = duration
    configs_resnet.append(config_resnet_items)
    count += 1
    print("ResNet: {}/{}".format(count, num_skip_config))


def save_list_dict_csv(dict_list, path):
    f = open(path, 'w')

    with f:
        writer = csv.DictWriter(f, fieldnames=dict_list[0].keys())

        for dic in dict_list:
            writer.writerow(dic)

save_list_dict_csv('skip.csv', configs_skip)
save_list_dict_csv('resnet.csv', configs_resnet)




