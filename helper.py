from __future__ import print_function
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models.resnet import ResNet
from models.skip import skip
import torch
import torch.optim

from utils.inpainting_utils import *

from skued import dmread
import inpystem
from PIL import Image
from matplotlib import cm
import SSIM as SSIM
import time

import scipy.io as sio

def results(out_np, time_elapsed, PCA, is_test=True, orig_img=None):
    print('Seconds elapsed : %d ' % (time_elapsed))

    restored = inverse_pca(out_np,PCA)

    restored_summed = sum(restored)
    restored_summed = scale_image(restored_summed)

    np.save('Restored_PCs_{}_base.npy'.format(out_np.shape[0]), out_np)
    np.save('Restored_original_base.npy', restored)
    np.save('Restored_original_base_summed.npy', restored_summed)

    if not is_test:
        orig_summed = sum(orig_img)
        #print_metrics(torch.tensor(orig_summed).unsqueeze(0).unsqueeze(0).float(),torch.tensor(restored_summed).unsqueeze(0).unsqueeze(0).float())
        return restored_summed , orig_summed

    else:
        return restored_summed

###############################################################################

def train(net, partial_pca_img, mask, optimizer_type='adamw', loss_name='master_metric', num_iter=3001, grad_clipping=True, LR=0.01, reg_noise_std = 0.01, show_every=100):

    dtype = torch.cuda.FloatTensor

    start_time = time.time()

    INPUT = 'noise'
    input_depth = partial_pca_img.shape[0]
    output_depth = partial_pca_img.shape[0]

    img_var = np_to_torch(partial_pca_img).type(dtype)
    mask_var = np_to_torch(mask).type(dtype)

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, partial_pca_img.shape[1:],var=1).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    net_input = net_input_saved        

    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)  

    if optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

    elif optimizer_type == 'adamw':
        print('Starting optimization with ADAMW')
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=False, patience=100, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=5e-6)

    for j in range(num_iter):

        out = net(net_input)

        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(net_input)
        
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

        if j % show_every == 0:
            print ('Iteration %d    Loss %.4f' % (j, total_loss.item()))
            print_metrics(out * mask_var, img_var * mask_var)
            out_np = torch_to_np(out)
            plot_spectre(out_np)

        optimizer.step()
        scheduler.step(total_loss)

    out_np = torch_to_np(out)

    elapsed = time.time() - start_time

    return get_final_metrics(out_np*mask, partial_pca_img), elapsed, out_np

###############################################################################

def scale_image(img_np):
    a = np.min(img_np)
    b = np.max(img_np)
    return 255/(b-a)*(img_np-a)

###############################################################################

def plot_spectre(img_np):
    if img_np.shape[0] > 1:
        img_np = scale_image(img_np)
        columns = 2
        rows = 1
        f, axs = plt.subplots(rows,columns,figsize=(16,8))
        
        axs[0].imshow(img_np[0],cmap='gray')
        axs[0].set_title('Spectrum[0]')

        axs[1].imshow(img_np[1],cmap='gray')
        axs[1].set_title('Spectrum[1]')
    
    else:
        img_np = scale_image(img_np)
        columns = 1
        rows = 1
        f, axs = plt.subplots(rows,columns,figsize=(8,8))
        plt.imshow(img_np[0],cmap='gray')

    plt.show()

    return f

###############################################################################

def plot_final(out_np,orig_np):
    print_metrics(torch.tensor(out_np).unsqueeze(0).float(), torch.tensor(orig_np).unsqueeze(0).float())
    a = np.min(orig_np)
    b = np.max(orig_np)
    out_np , orig_np = 255/(b-a)*(out_np-a) , 255/(b-a)*(orig_np-a)
    print('                                                 --- ORIGINAL ---')
    f = plot_spectre(orig_np)
    print('                                                 --- OUTPUT ---')
    f = plot_spectre(out_np)

###############################################################################

def get_final_metrics(out_np,orig_np):
    filled_image, real_image = torch.tensor(out_np).unsqueeze(0).float(), torch.tensor(orig_np).unsqueeze(0).float()
    psnr = master_metric(real_image, filled_image, 1, 0, 0, 'sum')
    ssim = master_metric(real_image, filled_image, 0, 1, 0, 'sum')
    sad = master_metric(real_image, filled_image, 0, 0, 1, 'sum')

    return ssim.item(), psnr.item(), sad.item()

###############################################################################

def print_metrics(real_image, filled_image):
    psnr = master_metric(real_image, filled_image, 1, 0, 0, 'sum')
    ssim = master_metric(real_image, filled_image, 0, 1, 0, 'sum')
    sad = master_metric(real_image, filled_image, 0, 0, 1, 'sum')
    print('SSIM : %.4f -- PSNR : %.4f -- SAD : %.4f' % (ssim,psnr,sad))

###############################################################################

def load_and_process_fc(path, PCA_th, p):
    
    '''
       ____________________________ 
    /!\ For fully completed images /!\
    \¡/____________________________\¡/
    
    Takes as inputs :
    
     - path : String corresponding to the path of your ".dm3" or ".dm4" data,
     - PCA_th : Int corresponding to the number of wavelengths desired with the PCA,
     - p : percentage of image you want to keep intact (0<p<1).
     
     The data's shape must be : Spectrum Size x m x n (where m and n matche with the size of the image).
     
     Returns :
     
     (The array shape is always as follows : Size x m x n)
     
     - full _image : A numpy Array of the full image with the principal wavelenghts (obtained thanks to a PCA),
     - partial_image : A numpy Array of the partial image (PCA realized on it),
     - mask : A numpy Array corresponding to the mask of (0, 1) used to hide the missing pixels of the image.
     - l1 : A numpy array corresponding to the percentage (0<l<1) of variance explained by the Spectrum associated 
            for the full_image,
     - l2 : A numpy array corresponding to the percentage (0<l<1) of variance explained by the Spectrum associated 
            for the partial_image,
     - PCA1 : The object PCA used by inPystem in order to achieve the inverse PCA on the full image,
     - PCA2 : The object PCA used by inPystem in order to achieve the inverse PCA on the partial image.
     
    '''
    
    try:
        img = dmread(path)
    except:
        print("File not found ! Edit your path or be sure to have skued.dmread as dmread.")
        
    print("Image loaded with success !")
    
    if len(np.shape(img))!=3 or np.shape(img)[0]==0 or np.shape(img)[1]==0 or np.shape(img)[2]==0:
        raise ValueError("Format Invalid ! Expected an S x m x n array with \".dm3\" or \".dm4\" format")
    
    try:
        p = min(max(0,p),1)
    except:
        print('"p" must be a Double or a Float (or equivalent) between 0 and 1')
    
    try:
        PCA_th = int(max(0,PCA_th))
    except:
        print('"PCA_th" must be an Integer between 1 and the Spectrum Size')
    
    m, n = np.shape(img)[1], np.shape(img)[2]
    N = int(p*m*n)
    mask = np.random.permutation([0]*(m*n-N)+[1]*N).reshape((m, n))
    
    print('Mask created with success !')
    
    if np.shape(img)[0]<PCA_th:
        raise ValueError('Not enough Spectrum dimensions for the PCA.')
    
    else:
        
        Y_1 = np.transpose(img, (1, 2, 0))
        Y_2 = np.transpose(mask*img, (1, 2, 0))
        PCA_1 = inpystem.tools.PCA.PcaHandler(Y_1, mask=None, PCA_transform=True, PCA_th = PCA_th, verbose=False)
        full_img = np.transpose(PCA_1.direct(), (2, 0, 1))
        
        mfi = np.max(full_img)
        mmfi = np.min(full_img)
        full_img = 1/(mfi-mmfi)*(full_img-mmfi)
        
        PCA_2 = inpystem.tools.PCA.PcaHandler(Y_2, mask=mask, PCA_transform=True, PCA_th = PCA_th, verbose=False)
        partial_img = mask*np.transpose(PCA_2.direct(), (2, 0, 1))
        
        mpi = np.max(partial_img)
        mmpi = np.min(partial_img)
        partial_img = 1/(mfi-mmfi)*(partial_img-mmfi)
        
        print('Both PCA done with success !')
        
        l1 = percentage_variance(img, mask)
        l2 = percentage_variance(mask*img, mask)
        
        print('Both weights calculated with success !')
    
    return full_img, partial_img, mask, l1, l2, PCA_1, PCA_2

###############################################################################

def percentage_variance(img, mask):
    
    '''
        Takes as inputs :
            - img : A numpy array (The array shape is always as follows : Size x m x n),
            - mask : A numpy array full of 0 & 1 corresponding to the mask --> It can be 'None'.
            
        Returns :
            - lambdas : A numpy array corresponding to the percentage (0<l<1) of variance explained by the Spectrum associated.
    
    '''
        
    if mask is None:
        mask = np.array([[1]*n]*m)
        
    Y = np.transpose(mask*img, (1, 2, 0))
    m, n, M = Y.shape
    N = int(mask.sum())
    P = m * n

    nnz = np.flatnonzero(mask)
    Yr = Y.reshape((n * m, M)).T

    Yrm = np.tile(np.mean(Yr[:, nnz], axis=1), (P, 1)).T
    Yrwm = Yr - Yrm
    [d, V] = np.linalg.eig(np.cov(Yrwm[:, nnz]))

    ind = np.argsort(d)[::-1]
    d = d[ind]
    
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.bar(list(range(1,21)),d[:20]/sum(d)*100)
    plt.title('Percentage of data Explained by Eigen vector')
    plt.xlabel('Eigen Values')
    plt.ylabel('%')
    
    lambdas = d/sum(d)
    
    return lambdas

###############################################################################

def inverse_pca(img, PCA, mfi=None, mmfi=None):
    
    '''
    This function can do the reverse PCA of an image in order to restore all the spectrum.
    
        Takes as inputs :
            - img : A numpy array (The array shape is always as follows : Size x m x n),
            - PCA : The class PCA (from inPystem) used and returned by a function 'load_and_process_?'.
            
        Returns :
            - recovered_img : The image with full recovery of the spectrum.
    '''
    
    if len(np.shape(img))!=3  or np.shape(img)[1]==0 or np.shape(img)[2]==0:
        raise ValueError("Format Invalid ! Expected an S x m x n numpy array.")
        
    if np.shape(img)[0]!=PCA.PCA_th:
        raise ValueError("Spectral dimension does not match with the PCA.")
    
    iimmgg = img

    if mfi != None and mmfi!= None:
        iimmgg = (mfi-mmfi)*np.transpose(img, (1, 2, 0))+mmfi

    recovered_img = np.transpose(PCA.inverse(iimmgg), (2, 0, 1))

    print('Inverse PCA done with success !')
    
    return recovered_img

###############################################################################

def master_metric(real_image, filled_image, a, b, c, type_):
    
    '''
        Takes as inputs :
            
            - real_image : The real image, format --> torch whose shape is (1,Spectrum_Size,m,n),
            - filled_image : The image filled by the estimator, format --> torch whose shape is (1,Spectrum_Size,m,n),
            
            - a,b,c : 3 parameters a,b,c >= 0 such as if type_== 'sum', master_metric = a x PSNR + b x SSIM + c x SAD,
                                                      if type_== 'product', master_metric = PSNR^a * SSIM^b * SAD^c,
            - type_ : 'sum' (if so, a+b+c=1) or 'product'.
            
        Returns :
        
            Return a metric which is a combinaison of the PSNR, the SSIM and the SAD metric (if a=1 and b,c=(0,0), the metric
            is equivalent to PSNR, if b=1 and a,c=(0,0), the metric is equivalent to SSIM and so on).
     
    '''
    
    try:
        a, b, c = np.abs(a), np.abs(b), np.abs(c)
    except:
        print("The parameters a,b,c must be numbers.")
    if a+b+c==0:
        raise ValueError("At least one parameter a,b or c should differ from 0.")
        
    ssim = SSIM.ssim(filled_image, real_image)
    loss_psnr = nn.MSELoss()
    psnr = 1/8*torch.log10(255*255/loss_psnr(filled_image, real_image))
    ri_f = torch.flatten(real_image)
    fi_f = torch.flatten(filled_image)
    sad = torch.dot(ri_f, fi_f)/(torch.norm(ri_f)*torch.norm(fi_f))
    
    if type_=='sum':
        a, b, c = a/(a+b+c), b/(a+b+c), c/(a+b+c)
        return a*psnr+b*ssim+c*sad
         
    elif type_=='product':
        return psnr**a * ssim**b * sad**c
    
    else:
        raise ValueError("\"type_\" must be 'sum' or 'product'.")

###############################################################################

def plot_spectra_comparison(real_image, filled_image):
    
    '''
    Plot the differences between the real image and the predicted one among the spectrum kept by the PCA.
    
        Takes as inputs : 
            - real_image : A numpy array (The array shape is always as follows : Size x m x n),
            - filled_image : A numpy array (The array shape is always as follows : Size x m x n).
    '''
    
    if np.shape(real_image)!=np.shape(filled_image):
        raise ValueError("Both images must have the same shape.")
    
    plt.rcParams['figure.figsize'] = [20, 20]
    plt.figure()
    S = np.shape(real_image)[0]
    for i in range(S):
        plt.subplot(S,2,i+1)
        plt.imshow(real_image[i,:,:])
        plt.title('Real Image, Spectrum n°:'+str(i))
        plt.subplot(S,2,i+2)
        plt.imshow(filled_image[i,:,:])
        plt.title('Filled Image, Spectrum n°:'+str(i))

###############################################################################

def plot_each_spectrum(img):
    
    '''
    Plot the 10 first spectra of the image (bette doing a PCA first).
    
        Takes as inputs : 
            - img : A numpy array (The array shape is always as follows : Size x m x n).
    '''
    
    plt.rcParams['figure.figsize'] = [20, 20]
    plt.figure()
    S = np.shape(img)[0]
    for i in range(min(S,10)):
        plt.subplot(5,2,i+1)
        plt.imshow(img[i,:,:])
        plt.title('Spectrum n°:'+str(i))

###############################################################################

def plot_colorized_spectrum(img):
    
    '''
    Plot the 3 prime spectra as RGB, RBG, GRB, GBR, BRG, BGR.
    
        Takes as inputs : 
            - img : A numpy array (The array shape is always as follows : Size x m x n).
    '''
    
    S = np.shape(img)[0]
    if S<3:
        raise ValueError("You need at least the 3 principal spectra obtained with a PCA.")
                
    l = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    
    img = np.transpose(img, (1,2,0))
    
    plt.rcParams['figure.figsize'] = [20, 12]
    plt.figure()
    
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.imshow(img[:,:,l[i]])

###############################################################################

def load_and_process_p(path, PCA_th, mask):
    
    '''
       ____________________
    /!\ For partial images /!\
    \¡/____________________\¡/
    
    Takes as inputs :
    
     - path : String corresponding to the path of your ".dm3" or ".dm4" data,
     - PCA_th : Int corresponding to the number of wavelengths desired with the PCA,
     - mask : A numpy Array corresponding to the mask of (0, 1) used to hide the missing pixels of the image.
     
     The data's shape must be : Spectrum Size x m x n (where m and n matche with the size of the image).
     
     Returns :
     
     (The array shape is as follows : Size x m x n)
     
     - partial_image : A numpy Array of the partial image (PCA realized on it),
     - mask : A numpy Array corresponding to the mask of (0, 1) used to hide the missing pixels of the image,
     - l : A numpy array corresponding to the percentage (0<l<1) of variance explained by the Spectrum,
     - PCA_ : The object PCA used by inPystem in order to achieve the inverse PCA.
     
    '''
    
    try:
        img = dmread(path)
    except:
        print("File not found ! Edit your path or be sure to have skued.dmread as dmread.")
        
    print("Image loaded with success !")
    
    if len(np.shape(img))!=3 or np.shape(img)[0]==0 or np.shape(img)[1]==0 or np.shape(img)[2]==0:
        raise ValueError("Format Invalid ! Expected an S x m x n array with \".dm3\" or \".dm4\" format")
            
    if len(np.shape(mask))!=2 or np.shape(img)[1]!=np.shape(mask)[0] or np.shape(img)[2]!=np.shape(mask)[1]:
        raise ValueError("Format Invalid ! Expected a m x n numpy array as mask.")
        
    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[0]):
            if np.isnan(mask[i,j]):
                mask[i,j]=0
                
    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[0]):
            if mask[i,j]!=0 and mask[i,j]!=1:
                raise ValueError("The mask should only contain 0 & 1 values (or NaN instead of 0).")
    
    try:
        PCA_th = int(max(0,PCA_th))
    except:
        print('"PCA_th" must be an Integer between 1 and the Spectrum Size')
    
    if np.shape(img)[0]<PCA_th:
        print('Not enough Spectrum dimensions for the PCA, the array is returned as it is.')
    
    else:
        
        Y = np.transpose(mask*img, (1, 2, 0))
                
        PCA_ = inpystem.tools.PCA.PcaHandler(Y, mask=mask, PCA_transform=True, PCA_th = PCA_th, verbose=False)
        partial_img = mask*np.transpose(PCA_.direct(), (2, 0, 1))
        
        print('PCA done with success !')
        
        mpi = np.max(partial_img)
        mmpi = np.min(partial_img)
        partial_img = 1/(mfi-mmfi)*(partial_img-mmfi)
        
        l = percentage_variance(img, mask)
        
        print('Weights calculated with success !')
    
    return partial_img, mask, l, PCA_

###############################################################################

def load_aviris(p, pca_bool, PCA_th):
    
    '''
    Takes as inputs :
    
     - p : percentage of image you want to keep intact (0<p<1),
     - pca_bool : Boolean --> Do the PCA or not ?
     - PCA_th : Int corresponding to the number of wavelengths desired with the PCA.
     
     The data's shape must be : Spectrum Size x m x n (where m and n matche with the size of the image).
     
     Returns :
     
     (The array shape is always as follows : Size x m x n)
     
     - full _image : A numpy Array of the full image with the principal wavelenghts (obtained thanks to a PCA),
     - img_rgb : the components RGB of the image,
     - partial_image : A numpy Array of the partial image (PCA realized on it),
     - mask : A numpy Array corresponding to the mask of (0, 1) used to hide the missing pixels of the image,
     - PCA1 : The object PCA used by inPystem in order to achieve the inverse PCA on the full image (None if not pca_bool),
     - PCA2 : The object PCA used by inPystem in order to achieve the inverse PCA on the partial image (None if not pca_bool).
     
    '''
    
    try:
        aviris = sio.loadmat('data/aviris.mat')
    except:
        print("File not found ! Be sure to have aviris.mat in your data folder.")
        
    img = aviris['aviris']
    print("Image loaded with success !")
    
    try:
        p = min(max(0,p),1)
    except:
        print('"p" must be a Double or a Float (or equivalent) between 0 and 1')
    
    try:
        PCA_th = int(max(0,PCA_th))
    except:
        print('"PCA_th" must be an Integer between 1 and the Spectrum Size')
    
    bands_rgb = (26, 20, 11)
    img_rgb = normalize_img(img[:,:,bands_rgb])
    
    print('RGB-Image created with success !')
    
    m, n = np.shape(img)[0], np.shape(img)[1]
    N = int(p*m*n)
    mask = np.random.permutation([0]*(m*n-N)+[1]*N).reshape((m, n))
    
    print('Mask created with success !')
    
    if pca_bool:
        
        if np.shape(img)[2]<PCA_th:
            raise ValueError('Not enough Spectrum dimensions for the PCA.')
        else:
            PCA_1 = inpystem.tools.PCA.PcaHandler(img, mask=None, PCA_transform=True, PCA_th = PCA_th, verbose=False)
            full_img = np.transpose(PCA_1.direct(), (2, 0, 1))

            mfi = np.max(full_img)
            mmfi = np.min(full_img)
            full_img = 1/(mfi-mmfi)*(full_img-mmfi)

            img2 = np.transpose(img, (2, 0, 1))
            PCA_2 = inpystem.tools.PCA.PcaHandler(np.transpose(mask*img2, (1, 2, 0)), mask=mask, PCA_transform=True, PCA_th = PCA_th, verbose=False)
            partial_img = mask*np.transpose(PCA_2.direct(), (2, 0, 1))

            mpi = np.max(partial_img)
            mmpi = np.min(partial_img)
            partial_img = 1/(mpi-mmpi)*(partial_img-mmpi)

            print('Both PCA done with success !')
        
    else:
        full_img = np.transpose(normalize_img(img), (2, 0, 1))
        partial_img = mask*full_img
        PCA_1, PCA_2 = None, None
    
    return full_img, img_rgb, partial_img, mask, PCA_1, PCA_2

def normalize_img(img):
    res = img;
    for k in np.arange(3):
        vmin = np.percentile(res[:,:,k],1)
        vmax = np.percentile(res[:,:,k],99)
        res[:,:,k] = (res[:,:,k] - vmin) / (vmax - vmin)
        np.clip(res[:,:,k], 0., 1., res[:,:,k]) # in-place clipping
    return res

###############################################################################

def load_and_process_test_image(PCA_th):
    
    data = np.load('data/Test/dtest_1_data.npy')
    mask = 1*np.load('data/Test/dtest_1_mask.npy')
    path = np.load('data/Test/dtest_1_scan.npy')
    m, p = data.shape
    m, n = mask.shape
    full_data = np.zeros((m*n, p))
    full_data[path,:] = data
    img = full_data.reshape(m, n, p)
    
    test_img = mask*np.transpose(img, (2, 0, 1))
    
    PCA = inpystem.tools.PCA.PcaHandler(np.transpose(test_img, (1, 2, 0)), mask=mask, PCA_transform=True, PCA_th = PCA_th, verbose=False)
    pca_test_img = np.transpose(PCA.direct(), (2, 0, 1))

    mmm=[]
    for i in range(test_img.shape[0]):
        for j in range(test_img.shape[1]):
            for k in range(test_img.shape[2]):
                if mask[i,j,k]==1:
                    mmm.append(test_image[i,j,k])
                    
    mfi = np.max(mmm)
    mmfi = np.min(mmm)
    pca_test_img = 1/(mfi-mmfi)*(pca_test_img-mmfi)
    
    return test_img, pca_test_img, mask, PCA, mfi, mmfi ## mfi : max value, mmfi : min value