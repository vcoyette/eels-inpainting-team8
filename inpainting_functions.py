import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import random as rd
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils.ssim as SSIM
import inpystem
import hyperspy.api as hs
import pathlib
import inquirer
import inpystem.tools.metrics as mt
import skimage.metrics as skm
from skued import dmread
###############################################################################

#from __future__ import print_function
#from deep_image_prior.models.resnet import ResNet
#from deep_image_prior.models.unet import UNet
#from deep_image_prior.models.skip import skip
#import torch.optim
#from deep_image_prior.utils.inpainting_utils import *

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
        partial_img = 1/(mpi-mmpi)*(partial_img-mmpi)

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

def inverse_pca(img, PCA):

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

    recovered_img = np.transpose(PCA.inverse(np.transpose(img, (1, 2, 0))), (2, 0, 1))

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

    if len(img.shape)==2:
        img=np.array([img])
    plt.rcParams['figure.figsize'] = [20, 30]
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
        partial_img = 1/(mpi-mmpi)*(partial_img-mmpi)

        l = percentage_variance(img, mask)

        print('Weights calculated with success !')

    return partial_img, mask, l, PCA_

###############################################################################

def numpy_metric(real_image, filled_image, a, b, c, type_):

    '''
        Takes as inputs :

            - real_image : The real image, format --> numpy whose shape is (1,Spectrum_Size,m,n),
            - filled_image : The image filled by the estimator, format --> numpy whose shape is (1,Spectrum_Size,m,n),

            - a,b,c : 3 parameters a,b,c >= 0 such as if type_== 'sum', master_metric = a x PSNR + b x SSIM + c x SAD,
                                                      if type_== 'product', master_metric = PSNR^a * SSIM^b * SAD^c,
            - type_ : 'sum' (if so, a+b+c=1) or 'product'.

        Returns :

            Return a metric which is a combinaison of the PSNR, the SSIM and the SAD metric (if a=1 and b,c=(0,0), the metric
            is equivalent to PSNR, if b=1 and a,c=(0,0), the metric is equivalent to SSIM and so on).

    '''

    metric = master_metric(torch.tensor(real_image), torch.tensor(filled_image), a, b, c, type_).item()
    return metric
