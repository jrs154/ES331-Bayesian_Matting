import numpy as np
import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import scipy.misc
from numba import jit
from scipy.cluster.vq import whiten
from sklearn import mixture
import warnings


#This function takes a pixel as input and returns a square window of dimension n*n around that pixel
@jit(nopython=True, cache=True)
def windowing(x, y, a, n):
    h, w = len(a), len(a[0])
    xl, xr = max(0, x - n//2), min(w, x + n//2 + 1)
    xlenl, xlenr = n//2 - x + xl, n//2 + xr - x
    yb, yu = max(0, y - n//2), min(h, y + n//2 + 1)
    ylenb, ylenu = n//2 - y + yb, n//2 + yu - y
    window = np.zeros((n, n, 3))
    #print(a[yb:yu, xl:xr])
    window[ylenb:ylenu, xlenl:xlenr] = a[yb:yu, xl:xr]
    
    return window


#This implemets all the mathematical calculations of Bayesian Matting
#It takes mean and covariance matrices of foreground and background as input and returns F, B and alpha
#The values returned by this function are the maximum possible
@jit(nopython=True, cache=True)
def maxProbability(meanFG, covFG, meanBG, covBG, opix, sigmaC, mean):
    
    I = np.identity(3)
    maxIterations, minProb = 1000, 1e-8
    fgMax, bgMax = np.zeros(3), np.zeros(3)
    invFG, invBG = np.linalg.inv(covFG), np.linalg.inv(covBG)
    x = invFG @ meanFG
    y = invBG @ meanBG
    C = np.atleast_2d(opix)
    mFG = np.atleast_2d(meanFG)
    mBG = np.atleast_2d(meanBG)
    aMax = 0
    
    myiter = 1
    prob = 0
    maxProb = -np.inf
    prevProb = np.inf
    
    while myiter < maxIterations and abs(prevProb - prob) > minProb:
        Axx = invFG + (I*mean**2)/sigmaC**2
        Axy = (I*mean*(1-mean))/sigmaC**2
        Ayy = invBG + (I*(1-mean)**2)/sigmaC**2
        Ax = np.hstack((Axx, Axy))
        Ay = np.hstack((Axy, Ayy))
        A = np.vstack((Ax, Ay))
        
        bx = x + (opix*mean)/(sigmaC**2)
        by = y + opix*(1-mean)/(sigmaC**2)
        bint = np.hstack((bx, by))
        b = np.atleast_2d(bint).T
        
        X = np.linalg.solve(A, b)           #Gives value of F and B
        F = np.maximum(0, np.minimum(1, X[0:3]))
        B = np.maximum(0, np.minimum(1, X[3:6]))
        fb = F - B
        sumfb2 = np.sum(fb**2)
        #solve for alpha
        lowest = np.minimum(1, ((C.T-B).T @ fb)/sumfb2)
        mean = np.maximum(0, lowest)[0,0]   #Gives value of alpha
       
        diffF = F - mFG.T
        diffB = B - mBG.T
        imgsum = mean*F + (1-mean)*B
        norm = np.sum((C.T - imgsum)**2)
        logC = -norm/sigmaC**2                      #Log(P(C|F,B,alpha))
        logF = (-(diffF.T @ invFG @ diffF)/2)[0,0]  #Log(P(F))
        logB = (-(diffB.T @ invBG @ diffB)/2)[0,0]  #Log(P(B))
        prob = logC + logF + logB           #Probability calculated in this iteration

        if prob > maxProb:                  #Checking whether this is the maximum probability obtained
            aMax = mean
            maxProb = prob
            fgMax = F.flatten()
            bgMax = B.flatten()

        prevProb = prob                     #Saving the probability of this iteration
        myiter += 1
    return fgMax, bgMax, aMax


img = imageio.imread("inputs/S18.png")[:, :, :3]        #Reading the input image
trimap = imageio.imread("trimaps/S18.png")[:, :, 0]     #Reading the trimap
matte = imageio.imread("answers/S18.png")[:, :, 0]      #Reading the true alpha matte of image
h, w, c = img.shape

normImg = img/255
sigma = 4
N = 21
minlen = 3

alpha = np.zeros((h, w))
fixFG, fixBG = np.zeros((h, w)), np.zeros((h, w))
fixFG[trimap == 255] = 1
fixBG[trimap == 0] = 1
unfixed = np.logical_not(np.logical_or(fixFG, fixBG))   #Unknown region
alpha[fixFG.astype(bool)] = 1
fixFG = np.repeat(fixFG[:, :, np.newaxis], 3, axis=2)
imgFG = normImg*fixFG                               #Known foreground
fixBG = np.repeat(fixBG[:, :, np.newaxis], 3, axis=2)
imgBG = normImg*fixBG                               #Known foreground

gaussKernel = cv.getGaussianKernel(N, sigma)        #Defining gaussian noise
normGaussKernel = gaussKernel/max(gaussKernel)

k = 1
alpha[unfixed] = -10
n = np.sum(unfixed)
temp = unfixed

kernel = np.ones((3, 3))
while k < n:
    temp = cv.erode(temp.astype(np.uint8), kernel, iterations=1)
    remPix = np.logical_xor(temp, unfixed).astype(int)
    remainPix = np.argwhere(remPix)             #List of pixels to be worked upon
    
    for [i, j] in remainPix:    
        opix = normImg[i, j]                    #Pixel of input image in consideration

        #Taking the surrounding pixels around our pixel in consideration 
        fpix = np.reshape(windowing(j, i, imgFG, N), (N**2, 3))
        bpix = np.reshape(windowing(j, i, imgBG, N), (N**2, 3))
        apix = windowing(j, i, alpha[:, :, np.newaxis], N)[:, :, 0]

        
        fwgts = (apix*normGaussKernel).flatten()
        bwgts = ((1-apix)*normGaussKernel).flatten()
        fvalid = fwgts > 0
        bvalid = np.logical_and((bwgts > 0), (bwgts <= 1))
        fwghts = np.repeat(fwgts[:, np.newaxis], 3, axis=1)
        bwghts = np.repeat(bwgts[:, np.newaxis], 3, axis=1)
        
        fpix = fpix[fvalid, :]
        fwgts = fwgts[fvalid]
        fwghts = fwghts[fvalid, :]                          #Gaussian weights for foreground

        bpix = bpix[bvalid, :]
        bwgts = bwgts[bvalid]
        bwghts = bwghts[bvalid, :]                          #Gaussian weights for foreground
        
        # if not enough data, return to it later...
        if len(fwgts) < minlen or len(bwgts) < minlen:
            continue
        
        #Clustering for foreground using gaussian mixture model
        gmm = mixture.GaussianMixture(n_components = c, covariance_type='full').fit(fpix*fwghts)
        muFG = np.sum(gmm.means_, axis=0)
        #mu_f = np.atleast_2d(gmm.means_[0])
        #sumw = np.sum(fwgts)
        muFG = muFG/3
        #print(gmm.covariances_)
        covFG = np.sum(gmm.covariances_, axis=0) 
        #print(covFG)
        covFG = covFG/3 + 1e-06
        #mu_f = mu_f.T
        #print(mu_f.shape)
        #print(sigma_f.shape)

        #Clustering for background using gaussian mixture model
        gmm = mixture.GaussianMixture(n_components = c, covariance_type='full').fit(bpix*bwghts)
        muBG = np.sum(gmm.means_, axis=0)
        muBG = muBG/3
        covBG = np.sum(gmm.covariances_, axis=0) 
        covBG = covBG/3 + 1e-06
        
        apix = apix.ravel()
        posInds = apix > 0
        apix = apix[posInds]
        mean = np.sum(apix)/len(apix)
        fgMax, bgMax, aMax = maxProbability(muFG, covFG, muBG, covBG, opix, 0.01, mean) #Obtaining the maximum F, B and alpha
        imgFG[i, j] = fgMax.flatten()
        imgBG[i, j] = bgMax.flatten()
        alpha[i, j] = aMax
        unfixed[i, j] = 0
    k += 1
#Uncomment to show foreground
'''
temp = img
#print(matte.shape)
for i in range(h):
    for j in range(w):
        if alpha[i][j] == 0:
            temp[i][j][0] = 0
            temp[i][j][1] = 0
            temp[i][j][2] = 0
'''

#Computing the absolute difference
abs_diff = np.sum(np.absolute(alpha-matte))
abs_diff = abs_diff/255
print(abs_diff)

#Showing the alpha matte of image obtaine0d
cv.imshow("Alpha matte", alpha)
plt.title("Foreground extracted")
plt.imshow(alpha, cmap='gray')
#plt.imshow(temp)
plt.show()




