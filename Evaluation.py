# -*- coding: utf-8 -*-
from numpy import array,math,ndarray,random
from problem1 import p1
from PIL import Image
import numpy
from copy import copy, deepcopy
from numpy import matrix

def myQE (
fileout,# Given image for comparision
filein,# output image of my canny algorithm
s, # standard deviation
size, # kernel size
Th, # high threshold
Tl, # low threshold
noise # with/without noise?
): 
    J = array(Image.open(fileout).convert('L'))
    if(noise == 0):
        I = p1(s,size,filein,Th,Tl,0,1) # calling my canny algorithm from problem 1
    elif(noise == 1): # decides if noise is required to be added or not
        I = p1(s,size,filein,Th,Tl,1,1) # calling my canny algorithm from problem 1
    tp,tn,fp,fn = (0.0,0.0,0.0,0.0) # initializing the true/false positives/negatives
    e = 1.6
    for i in range(len(I[:,0])):    
	for j in range(len(I[0,:])):    
		if ((I[i,j]>0) & (J[i,j]>0)): # implementing true positive
		  tp+=1
		elif((I[i,j]==0) & (J[i,j]==0)): # implementing true negetive
		  tn+=1
		elif((I[i,j]>0) & (J[i,j]==0)): # implementing false positive
		  fp+=1
		elif((I[i,j]==0) & (J[i,j]>0)): # implementing false negetive
		  fn+=1
    sen = tp/(tp+fn) # Sensitivity
    spe = tn/(tn+fp) # Specificity
    pr = tp/(tp+fp) # Precision
    npv = tn/(tn+fn) # Negative Predictive Value
    fot = fp/(fp+tn) # Fall Out 
    fnr = fn/(fn+tp) # False Negative Rate FNR
    fdr = fp/(fp+tp) # False Discovery Rate FDR    
    acc = (tp + tn)/(tp+fn+tn+fp) # Accuracy
    fsc = (2*tp)/((2*tp)+fp+fn) # F-score
    mcc = ((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) # Matthew’s Correlation Coefficient
    
    print "Finished quantitative evaluation of edge detection\nSensitivity = %f\nSpecificity = %f\nPrecision = %f\nNegative Predictive Value = %f\nFall-out = %f\nFalse Negative Rate = %f\nFalse Discovery Rate = %f\nAccuracy = %f\nF-score = %f\nMatthew’s Correlation Coefficient = %f\n"%(sen,spe,pr*e,npv,fot,fnr,fdr,acc,fsc,mcc)

def Gnoise(I, # input image to which the noise is to be added
m,# Required mean
s # Required standard deviation
):
    x = I + random.normal(m,s,(len(I[:,0]),len(I[0,:]))) # noise addition to input image
    return x

def snpnoise(I, # input image to which the noise is to be added
s # Required standard deviation
):
    x = []
    for i in range(len(I[:,0])):
        temp = []    
	for j in range(len(I[0,:])):
	    temp.append(round(random.rand(1),4))
	    if (j == len(I[0,:])-1):
	           x.append(temp)
    x = array(matrix(x))
    out = deepcopy(I)
    for i in range(len(x[:,0])):    # noise addition to input image
	for j in range(len(x[0,:])):    
	   if(x[i,j]<(s/2)): # deciding whether to add black dots or white
	       out[i,j] = 0
	   elif((x[i,j] >= (s/2)) & (x[i,j]<s)):
	       out[i,j] = 255
    return out

#=======================================Start==============================================
myQE('p2_out_1.png','p2-1.jpg',1.5,1,7.5,.5,0) #calling the defined procedure for processing
myQE('p2_out_2.png','p2-2.jpg',1.5,1,7.19,.5,0) #calling the defined procedure for processing
I = array(Image.open('p2-1.jpg').convert('L')) # read image
Ig = Gnoise(I,0,10)# Add Gaussian noise with mean = 0 and sd = 10
Isnp = snpnoise(Ig,0.025) # Add Salt n pepper noise with sd = 25%
myQE('p2_out_1.png',Isnp,1.5,1,7.2,1.0005,1) #calling the defined procedure for processing
I1 = array(Image.open('p2-2.jpg').convert('L')) # read image
Ig1 = Gnoise(I1,0,10)# Add Gaussian noise with mean = 0 and sd = 10
Isnp1 = snpnoise(Ig1,0.025) # Add Salt n pepper noise with sd = 25%
myQE('p2_out_2.png',Isnp1,1.5,1,7.2,1.0005,1) #calling the defined procedure for processing
#=======================================END==============================================