from PIL import Image
from numpy import *
from scipy import signal
import numpy
from numpy import matrix
from scipy.ndimage import filters
from matplotlib import pyplot as plt
from pylab import *
from copy import copy, deepcopy


def p4(file): # creating a function to return the maximum total entropy for the selected image
	I = array(Image.open(file).convert('L')) # read image
	h = histogram(I,range(257)) # finding the histogram of the read image
	h = array(matrix(h)) # converting to array for further use
	n = h[0,0] # array n contains the histogram values since h = ([n0,n1,n2....n255],[0,1,2,....255]) is split
	i = h[0,1] # into teo arrays namely n = [n0,n1,n2....n255] and i = [0,1,2,....255]
	i = delete(i,256) # the function histogram creates L values while need L-1 values and hence neglecting the last digit which is 256
	
	N = 0
	for j in range(len(n)): #finding the total number of pixels
		N += n[j]
	
	N1 = [] # finding the total number of pixels for a given i
	tempn = 0
	for j in range(len(n)):
		tempn += n[j]
		N1.append(tempn)
	
	
	p = [] # the probability of a piven ixel 
	for j in range(len(n)):
		p.append(float(n[j])/N)
	sump = 0
	for j in range(len(n)):
		sump += p[j]
	
	
	T = i # the probability of a pixel having gray level i
	PT = []
	for j in range(len(T)):
		sumpt = 0
		for k in range(0,j):
			sumpt += float(n[k])/N
		PT.append(sumpt)
	
	A = [] # the probability of a pixel for given threshold t in the region A
	for j in range(256):
		try:
			A.append(p[j]/PT[j])
		except ZeroDivisionError:
			A.append(0.0)
	
	B = [] # the probability of a pixel for given threshold t in the region B, basically it takes the values other than the region A
	for j in range(256):
		try:
			B.append(p[j+1]/(1-PT[j]))
		except ZeroDivisionError:
			B.append(0.0)
		except IndexError:
			B.append(p[j]/(1-PT[j]))    
	
	
	HA = [] #empty array to store the entropy for the region A
	HB = [] #empty array to store the entropy for the region B
	for j in range(len(T)):
		t = T[j]
		suma=0 # to store summation values for A
		sumb=0 # to store summation values for B
		for k in range(t): #Finding the entropy for the region A
			tempa = 0
			if(k==0):
				tempa = tempa
			else:
				try:
					tempa = A[k]*math.log(A[k]) # error handling for log(0)
				except ValueError:
					tempa = 0
			suma+=tempa
		HA.append(-suma) # Appending each element to form an array
		for l in range(t+1,256): #Finding the entropy for the region A
			tempb = 0
			if(l==0):
				tempb = tempb 
			else:            
		          try:
		              tempb = B[l]*math.log(B[l]) # error handling for log(0)
		          except ValueError:
		              tempb = 0
			sumb+=tempb
		HB.append(-sumb) # Appending each element to form an array
	
	HT = [] #empty array to store the entropy for the region A
	for j in range(len(HA)):
		HT.append(HA[j]+HB[j]) # Adding element to element in order to achieve H(A)
	
	Tvalue = T[HT.index(max(HT))]
	print "The value of T corresponding to maximum total entropy H(T) for the image named %s is %d"%(file,Tvalue)
	
	out = deepcopy(I) #copying the input to apply thresholding
	for i in range(len(I[:,0])):    
		for j in range(len(I[0,:])):
			if(I[i,j]<Tvalue): #applying threshold value to obtain the binary image
				out[i,j] = 0
			else: out[i,j] = 255
    
       # plotting the binary image
        fig = plt.figure()
	a=fig.add_subplot(2,1,1)
	a.set_title('I, the original image')
	imgplot = plt.imshow(I,cmap=cm.gray)
	a=fig.add_subplot(2,1,2)
	a.set_title('The binary image after thresholding')
	imgplot = plt.imshow(out,cmap=cm.gray)
        show()
        return out,Tvalue
#=======================================Start==============================================
p4('gray1.jpg') #calling the defined function for thresholding by entropy
p4('yak.jpg')
p4('ppl.jpg')
#=======================================END==============================================