# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
from scipy import signal
import numpy
from numpy import matrix
from scipy.ndimage import filters
from matplotlib import pyplot as plt
from pylab import *
from copy import copy, deepcopy


def gmask (x,y,s): # the function for gaussian filter
    gmask = (1/(math.sqrt(2*(math.pi))*s))*numpy.exp(-((x**2) + (y**2))/2/s**2)
    return gmask
    
def gmask1 (x,y,s,z): # function implementing the first derivative of the gaussian filter
    if(z =='x'):
        gmask1 = gmask(x,y,s)*(-x/(s**2))
    elif(z=='y'):
        gmask1 = gmask(x,y,s)*(-y/(s**2))
    return gmask1

def p1draw (I,Ix,Iy,Ix1,Iy1,M,M1,H): # function used to plot the imtermediate images we get in later stages
        fig = plt.figure()
	a=fig.add_subplot(3,3,1)
	a.set_title('I, the original image')
	imgplot = plt.imshow(I,cmap=cm.gray)
	a=fig.add_subplot(3,3,2)
	a.set_title('Ix = I * G --> x direction')
	imgplot = plt.imshow(Ix,cmap=cm.gray)
	a=fig.add_subplot(3,3,3)
	a.set_title('Iy = I * G --> y direction')
	imgplot = plt.imshow(Iy,cmap=cm.gray)
	a=fig.add_subplot(3,3,5)
	a.set_title('Ix` = Ix * Gx')
	imgplot = plt.imshow(Ix1,cmap=cm.gray)
	a=fig.add_subplot(3,3,6)
	a.set_title('Iy` = Iy * Gy')
	imgplot = plt.imshow(Iy1,cmap=cm.gray)
	a=fig.add_subplot(3,3,7)
	a.set_title('M, the magnitude of edge response')
	imgplot = plt.imshow(M,cmap=cm.gray)
	a=fig.add_subplot(3,3,8)
	a.set_title('Image after non-maximum suppression')
	imgplot = plt.imshow(M1,cmap=cm.gray)
	a=fig.add_subplot(3,3,9)
	a.set_title('Image with Edges alone')
	imgplot = plt.imshow(H,cmap=cm.gray)
	show()

# function implementing all the steps mentioned in problem one
def p1(s, # to choose Standard Deviation
size, # to choose the size of the kernel
file, # to choose the image from working directory
h, # High threshold value for hysteresis thresholding
l, # Low threshold value for hysteresis thresholding
noise, # to check if the input image has already been through a gaussian/saltnpepper noise distribution
plotreq # decides if the plotting of the image is required or not 
):        
	# Step 1: To read the given image
	if(noise == 0):
   	    I = array(Image.open(file).convert('L'))
   	elif(noise == 1):
   	    I = file
	# Step 2: To create a 1D-Gaussian kernel
	G = []
	for i in range(-size,size+1):
		G.append(gmask(i,0,s)) # equating y to 0 since we need a 1D matrix
	# Step 3: To create a first derivative of Gaussian kernel in both x and y directions
	Gx = []
	for i in range(-size,size+1):
		Gx.append(gmask1(i,0,s,'x')) 
	
	Gy = []
	for i in range(-size,size+1):
		Gy.append([gmask1(0,i,s,'y')]) 
	
	# Step 4 : To obtain Ix = I * G along x and y direction seperately. where * = convolution
	Ix = []
	for i in range(len(I[:,0])):
		Ix.extend([numpy.convolve(I[i,:],G)]) # I*G ----> x direction
	Ix = array(matrix(Ix)) # since the rows are stored as tuples in Ix we convert it to matrix. we also need to keep in mind that 
	#the function numpy convolve cannot be used on a matrix. Hence, we renistate the array but this time, the values are stored as
	#integers and not tuple.
	
	Iy = []
	for i in range(len(I[0,:])):
		Iy.extend([numpy.convolve(I[:,i],G)]) # I*G ----> y direction
	Iy = array(matrix(transpose(Iy))) 
	# since append can add only rows, we take each column as a row, convolve it with G and finally reinstate that array as a column
	
	#Step 5 : To obtain Ix' = Ix * Gx i.e to convolve the x component of our image matrix I with the derivative of the Gaussian mask
	Ix1 = []
	for i in range(len(Ix[:,0])):
		Ix1.extend([numpy.convolve(Ix[i,:],Gx)]) # Ix * Gx ----> x direction
	Ix1 = array(matrix(Ix1)) 
	
	Iy1 = []  
	for i in range(len(Iy[0,:])):
		Iy1.extend([numpy.convolve(Iy[:,i],Gx)]) # Iy * Gy ----> y direction
	Iy1 = array(matrix(transpose(Iy1))) 
	
	# Step 6: To compute the magnitude of the edge response
	M = []
	for i in range(len(Ix1[:,0])):    
		temp = []
		for j in range(len(Iy1[0,:])):    
			temp.append(math.sqrt((Ix1[i,j])**2+(Iy1[i,j])**2))
			if (j == len(Iy1[0,:])-1):
				M.extend(array(matrix(temp)))
	M = array(matrix(M))
	# to compute the direction of gradient as θ = arc tan inverse (I'y/I'x)	
	T = []         
	for i in range(len(Ix[:,0])):    
		temp = []
		for j in range(len(Iy[0,:])):    
			temp1 = math.degrees(math.atan2(I[i,j],I[i,j]))
                        temp1 = (temp1 + 360) % 360
                        temp.append(temp1)
			if (j == len(Iy[0,:])-1):
				T.extend(array(matrix(temp)))
	T = array(matrix(T)) 
	M1 = deepcopy(M)                     
	# step 7: To implement non-maximum supression algorithm
	for i in range(len(M[:,0])-1):    
		for j in range(len(M[0,:])-1):   
			test = T[i,j]
			test1 = M[i,j] 
			if((0<test<math.radians(22.5)) | (math.radians(337.6) < test< math.radians(360)) | (math.radians(157.6)<test<math.radians(202.5))):
				if not ((test1>M[i+1,j]) & (test1>M[i-1,j])):
					M1[i,j] = 0
			elif ((math.radians(22.6) < test < math.radians(67.5)) | (math.radians(202.6) < test < math.radians(247.5))):
				if not ((test1>M[i+1,j+1]) & (test1>M[i-1,j-1])):
					M1[i,j] = 0
			elif ((math.radians(67.6) < test < math.radians(112.5)) | (math.radians(247.6) < test < math.radians(292.5))):        
				if not ((test1>M[i,j+1]) & (test1>M[i,j-1])):
					M1[i,j] = 0
			elif ((math.radians(112.6) < test < math.radians(157.5)) | (math.radians(292.6) < test < math.radians(337.5))):
				if not ((test1>M[i-1,j+1]) & (test1>M[i+1,j-1])):
					M1[i,j] = 0  
	M1 = T = array(matrix(M1))
	# Step 8: To apply hysterisis thresholding                  
	H = deepcopy(M1)
	qx = 0
	qy = 0
	m = max(H.max(axis = 1))
	for i in range(len(H[:,0])-1):    
		for j in range(len(H[0,:])-1):
			qx = i
			qy = j
			while((qx!=0)&(qy!=0)):
				if (H[qx,qy]>=h):
				        H[qx,qy] = m
					try:
						if (l<=H[qx+1,qy]<h):
							H[qx+1,qy] = m
							qx = qx+1
							qy = qy			
						elif (l<=H[qx-1,qy]<h):
							H[qx-1,qy] = m
						        qx = qx-1 
							qy = qy
						elif (l<=H[qx+1,qy+1]<h):
							H[qx+1,qy+1] = m
							qx = qx+1
							qy = qy+1
						elif (l<=H[qx-1,qy-1]<h):
							H[qx-1,qy-1] = m
							qx = qx-1
							qy = qy-1
						elif (l<=H[qx,qy+1]<h):
							H[qx,qy+1] = m
							qx = qx
							qy = qy+1
						elif (l<=H[qx,qy-1]<h):
							H[qx,qy-1] = m
							qx = qx
							qy = qy-1
						elif (l<=H[qx-11,qy+1]<h):
							H[qx-1,qy+1] = m
							qx = qx-1
							qy = qy+1
						elif (l<=H[qx+1,qy-1]<h):
							H[qx+1,qy-1] = m
							qx = qx+1
							qy = qy-1
						else: 
							qx = 0
							qy = 0
					except IndexError:
							qx = 0
							qy = 0
				elif (l<=H[qx,qy]<h):
					H[qx,qy] = m
				else:
					H[qx,qy] = 0
                                        qx = 0
                                        qy = 0
	# Finally, to plot the image
        if (plotreq == 1):
            p1draw (I,Ix,Iy,Ix1,Iy1,M,M1,H)
        else:
            pass
        print max(M1.max(axis = 1))
        print min(M1.min(axis = 1))
        return H

#=====================================start==================================
s = 1.4 # defining the value for standard deviation
size = 1 # defining the size of the gaussian kernel as size = 2(input)+1
p1(s,size,'flag.jpg',9.366,4.3,0,1) #calling the defined procedure for canny
p1(s,size,'gray1.jpg',9.5,2.5,0,1) #calling the defined procedure for canny
p1(s,size,'elephant.jpg',7,2.5,0,1) #calling the defined procedure for canny
#=====================================END==================================