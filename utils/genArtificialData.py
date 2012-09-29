'''
Created on Aug 5, 2011

@author: danu
'''

from matplotlib import pyplot as plt
import numpy as np

def twoGaussians(n):
	"""
	Generates 2d n data points from two Gaussians. 
	Returns an array of 2D vectors corresponding to the data points
	and a 1d array indicating the true labels of those data points.
	"""
	flag = np.zeros(n, dtype='float')
	randVals = np.random.rand(n)
	for i in range(0, n):
		if randVals[i] > 0.5:
			flag[i] = 1
		else:
			flag[i] = 0
	flag.sort()
	gaussian = np.random.randn(2, n)
	shift = np.zeros((2,n), dtype='float')
	shift[0,:] = flag * 2 - 1
	x = gaussian + 5 * shift
	ytrue = flag  
	# Centralization.
	meanVect = np.array([np.mean(x[0,:]), np.mean(x[1,:])])
	x = x - meanVect.reshape(-1, 1)
	# Normalization.
	stdVect = np.array([np.std(x[0,:]), np.std(x[1,:])])
	x = x / stdVect.reshape(-1,1)
	return x, ytrue

def fourGaussians(n):
	"""
	Generates 2d n data points from four Gaussians. 
	Returns an array of 2D vectors corresponding to the data points
	and a 1d array indicating the true labels of those data points.
	"""
	flag1 = np.zeros(n, dtype='float')
	randVals1 = np.random.rand(n)
	for i in range(0, n):
		if randVals1[i] > 0.5:
			flag1[i] = 1
		else:
			flag1[i] = 0
	flag2 = np.zeros(n, dtype='float')
	randVals2 = np.random.rand(n)
	for i in range(0, n):
		if randVals2[i] > 0.5:
			flag2[i] = 1
		else:
			flag2[i] = 0
	gaussian = np.random.randn(2, n)
	shift = np.zeros((2,n), dtype='float')
	shift[0,:] = flag1 * 4 - 2
	shift[1,:] = flag2 * 4 - 2
	x = 0.5 * gaussian + shift
	ytrue = flag1 + flag2 * 2
	# Centralization.
	meanVect = np.array([np.mean(x[0,:]), np.mean(x[1,:])])
	x = x - meanVect.reshape(-1, 1)
	# Normalization.
	stdVect = np.array([np.std(x[0,:]), np.std(x[1,:])])
	x = x / stdVect.reshape(-1,1)
	return x, ytrue

def spiral(n):
	"""
	Generates two spirals. The datapoints are given by vectors in
	x and their corresponding class labels in ytrue.
	"""
	x = np.zeros((2,n), dtype='float')
	for i in range(0, int(n / 2)):
		r = 1 + (4 * float(i -1)) / float(n)
		t = (np.pi * float(i - 1) * 3) / float(n)
		x[0,i] = r * np.cos(t)
		x[1,i] = r * np.sin(t)
		x[0, i + int(n / 2)] = r * np.cos(t + np.pi)
		x[1, i + int(n / 2)] = r * np.sin(t + np.pi)
	x = x + 0.1 * np.random.randn(2, n)
	ytrue = np.zeros(n, dtype='float')
	for i in range(int(n / 2), n):
		ytrue[i] = 1
	# Centralization.
	meanVect = np.array([np.mean(x[0,:]), np.mean(x[1,:])])
	x = x - meanVect.reshape(-1, 1)
	# Normalization.
	stdVect = np.array([np.std(x[0,:]), np.std(x[1,:])])
	x = x / stdVect.reshape(-1,1)
	return x, ytrue

def highLowDensities(n):
	"""
	Creates a dataset with two Gaussians, where one has a high
	density and the other has a low density. Data points are
	arranged in column arrays in x, and their corresponding
	class labels are given in ytrue.
	"""
	gaussian = np.random.randn(2, int(n / 2))
	x = np.zeros((2,n), dtype='float')
	x[:,:int(n / 2)] = 0.1 * gaussian
	x[:,int(n / 2):] = gaussian
	ytrue = np.zeros(n)
	ytrue[int(n / 2):] = np.ones(int(n / 2))
	# Centralization.
	meanVect = np.array([np.mean(x[0,:]), np.mean(x[1,:])])
	x = x - meanVect.reshape(-1, 1)
	# Normalization.
	stdVect = np.array([np.std(x[0,:]), np.std(x[1,:])])
	x = x / stdVect.reshape(-1,1)
	return x, ytrue


def circleAndGaussian(n):
	"""
	Generates a circle and a Gaussian.
	Data points are arranged as 2d column vectors in x,
	and their corresponding labels are given in ytrue.
	"""
	x = np.zeros((2,n), dtype='float')
	x[0,: (n / 2)] = 5 * np.cos(np.linspace(0, 2 * np.pi, n / 2))
	x[0, (n / 2):] = np.random.randn(1, n / 2)
	x[1,: (n / 2)] = 5 * np.sin(np.linspace(0, 2 * np.pi, n / 2))
	x[1, (n / 2):] = np.random.randn(1, n / 2)
	x = x + 0.1 * np.random.randn(2, n)
	ytrue = np.zeros(n, dtype='float')
	ytrue[(n / 2):] = np.ones((n / 2), dtype='float')
	# Centralization.
	meanVect = np.array([np.mean(x[0,:]), np.mean(x[1,:])])
	x = x - meanVect.reshape(-1, 1)
	# Normalization.
	stdVect = np.array([np.std(x[0,:]), np.std(x[1,:])])
	x = x / stdVect.reshape(-1,1)
	return x, ytrue	


def writeDataset(fname, x, y, n):
	"""
	Writes the dataset to a feature vector file.
	"""	
	F = open(fname, 'w')
	for i in range(0,n):
		lbl = 2* y[i] - 1 
		#lbl = y[i] + 1
		F.write("%d 1:%f 2:%f\n" % (lbl, x[:,i][0], x[:,i][1]))
	F.close()
	pass	

def main():
	dataSet = 5
	n = 2000
	
	if dataSet == 1:
		x, ytrue = twoGaussians(n)
	elif dataSet == 2:
		x, ytrue = fourGaussians(n)
	elif dataSet == 3:
		x, ytrue = spiral(n)
	elif dataSet == 4:
		x, ytrue = highLowDensities(n)
	elif dataSet == 5:
		x, ytrue = circleAndGaussian(n)
	print "Plotting..."	
	symbols = ['bo', 'rx', 'g*', 'ks']
	for i in range(0,n):
		plt.plot(x[:,i][0], x[:,i][1], symbols[int(ytrue[i])])
	plt.show()
	
	writeDataset("circleAndGuassian", x, ytrue, n)
	pass

if __name__ == "__main__":
	main()

