# Written 2018-11-14 by Aaron Goldfogel
# Many useful functions for CLT analysis (for AA 532)

import configparser
import numpy as np
import math
import re
import matplotlib.pyplot as plt

def sin(t):
	return math.sin(t)
def cos(t):
	return math.cos(t)

def inputsFromFile(fileName = 'inputs.ini', strengthParams = False):
	# Take inputs from inputs.ini
	config = configparser.ConfigParser()
	config.read(fileName)

	e1 = config.getint('material_properties', 'e1')
	e2 = config.getint('material_properties', 'e2')
	v12 = config.getfloat('material_properties', 'v12')
	g12 = config.getfloat('material_properties', 'g12')

	plySequenceString = config.get('geometry', 'sequence')
	h = config.getfloat('geometry', 'h')
	if (h == 0):
		t = config.getfloat('geometry', 't')
		n = config.getint('geometry', 'n')
		h = t*n


	nx = config.getfloat('loading', 'nx')
	ny = config.getfloat('loading', 'ny')
	nxy = config.getfloat('loading', 'nxy')
	mx = config.getfloat('loading', 'mx')
	my = config.getfloat('loading', 'my')
	mxy = config.getfloat('loading', 'mxy')

	if (strengthParams):
		s1ten  = config.getint('strength', 's1ten')
		s1comp = config.getint('strength', 's1comp')
		s2ten  = config.getint('strength', 's2ten')
		s2comp = config.getint('strength', 's2comp')
		s12    = config.getint('strength', 's12')

		return e1, e2, v12, g12, plySequenceString, h, nx, ny, nxy, mx, my, mxy, s1ten, s1comp, s2ten, s2comp, s12

	return e1, e2, v12, g12, plySequenceString, h, nx, ny, nxy, mx, my, mxy

def buildQ(e1, e2, v12, g12, verbose = True):
	# Build [Q]
	###########
	d = 1 - (e2/e1)*(v12**2)

	q = np.array([
		[e1/d, v12*e2/d, 0],
		[v12*e2/d, e2/d, 0],
		[0, 0, g12]
	])

	if(verbose):
		print('[Q]:')
		print(q)

	return q

def parsePlySequence(plySequenceString):	
	plySequence = list(map(int, re.findall(r'-?\d+', plySequenceString))) # get the numbers in the string
	thetaSequence = np.radians(plySequence) # convert theta from degrees to radians

	return thetaSequence

def buildQBarSequence(thetaSequence, q):
	# Build [Q_bar] for each ply
	############################

	qBarSequence = [] #To be filled with qBar matrices

	for theta in thetaSequence:
		tSigmaInverse = np.array([
			[cos(theta)**2, sin(theta)**2, -2*sin(theta)*cos(theta)],
			[sin(theta)**2, cos(theta)**2, 2*sin(theta)*cos(theta)],
			[sin(theta)*cos(theta), -1*sin(theta)*cos(theta), cos(theta)**2-sin(theta)**2]
		])

		tEpsilon = np.array([
			[cos(theta)**2, sin(theta)**2, sin(theta)*cos(theta)],
			[sin(theta)**2, cos(theta)**2, -1*sin(theta)*cos(theta)],
			[-2*sin(theta)*cos(theta), 2*sin(theta)*cos(theta), cos(theta)**2-sin(theta)**2]
		])

		qBar = tSigmaInverse.dot(q).dot(tEpsilon)
		qBarSequence.append(qBar)

	return qBarSequence

def buildABD(thetaSequence, qBarSequence, h, verbose = True):
	# Build [A], [B], [D]
	#####################
	z = [x*h/len(thetaSequence)-h/2 for x in range(len(thetaSequence))]
	z.append(h/2)
	# z is now filled with the z locations of each ply boundary, including both ends

	a = np.zeros((3,3))
	b = np.zeros((3,3))
	d = np.zeros((3,3))

	for i in range(3):
		for j in range(3):
			for k in [x+1 for x in range(len(thetaSequence))]: 
				#k is integers between 1 and the number of plies
				#z is kind-of 1-indexed, because z[0] is the bottom.
				a[i,j] += qBarSequence[k-1][i,j]*(z[k]-z[k-1])
				b[i,j] += qBarSequence[k-1][i,j]*(z[k]**2-z[k-1]**2)/2 #I suppose dividing each term of a sum is inefficient but it's harder to forget this way
				d[i,j] += qBarSequence[k-1][i,j]*(z[k]**3-z[k-1]**3)/3

	if(verbose):
		print('\n[A]:')
		print(a)
		print('\n[B]:')
		print(b)
		print('\n[D]:')
		print(d)

	return a, b, d

def getMidplaneStrains(a, b, d, nx, ny, nxy, mx, my, mxy, verbose = True):
	# Strains and curvature of the mid-plane
	########################################

	globalStiffnessMatrix = np.vstack((np.hstack((a,b)),np.hstack((b,d)))) 
	#[[A, B],
	# [B, D]]

	if (verbose):
		print('\nGlobal Stiffness Matrix:')
		print(globalStiffnessMatrix)

	loadVector = np.array([
		[nx],
		[ny],
		[nxy],
		[mx],
		[my],
		[mxy]
	])

	globalComplianceMatrix = np.linalg.inv(globalStiffnessMatrix) #inverse

	# print('\nGlobal Compliance Matrix:')
	# print(globalComplianceMatrix)

	midplaneVector = globalComplianceMatrix.dot(loadVector)
	midplaneStrains = midplaneVector[0:3, 0] #top half
	curvature = midplaneVector[3:6, 0] #bottom half

	if(verbose):
		print('\nMid-plane Strains:')
		print(midplaneStrains)
		print('\nCurvature:')
		print(curvature)

	return midplaneStrains, curvature

def allStrains(midplaneStrains, curvature, z, plotPoints):
	# Strains throughout the laminate thickness
	###########################################

	allStrains = np.zeros((3,plotPoints))
	for i in range(plotPoints):
		allStrains[:, i] = (midplaneStrains + z[i]*curvature).flatten() #flatten() shouldn't be necessary but it is.

	return allStrains

def allStresses(allStrains, qBarSequence, thetaSequence, plotPoints):
	# Stresses throughout the laminate thickness
	############################################
	allStresses = np.zeros((3, plotPoints))
	for i in range(plotPoints):
		qMap = int(i*len(thetaSequence)/plotPoints) #figures out which z bin we're in
		allStresses[:, i] = qBarSequence[qMap].dot(allStrains[:, i]).flatten()

	return allStresses

def maximumStressCriterion(sigma1, sigma2, tau12, s1ten, s1comp, s2ten, s2comp, s12):
	# Returns False if it's outside the envelope, True if it's inside the envelope.
	if (sigma1 > s1ten or
		sigma1 < -s1comp or
		sigma2 > s2ten or
		sigma2 < -s2comp or
		abs(tau12) > s12):
		return False
	return True

def maximumStressCriterionGrid(sigma1, sigma2, tau12, s1ten, s1comp, s2ten, s2comp, s12):
	toReturn = np.zeros_like(sigma1)
	for i in range(len(toReturn)):
		for j in range(len(toReturn[i])):
			toReturn[i,j] = maximumStressCriterion(sigma1[i,j], sigma2[i,j], tau12[i,j], s1ten, s1comp, s2ten, s2comp, s12)

	return toReturn

def findMaxStrains(q, s1ten, s1comp, s2ten, s2comp, s12):
	qinv = np.linalg.inv(q)
	eta1ten  = np.dot(qinv, np.array([s1ten, 0, 0]))[0]
	eta2ten  = np.dot(qinv, np.array([0, s2ten, 0]))[1]
	eta1comp = np.dot(qinv, np.array([s1comp, 0, 0]))[0]
	eta2comp = np.dot(qinv, np.array([0, s2comp, 0]))[1]

	eta12 = np.dot(qinv, np.array([0, 0, s12]))[2]

	return eta1ten, eta1comp, eta2ten, eta2comp, eta12

def stressFromStrains(q, epsilon1, epsilon2, gamma12):
	return np.dot(q, np.array([epsilon1, epsilon2, gamma12]))

def maximumStrainCriterion(sigma1, sigma2, tau12, eta1ten, eta1comp, eta2ten, eta2comp, eta12, qinv):
	# Returns False if it's outside the envelope, True if it's inside the envelope.
	strains = np.dot(qinv, np.array([sigma1, sigma2, tau12]))
	epsilon1 = strains[0]
	epsilon2 = strains[1]
	gamma12  = strains[2]

	if (epsilon1 > eta1ten or
		epsilon1 < -eta1comp or
		epsilon2 > eta2ten or
		epsilon2 < -eta2comp or
		abs(gamma12) > eta12):
		return False
	return True

def maximumStrainCriterionGrid(sigma1, sigma2, tau12, eta1ten, eta1comp, eta2ten, eta2comp, eta12, q):
	toReturn = np.zeros_like(sigma1)
	for i in range(len(toReturn)):
		for j in range(len(toReturn[i])):
			toReturn[i,j] = maximumStrainCriterion(sigma1[i,j], sigma2[i,j], tau12[i,j], eta1ten, eta1comp, eta2ten, eta2comp, eta12, np.linalg.inv(q))

	return toReturn

def tsaiWuCriterion(s1ten, s1comp, s2ten, s2comp, s12, sigma1, sigma2, tau12):
	# Predicts failure for return values >1
	F1 = 1/s1ten - 1/s1comp
	F2 = 1/s2ten - 1/s2comp
	F11 = 1/(s1ten*s1comp)
	F22 = 1/(s2ten*s2comp)
	F66 = 1/(s12**2)
	F12 = -1/2*math.sqrt(F11*F22)

	return F1*sigma1 + F2*sigma2 + F11*(sigma1**2) + F22*(sigma2**2) + F66*(tau12**2) + 2*F12*sigma1*sigma2

def stressTransformation(sigmaX, sigmaY, tauXY, z, thetaSequence, h):
	# Finds local stress (in-ply) given global stress
	layer = int((z+h/2)/h*len(thetaSequence)) #Which ply are we in?
	theta = thetaSequence[layer] #What's theta at that ply? (radians)

	rotationMatrix = np.array([
		[cos(theta), sin(theta)],
		[-sin(theta), cos(theta)]
	]) # -theta because we're rotating back from theta to 0.

	# localStress = rotationMatrix.globalStressTensor.rotationMatrixTranspose
	globalStressTensor = np.array([
		[sigmaX, tauXY],
		[tauXY, sigmaY]
	])
	localStress = np.dot(rotationMatrix, np.dot(globalStressTensor, np.transpose(rotationMatrix)))
	sigma1 = localStress[0,0]
	sigma2 = localStress[1,1]
	tau12  = localStress[0,1]

	return sigma1, sigma2, tau12






