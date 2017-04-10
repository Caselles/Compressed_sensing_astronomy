import numpy as np
import scipy.linalg as lng 
import copy as cp

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	

def mad(xin = 0):

	z = np.median(abs(xin - np.median(xin)))/0.6735
	
	return z
	
################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	

def GenerateMixture(n=2,t=1024,m=2,p=0.02,SType=1,CdA = 1,noise_level = 120):

	import numpy as np
	import scipy.linalg as lng 
	
	A = np.random.randn(m,n)
	uA,sA,vA = np.linalg.svd(A)
	sA = np.linspace(1/CdA,1,n) 
	A = np.dot(uA[:,0:n],np.dot(np.diag(sA),vA[:,0:n].T))
 
	S = np.zeros((n,t))
	
	if SType == 0:
		#print('Generating Bernoulli-Gaussian sources')
			
		K = np.floor(p*t)
			
		for r in range(0,n):
			u = np.arange(0,t)
			np.random.shuffle(u)
			S[r,u[0:K]] = np.random.randn(K)
			
	elif SType == 1:
		#print('Gaussian sources')
		
		S = np.random.randn(n,t)
		
	elif SType == 2:
		#print('Uniform sources')

		S = np.random.rand(n,t) - 0.5
		
	elif SType == 4:
		#print('SPC sources')
		
		K = np.floor(p*t)
		Kc = np.floor(K*0.3) #--- 30% have exactly the same locations
			
		u = np.arange(0,t)
		np.random.shuffle(u)
		S[:,u[0:Kc]] = np.random.randn(n,Kc)
			
		for r in range(0,n):
			v = Kc + np.arange(0,t - Kc)
			df = K-Kc
			np.random.shuffle(v)
			v = v[0:df]
			v = v.astype(np.int64)
			S[r,u[v]] = np.random.randn(df)
		
	elif SType == 3:
		#print('Approx. sparse sources')
		
		S = np.random.randn(n,t)
		S = np.power(S,3)
		
	else:
		print('SType takes an unexpected value ...')
		
	
	X = np.dot(A,S)
	
	if noise_level < 120:
		#--- Add noise
		N = np.random.randn(m,t)
		N = np.power(10.,-noise_level/20.)*lng.norm(X)/lng.norm(N)*N
		X = X + N
	
	return X,A,S

################# CODE TO PERFORM A PCA

def Perform_PCA(X,n):
	
	Y = cp.copy(X)
	Yp = np.dot(np.diag(np.mean(Y,axis=1)),np.ones(np.shape(Y)))
	Y = Y - Yp

	Ry = np.dot(Y,Y.T)
	Uy, Dy, Vy = np.linalg.svd(Ry)
	
	Uy = Uy[:,0:n]
	Sy = np.dot(np.diag(1./(1e-12 + np.sqrt(Dy[0:n]))),np.dot(Uy.T,X))
	
	return Uy , Sy

################ END OF PCA

################# CODE TO PERFORM ILC

def Perform_ILC(X,colcmb):
	
	# - Remove the mean value
	
	Y = cp.copy(X)
	Yp = np.dot(np.diag(np.mean(Y,axis=1)),np.ones(np.shape(Y)))
	Y = Y - Yp

	iRy = lng.inv(np.dot(Y,Y.T))

	c = np.reshape(colcmb,(len(colcmb),1))
	
	w = 1./np.dot(c.T,np.dot(iRy,c)) * np.dot(c.T,iRy)
	
	s = np.dot(w,X)
	
	return s,w

################ END OF ILC

################# CODE TO PERFORM FastICA

def Perform_FastICA(Y,n):

	from sklearn.decomposition import FastICA
	
	X = cp.copy(Y)
	X = X.T
	
	fpica = FastICA(n_components=n)

	S = fpica.fit(X).transform(X).T  # Get the estimated sources
	A = fpica.mixing_  # Get estimated mixing matrix

	return A , S

################ END OF FastICA

################# CODE TO PERFORM BASIC GMCA	

def Perform_GMCA(X,n,nmax=100,mints=0.5,maxts = 0):

	z = np.shape(X)
	t = z[1]
	m = z[0]

	A = np.random.randn(m,n)
	for r in range(0,n):
		A[:,r] = A[:,r]/lng.norm(A[:,r])
		
	Ra = np.dot(A.T,A)
	S = np.dot(np.diag(1./np.diag(Ra)),np.dot(A.T,X))	

	ts = 0
	if maxts == 0:
		for r in range(1,n):
			maxts = np.max([np.max(abs(S[r,:]))/mad(S[r,:]),ts])
				
	vts = np.exp((np.log(maxts) - np.log(mints))*np.linspace(0,1,nmax)[::-1] + np.log(mints))
	vepsilon = np.power(10,4*np.linspace(0,1,nmax)[::-1] - 5)
	
	for nit in range(0, nmax):
		
		# Estimate the sources for fixed A
		
		epsilon = vepsilon[nit]
		ts = vts[nit]
		
		Ra = np.dot(A.T,A)
		mRa = lng.norm(Ra,2)
		
		Ra = Ra + epsilon*mRa*np.identity(n)
		PinvA = np.dot(lng.inv(Ra),A.T)
		S = np.dot(PinvA,X)
		
		for ns in range(0, n):
			temp = S[ns,:]
			thrd = ts*mad(temp)
			S[ns,(abs(temp) < thrd)] = 0	
			
				
		# Estimate the mixing matrix for fixed sources S
		
		vs = np.sqrt(np.sum(S*S,axis=1))
		
		indA = np.where(vs > 1e-6)		
		indA = indA[0]
						
		nactive = len(indA)
															
		if nactive > 1:
		
			temp = S[indA,:]
			Rp = np.dot(temp,temp.T)
			mRp = lng.norm(Rp,2)
			Rp = Rp + epsilon*mRp*np.identity(nactive)
			PinvS = np.dot(temp.T,lng.inv(Rp))
			A[:,indA] = np.dot(X,PinvS)
			
			for ns in indA:
				A[:,ns] = 	A[:,ns]/float(lng.norm(A[:,ns] + 1e-6))
				if lng.norm(A[:,ns]) < 1e-6:
					A[:,ns] = np.random.randn(m)
													
	return A,S,PinvA
	
################ END OF GMCA

	
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)	
	
def Eval_BSS(A0,S0,A,S):

	import numpy as np
	import scipy.linalg as lng 
		
	Diff = np.dot(lng.inv(np.dot(A.T,A)),np.dot(A.T,A0))
	
	z = np.shape(A)
	
	for ns in range(0,z[1]):
		Diff[ns,:] = abs(Diff[ns,:])/max(abs(Diff[ns,:]))
		
	Q = np.ones(z)
	Sq = np.ones(np.shape(S))
	
	for ns in range(0,z[1]):
		Q[:,np.nanargmax(Diff[ns,:])] = A[:,ns]
		Sq[np.nanargmax(Diff[ns,:]),:] = S[ns,:]
			
	Diff = np.dot(lng.inv(np.dot(Q.T,Q)),np.dot(Q.T,A0))
	
	for ns in range(0,z[1]):
		Diff[ns,:] = abs(Diff[ns,:])/max(abs(Diff[ns,:]))
		
	p = (np.sum(Diff) - z[1])/(z[1]*(z[1]-1))
	
	return p
	
################ END OF EVALUATION CRITERION

################# Data cube to matrix conversion

def Cube2Mat(X,sdim=0):
		
	Y = cp.copy(X)
	z = np.shape(Y)
	
	if sdim == 0:
		M = np.zeros((z[0],z[1]*z[2]))
		for r in range(0,z[0]):
			M[r,:] = np.reshape(Y[r,:,:],(1,z[1]*z[2]))
		
	if sdim == 1:
		M = np.zeros((z[1],z[0]*z[2]))
		for r in range(0,z[1]):
			M[r,:] = np.reshape(Y[:,r,:],(1,z[0]*z[2]))
	
	if sdim == 2:
		M = np.zeros((z[2],z[0]*z[1]))
		for r in range(0,z[2]):
			M[r,:] = np.reshape(Y[:,:,r],(1,z[0]*z[1]))
	
	return M

################ Cube2Mat

################# Matrix to data cube conversion

def Mat2Cube(M,nx,ny,sdim=0):
		
	Y = cp.copy(M)
	z = np.shape(Y)
	
	if sdim == 0:
		X = np.zeros((z[0],nx,ny))
		for r in range(0,z[0]):
			X[r,:,:] = np.reshape(Y[r,:],(nx,ny))
		
	if sdim == 1:
		X = np.zeros((nx,z[0],ny))
		for r in range(0,z[0]):
			X[:,r,:] = np.reshape(Y[r,:],(nx,ny))
	
	if sdim == 2:
		X = np.zeros((nx,ny,z[0]))
		for r in range(0,z[0]):
			X[:,:,r] = np.reshape(Y[r,:],(nx,ny))
	
	return X
