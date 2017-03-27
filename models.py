################# DEFINES SoftTHRD	

def SoftTHRD(x,kmad=3,J=2):
    
    import numpy as np
    from copy import deepcopy as dp
    
    xout = dp(x)

    c,w = Starlet_Forward(x=xout,J=J)
    
    thrd = kmad*mad(w[:,:,0])
    
    w = (w - thrd*np.sign(w))*(abs(w) > thrd)

    xout = Starlet_Inverse(c=c,w=w) 
    
    return xout
    
################# DEFINES HardTHRD	

def HardTHRD(x,kmad=3,J=2):
    
    from copy import deepcopy as dp
    
    xout = dp(x)

    c,w = Starlet_Forward(x=xout,J=J)
    
    thrd = kmad*mad(w[:,:,0])
    
    w = w*(abs(w) > thrd)

    xout = Starlet_Inverse(c=c,w=w) 
    
    return xout

################# DEFINES MRDENOISE

def MRDenoise(b = 0,kmad = 3,J=2,nmax=2):
    
    import numpy as np
    from copy import deepcopy as dp
    
    # Estimating the MR mask

    c0,w = Starlet_Forward(x=b,J=J,boption=3)

    mask = 0*dp(w)
    
    for r in range(0,J):
        
        temp = w[:,:,r]
        temp = np.sign(abs(temp) - kmad*mad(temp))
        mask[:,:,r] =  temp*(temp > 0)
    
    xout = 0.*dp(b)
        
    for r in range(0,nmax):
        
        q = b - xout
        cq,wq = Starlet_Forward(x=q,J=J,boption=3)
        wq = wq*mask
        xout= xout + Starlet_Inverse(c=cq,w=wq)        
    
    return xout,mask

#
#
#   Performs sparse deconvolution using the forward-backward algorithm
#
#   Input : b - N x M array (input image)
#           h - N x M array (psf in the pixel domain)
#           nmax - scalar (number of iterations)
#           J - scalar (number of starlet scales)
#           k_mad - scalar (value of the k-mad threshold)
#
#
#   Output : x - N x M array (estimated deconvolved image)
#
#

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	

def mad(xin = 0):

	import numpy as np

	z = np.median(abs(xin - np.median(xin)))/0.6735
	
	return z

##### Starlet transform

def length(x=0):

    import numpy as np
    l = np.max(np.shape(x))
    return l

################# 1D convolution	

def filter_1d(xin=0,h=0,boption=3):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp    
    
    x = np.squeeze(cp.copy(xin));
    n = length(x);
    m = length(h);
    y = cp.copy(x);

    z = np.zeros(1,m);

    for r in range(np.int(np.floor(m/2))):
                
        if boption == 1: # --- zero padding
                        
            z = np.concatenate([np.zeros(m-r-np.floor(m/2)-1),x[0:r+np.floor(m/2)+1]],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[n-(m-(r+np.floor(m/2)))+1:n],x[0:r+np.floor(m/2)+1]],axis=0)
        
        if boption == 3: # --- mirror
            
            u = x[0:m-(r+np.floor(m/2))-1];
            u = u[::-1]
            z = np.concatenate([u,x[0:r+np.floor(m/2)+1]],axis=0)
                                     
        y[r] = np.sum(z*h)
        
        
    

    a = np.arange(np.int(np.floor(m/2)),np.int(n-m+np.floor(m/2)),1)

    for r in a:
        
        y[r] = np.sum(h*x[r-np.floor(m/2):m+r-np.floor(m/2)])
    

    a = np.arange(np.int(n-m+np.floor(m/2)+1),n,1)

    for r in a:
            
        if boption == 1: # --- zero padding
            
            z = np.concatenate([x[r-np.floor(m/2):n],np.zeros(m - (n-r) - np.floor(m/2))],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[r-np.floor(m/2):n],x[0:m - (n-r) - np.floor(m/2)]],axis=0)
        
        if boption == 3: # --- mirror
                        
            u = x[n - (m - (n-r) - np.floor(m/2) -1)-1:n]
            u = u[::-1]
            z = np.concatenate([x[r-np.floor(m/2):n],u],axis=0)
                    
        y[r] = np.sum(z*h)
    	
    return y
 
################# 1D convolution with the "a trous" algorithm	

def Apply_H1(x=0,h=0,scale=1,boption=3):

	import numpy as np
	import copy as cp
	
	m = length(h)
	
	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h
	
	else:
		g = h
				
	y = filter_1d(x,g,boption)
	
	return y
	

################# 2D "a trous" algorithm

def Starlet_Forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import numpy as np
	import copy as cp
	
	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]))
	w = np.zeros((nx[0],nx[1],J))

	c = cp.copy(x)
	cnew = cp.copy(x)
	
	for scale in range(J):
		
		for r in range(nx[0]):
			
			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)
			
		for r in range(nx[1]):
		
			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)
			
		w[:,:,scale] = c - cnew;
		
		c = cp.copy(cnew);

	return c,w
	
	################# 2D "a trous" algorithm

def Starlet_Inverse(c=0,w=0):

	import numpy as np
	
	x = c+np.sum(w,axis=2)

	return x

