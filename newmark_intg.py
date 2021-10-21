"FUNCTION FOR NEWMARK INTEGRATION"

def NEWMARK(MG,CG,KG,t,xi,vi,fi): 

  import numpy as np 

  dt = t[1]-t[0]                                      
  # F=fi[:,0]                                   
  # ai=np.dot(np.linalg.inv(MG),(F-np.dot(CG,vi)-np.dot(KG,xi)))
  ai = np.zeros((np.shape(xi)[0]))                      
  alpha = 0.25 
  delta = 0.5                  
  a0 = 1/(alpha*dt**2)
  a1 = delta/(alpha*dt)                  
  a2 = 1/(alpha*dt) 
  a3 = (1/(2*alpha))-1                    
  a4 = (delta/alpha)-1 
  a5 = (dt/2)*(delta/alpha-2)              
  a6 = dt*(1-delta) 
  a7 = delta*dt                         
  Kb = KG+a0*MG+a1*CG 
  x = np.zeros((np.shape(xi)[0],np.shape(t)[0]))
  v = np.zeros((np.shape(xi)[0],np.shape(t)[0]))
  a = np.zeros((np.shape(xi)[0],np.shape(t)[0])) 
  x[:,0] = xi
  v[:,0] = vi
  a[:,0] = ai  

  for i in np.arange(1,np.shape(t)[0],1):                                
    F = fi[:,i]                                          
    Fb = F+np.dot(MG,(a0*x[:,i-1]+a2*v[:,i-1]+a3*a[:,i-1]))+np.dot(CG,(a1*x[:,i-1]+a4*v[:,i-1]+a5*a[:,i-1]))
    x[:,i] = np.dot(np.linalg.inv(Kb),Fb)
    # v[:,i] = a1*(x[:,i]-x[:,i-1])-a4*v[:,i-1]-a5*a[:,i-1]
    a[:,i] = a0*(x[:,i]-x[:,i-1])-a2*v[:,i-1]-a3*a[:,i-1]
    v[:,i] = v[:,i-1]+a6*a[:,i-1]+a7*a[:,i]

  return [x,v,a]    
