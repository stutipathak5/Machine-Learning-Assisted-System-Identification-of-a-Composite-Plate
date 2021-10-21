"SMT FUNCTION FOR LHS"

from smt.sampling_methods import LHS
import numpy as np 
exec(open("finite_element_analysis.py").read())

Lx=0.5 
Ly=0.5 
h=0.005
rho=1846
no_sam=300000
lim=np.array([[50*10**9, 150*10**9], [5*10**9, 10*10**9],[0.25,0.45],[2*10**9,8*10**9]])
sam=LHS(xlimits=lim)
output=sam(no_sam)
inputTD=np.zeros((no_sam,6001))
inputFD=np.zeros((no_sam,27))

for i in range(no_sam):
  print(i)                                        
  E1=output[i][0] 
  E2=output[i][1]
  nu12=output[i][2]
  G12=output[i][3]
  iFD=FEM(E1,E2,nu12,G12,rho,Lx,Ly,h,'FD')
  iTD=FEM(E1,E2,nu12,G12,rho,Lx,Ly,h,'TD')
  inputTD[i,:]=iTD
  inputFD[i,:]=iFD

np.save('/content/drive/MyDrive/FEM/LHS(300000)_inputTD.npy', inputTD) 
np.save('/content/drive/MyDrive/FEM/LHS(300000)_inputFD.npy', inputFD) 
np.save('/content/drive/MyDrive/FEM/LHS(300000)_output.npy', output) 

"ALTERNATIVE FUNCTION FOR LHS"

def LHS(mini,maxi,no_sam):
  
  no_var = np.shape(mini)[0]
  r = np.random.rand(no_sam,no_var)
  s = np.zeros((no_sam,no_var))
  for i in np.arange(no_var):
    f = np.random.permutation(no_sam)
    p = (f-r[:,i])/no_sam
    s[:,i]=mini[i]+(p*(maxi[i]-mini[i]))

  return s 
