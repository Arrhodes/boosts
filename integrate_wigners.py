"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import treams.special as sp

jay1=1
m1=1
lam1=1

jay2=1
m2=1
lam2=1

num=1000

# Integration over ETA
eta_list = np.linspace(-1,1,num)
wigs1eta = sp.wignersmalld(jay1,m1,lam1,np.arccos(eta_list))
wigs2eta = sp.wignersmalld(jay2,m2,lam2,np.arccos(eta_list))

measure_eta = np.diff(eta_list, append=eta_list[-1])
integral_eta = np.einsum(
    'e,e,e->',
    measure_eta,
    wigs1eta,
    wigs2eta
    ) * np.sqrt(2*jay1+1) * np.sqrt(2*jay2+1) / 2



# Integration over THETA
theta_list = np.linspace(0, np.pi, num)
wigs1theta = sp.wignersmalld(jay1,m1,lam1,theta_list)
wigs2theta = sp.wignersmalld(jay2,m2,lam2,theta_list)

measure_theta = np.diff(theta_list, append=theta_list[-1])*np.sin(theta_list)
integral_theta = np.einsum(
    'e,e,e->',
    measure_theta,
    wigs1theta,
    wigs2theta
    ) * np.sqrt(2*jay1+1) * np.sqrt(2*jay2+1) / 2

print(integral_eta)
print(integral_theta)