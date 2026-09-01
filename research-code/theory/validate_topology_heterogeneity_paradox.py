#!/usr/bin/env python3
"""Numerically check all sign cases in the network-derived theorem."""
import math
import numpy as np

def r(rho,lam,ell,D,U): return max(rho*D-U,0)/(lam+rho*ell)
def p(rho,lam,ell,D,U,C): return r(rho,lam,ell,D,U)+C/rho
def deriv(rho,lam,ell,D,U,C):
 return (D*lam+U*ell)/(lam+rho*ell)**2-C/rho**2

# Interior optimum case, active on the tested interval.
lam,ell,D,U,C=1.4,.25,2.2,.15,.18
A=D*lam+U*ell
star=lam*math.sqrt(C)/(math.sqrt(A)-ell*math.sqrt(C))
assert star>U/D and deriv(star,lam,ell,D,U,C)==0 or abs(deriv(star,lam,ell,D,U,C))<1e-12
assert deriv(.8*star,lam,ell,D,U,C)<0<deriv(1.2*star,lam,ell,D,U,C)
grid=np.linspace(max(U/D+.001,.2),3*star,5000)
numeric=grid[np.argmin([p(x,lam,ell,D,U,C) for x in grid])]
assert abs(numeric-star)<.002

# Monotone case: contraction dominates exposed mismatch.
lam,ell,D,U,C=1.,2.,.2,0.,.3
assert math.sqrt(D*lam)<=ell*math.sqrt(C)
assert all(deriv(x,lam,ell,D,U,C)<=1e-12 for x in np.linspace(.1,10,500))

# Topology directional derivative agrees with finite differences.
rho,lam,D,ell,U=1.1,1.3,1.4,.7,.2
Dp,ellp=.18,.05
analytic=(rho*(lam+rho*ell)*Dp-rho*(rho*D-U)*ellp)/(lam+rho*ell)**2
eps=1e-6
numeric=(r(rho,lam,ell+eps*ellp,D+eps*Dp,U)-r(rho,lam,ell,D,U))/eps
assert abs(analytic-numeric)<1e-6
print({'interior_optimum':star,'monotone_case_pass':True,'topology_derivative_error':abs(analytic-numeric)})
