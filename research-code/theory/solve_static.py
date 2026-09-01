#!/usr/bin/env python3
"""Auditable static-branch certificate for the heterogeneous Chua example.

This script does not fit a pre-existing certificate.  It solves (i) a diagonal
stability SDP for the pinned follower Laplacian and (ii) a common diagonal
Lyapunov/IQC SDP for the seven heterogeneous follower matrices.  It then
integrates the isolated target to estimate a clearly labelled finite-horizon
tail disturbance envelope and reports the corresponding theorem bound.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
_BUNDLED = Path("/private/tmp/cvxdeps")
if _BUNDLED.exists():
    sys.path.insert(0, str(_BUNDLED))
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent

def chua_AB(i: int):
    A=np.array([[-(9-.1*(i-1))*2/7, 9-.1*(i-1), 0],
                [1,-1,1],[0,-(14.286-.1*(i-1)),.0005+.0001*i]],float)
    B=np.diag([(9-.1*(i-1))*3/7,0,0])
    return A,B

def phi(x):
    return np.array([.5*(abs(x[0]+1)-abs(x[0]-1)),0,0])

def laplacian():
    Lt=np.array([[1,0,0,0,0,0,0,-1],[-2,2,0,0,0,0,0,0],
      [0,-2,2,0,0,0,0,0],[-2,0,-1,3,0,0,0,0],
      [0,0,-3,0,3,0,0,0],[0,0,0,0,0,1,0,-1],
      [0,0,0,0,0,0,1,-1],[0,0,0,0,0,0,0,0]],float)
    return Lt[:7,:7]

def solve_G(L):
    g=cp.Variable(7); margin=cp.Variable()
    G=cp.diag(g); S=G@L+L.T@G
    # Convex normalization: maximize an ordinary Euclidean margin, then report
    # the exact generalized margin relative to G a posteriori.
    prob=cp.Problem(cp.Maximize(margin),[cp.sum(g)==7,g>=1e-4,S-margin*np.eye(7)>>0])
    prob.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_feas=1e-10)
    gv=np.asarray(g.value); Sv=np.diag(gv)@L+L.T@np.diag(gv)
    mu=np.linalg.eigvalsh(np.diag(gv**-.5)@Sv@np.diag(gv**-.5)).min()
    return gv,mu,prob.status

def solve_Q(alpha=16.1, H=np.diag([12.,10.,11.]), L_phi=1.0):
    # Q diagonal is required so QH=H'Q. trace(Q)=3 fixes scale.
    q=cp.Variable(3); r=cp.Variable(); tau=cp.Variable(7,nonneg=True)
    Q=cp.diag(q); cons=[cp.sum(q)==3,q>=0.05,q<=20]
    for k in range(7):
        A,B=chua_AB(k+1)
        top=Q@A+A.T@Q-r*np.eye(3)+tau[k]*L_phi**2*np.eye(3)
        M=cp.bmat([[top,Q@B],[B.T@Q,-tau[k]*np.eye(3)]])
        cons += [M << -1e-8*np.eye(6),tau[k]>=1e-7]
    prob=cp.Problem(cp.Minimize(r),cons)
    prob.solve(solver="CLARABEL",tol_gap_abs=1e-9,tol_feas=1e-9)
    return np.diag(np.asarray(q.value)),float(r.value),np.asarray(tau.value),prob.status

def target_tail_omega(Q,g,T=200.,tail=50.):
    A0,B0=chua_AB(8)
    sol=solve_ivp(lambda t,x:A0@x+B0@phi(x),(0,T),[.1,-.05,.08],
                  rtol=2e-10,atol=1e-12,max_step=.005,dense_output=False)
    keep=sol.t>=T-tail; vals=[]
    for x in sol.y[:,keep].T:
        ws=[]
        for i in range(1,8):
            A,B=chua_AB(i)
            w=(A-A0)@x+(B-B0)@phi(x)
            ws.append(w)
        vals.append(np.sqrt(sum(g[i]*(ws[i]@Q@ws[i]) for i in range(7))))
    return (float(np.max(vals)),float(np.quantile(vals,.999)),len(vals),
            np.max(np.abs(sol.y[:,keep]),axis=1).tolist())

def main():
    L=laplacian(); g,mu,gstat=solve_G(L)
    H=np.diag([12.,10.,11.]); Q,r,tau,qstat=solve_Q(H=H)
    M=Q@H
    coupling=16.1*mu*np.linalg.eigvalsh(M).min()
    d=coupling-r
    a=d/np.linalg.eigvalsh(Q).max()
    omega,q999,n,tail_state_absmax=target_tail_omega(Q,g)
    bound=2*omega/(a*np.sqrt(g.min()*np.linalg.eigvalsh(Q).min())) if a>0 else np.inf
    lmi_max=[]
    for k in range(7):
        A,B=chua_AB(k+1); tk=tau[k]
        top=Q@A+A.T@Q-r*np.eye(3)+tk*np.eye(3)
        Z=np.block([[top,Q@B],[B.T@Q,-tk*np.eye(3)]])
        lmi_max.append(float(np.linalg.eigvalsh(Z).max()))
    out=dict(G=g.tolist(),mu_G=mu,G_status=gstat,Q=Q.tolist(),H=H.tolist(),
      r=r,tau=tau.tolist(),Q_status=qstat,alpha=16.1,L_phi=1.0,coupling_margin=coupling,
      d=d,a=a,tail_weighted_omega=omega,tail_q999=q999,tail_samples=n,
      tail_state_absmax=tail_state_absmax,
      theorem_bound_using_finite_tail_envelope=bound,
      diagonal_stability_residual_min_eig=float(np.linalg.eigvalsh(np.diag(g)@L+L.T@np.diag(g)-mu*np.diag(g)).min()),
      iqc_lmi_max_eigs=lmi_max,
      eig_L=[[float(z.real),float(z.imag)] for z in np.linalg.eigvals(L)])
    (HERE/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
