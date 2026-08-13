from __future__ import annotations
from dataclasses import dataclass
import sys, time
from pathlib import Path
import numpy as np

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))
import casadi as ca

from .model import SwarmParameters


@dataclass
class CasadiResult:
    controls: np.ndarray
    states: np.ndarray
    success: bool
    iterations: int
    solve_seconds: float
    objective: float
    minimum_constraint_margin: float


class CasadiNMPC:
    """Sparse symbolic single-shooting translation of the archived NMPC."""
    def __init__(self, p: SwarmParameters, backend: str = "sqpmethod"):
        self.p = p; self._warm = None
        n, horizon, dt = p.n, p.horizon_steps, p.dt
        nx, nu = 6*n, 3*n
        x0 = ca.MX.sym("x0", nx)
        controls = ca.MX.sym("u", nu, horizon)
        x = x0; states = [x]; objective = 0; constraints = []
        limit = p.component_acceleration_limit
        lbx = [-limit] * (nu*horizon); ubx = [limit] * (nu*horizon)
        weights_stage = ([1/p.max_neighbours]*(n*p.max_neighbours)+[5]*n+[5]*n+[.4]*(3*n))
        weights_terminal = ([1]*(n*p.max_neighbours)+[5]*n+[5]*n)

        def outputs(state, control=None):
            pos = state[:3*n]; vel = state[3*n:]
            sep = []
            for agent in range(n):
                dist2 = []
                for other in range(n):
                    if other == agent: dist2.append(ca.MX(1e10))
                    else:
                        delta = pos[3*other:3*other+3]-pos[3*agent:3*agent+3]
                        dist2.append(ca.dot(delta,delta))
                available = [ca.MX(1) for _ in range(n)]
                for _rank in range(p.max_neighbours):
                    chosen = ca.MX(-1); minimum = ca.MX(1e9)
                    for candidate in range(n):
                        cond = ca.logic_and(dist2[candidate] < minimum,
                               ca.logic_and(dist2[candidate] < p.communication_radius**2,
                                            available[candidate] > .5))
                        chosen = ca.if_else(cond, candidate, chosen)
                        minimum = ca.if_else(cond, dist2[candidate], minimum)
                    value = ca.MX(p.reference_distance**2)
                    for candidate in range(n):
                        value = ca.if_else(chosen == candidate, dist2[candidate], value)
                        available[candidate] = ca.if_else(chosen == candidate, 0, available[candidate])
                    sep.append((value-p.reference_distance**2)/p.max_neighbours)
            direction=[]; navigation=[]
            ref=ca.DM(p.reference_direction)
            for agent in range(n):
                v=vel[3*agent:3*agent+3]; speed2=ca.dot(v,v)
                direction.append(ca.if_else(speed2>1e-14,1-ca.dot(v,ref)**2/speed2,1))
                navigation.append(speed2-p.reference_speed**2)
            values=sep+direction+navigation
            if control is not None: values += [0.4*control[i] for i in range(nu)]
            return ca.vertcat(*values)

        for k in range(horizon):
            u=controls[:,k]; y=outputs(x,u)
            objective += dt*ca.dot(ca.DM(weights_stage)*y,y)
            pos=x[:3*n]
            for i in range(n-1):
                for j in range(i+1,n):
                    d=pos[3*j:3*j+3]-pos[3*i:3*i+3]
                    constraints.append(ca.dot(d,d)-4*p.collision_radius**2)
            for i in range(n):
                for cx,cy,radius in np.asarray(p.cylinders).reshape(-1,3):
                    d=pos[3*i:3*i+2]-ca.DM([cx,cy])
                    constraints.append(ca.dot(d,d)-radius**2-p.safety_margin**2)
            pos_next=pos+dt*x[3*n:]+.5*dt*dt*u
            vel_next=x[3*n:]+dt*u
            x=ca.vertcat(pos_next,vel_next); states.append(x)
        terminal=outputs(x); objective += ca.dot(ca.DM(weights_terminal)*terminal,terminal)
        decision=ca.reshape(controls,-1,1); g=ca.vertcat(*constraints)
        nlp={"x":decision,"p":x0,"f":objective,"g":g}
        if backend == "ipopt":
            options={"ipopt.print_level":0,"print_time":False,"ipopt.max_iter":100,
                     "ipopt.tol":1e-6,"ipopt.acceptable_tol":1e-4}
        elif backend == "sqpmethod":
            # Plugin-independent fallback for hosts on which the distributed
            # IPOPT binary cannot be loaded. The OCP itself is unchanged.
            options={"print_time":False,"qpsol":"qrqp","max_iter":100,
                     "print_header":False,"print_iteration":False,
                     "print_status":False,
                     "qpsol_options":{"print_header":False,"print_iter":False}}
        else:
            raise ValueError(f"unsupported backend: {backend}")
        self.backend = backend
        self.solver=ca.nlpsol("official_nmpc_translation",backend,nlp,options)
        self.lbx=np.asarray(lbx); self.ubx=np.asarray(ubx)
        self.lbg=np.zeros(int(g.numel())); self.ubg=np.full(int(g.numel()),np.inf)
        self.state_fun=ca.Function("rollout",[x0,decision],[ca.hcat(states)])

    def solve(self,x0):
        initial=np.zeros_like(self.lbx) if self._warm is None else self._warm.reshape(-1,order="F")
        start=time.perf_counter()
        solution=self.solver(x0=initial,p=np.asarray(x0),lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
        elapsed=time.perf_counter()-start
        controls=np.asarray(solution["x"]).reshape(3*self.p.n,self.p.horizon_steps,order="F").T
        states=np.asarray(self.state_fun(np.asarray(x0),solution["x"])).T
        self._warm=np.vstack((controls[1:],controls[-1:]))
        stats=self.solver.stats()
        margins=np.asarray(solution["g"],dtype=float).reshape(-1)
        return CasadiResult(controls,states,bool(stats.get("success",False)),int(stats.get("iter_count",-1)),elapsed,float(solution["f"]),float(np.min(margins)))
