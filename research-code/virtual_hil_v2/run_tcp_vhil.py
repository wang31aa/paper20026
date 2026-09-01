#!/usr/bin/env python3
"""Independent TCP transport backend for the frozen eight-domain plants."""
from pathlib import Path
import csv,json,multiprocessing as mp,socket,struct,time,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'virtual_hil_v1'))
import run_vhil as base
C=json.loads((HERE/'VHIL_V2_FROZEN_CONTRACT.json').read_text()); OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def send(s,obj):
    b=json.dumps(obj,separators=(',',':')).encode(); s.sendall(struct.pack('!I',len(b))+b)
def recv(s):
    h=b''
    while len(h)<4: h+=s.recv(4-len(h))
    n=struct.unpack('!I',h)[0]; b=b''
    while len(b)<n: b+=s.recv(n-len(b))
    return json.loads(b)

def server(portq,domain,pars,x0,v0,noise):
    srv=socket.socket(); srv.bind(('127.0.0.1',0)); srv.listen(1); portq.put(srv.getsockname()[1]); c,_=srv.accept()
    dt=C['sample_period_s']; spec=base.SPECS[domain]; x=x0.copy(); v=v0.copy()
    for k in range(C['cycles']):
        send(c,{'cycle':k,'x':x.tolist(),'v':v.tolist()}); u=np.asarray(recv(c)['applied'])
        bias=.018*np.sin(.071*k+np.arange(C['nodes']))+noise[k]
        if spec['kind']=='second':
            a=(u-pars['p2']*v-bias)/pars['p1']; v+=dt*a; x+=dt*v
        else:
            flow=pars['p1']*u-pars['p2']*x+bias; x+=dt*flow; v=flow
        send(c,{'cycle':k,'x':x.tolist(),'v':v.tolist()})
    c.close(); srv.close()

def client(port,domain,policy,pars,outq):
    s=socket.socket(); s.connect(('127.0.0.1',port)); spec=base.SPECS[domain]; rows=[]; n=C['nodes']
    for k in range(C['cycles']):
        t=time.perf_counter(); m=recv(s); x=np.asarray(m['x']); v=np.asarray(m['v']); residual=x-spec['target']; trusted=np.abs(residual)<=spec['margin']
        A=np.zeros((n,n),int)
        for i in range(1,n):
            if policy=='all_coupled' or trusted[i-1] or i-1==0: A[i,i-1]=1
        nei=np.array([sum(A[i,j]*(x[j]-x[i]) for j in range(n)) for i in range(n)])
        req=-1.05*residual-.42*v+.34*nei
        if policy=='two_layer_gate': req-=.18*residual*trusted
        if policy=='physical_filter': req-=.32*np.tanh(residual/max(spec['margin'],1e-9))
        app=np.clip(req,-pars['limit'],pars['limit']); send(s,{'applied':app.tolist()}); nxt=recv(s)
        margin=spec['margin']-float(np.max(np.abs(np.asarray(nxt['x'])-spec['target'])))
        rows.append(dict(domain=domain,policy=policy,cycle=k,state_json=json.dumps(nxt['x']),requested_control_json=json.dumps(req.tolist()),applied_control_json=json.dumps(app.tolist()),adjacency_json=json.dumps(A.tolist()),task_margin=margin,transport_latency_s=time.perf_counter()-t))
    outq.put(rows); s.close()

def episode(domain,policy,q):
    pars=base.parameters(domain); noise=base.innovations(domain); rng=np.random.default_rng(base.C['seeds'][domain]+7)
    x0=base.SPECS[domain]['target']+rng.normal(0,.16,C['nodes']); v0=rng.normal(0,.04,C['nodes']); pq=mp.Queue()
    p=mp.Process(target=server,args=(pq,domain,pars,x0,v0,noise)); p.start(); port=pq.get()
    c=mp.Process(target=client,args=(port,domain,policy,pars,q)); c.start(); p.join(); c.join()
    if p.exitcode or c.exitcode: raise RuntimeError((domain,policy,p.exitcode,c.exitcode))

def main():
    q=mp.Queue(); rows=[]
    for d in C['domains']:
      for p in C['policies']:
        j=mp.Process(target=episode,args=(d,p,q)); j.start(); rows.extend(q.get()); j.join()
        if j.exitcode: raise RuntimeError((d,p))
    with (OUT/'tcp_cycles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)
    summary={'qualification':'independent_tcp_virtual_platform','hardware_hil':False,'entity_platform':False,'rows':len(rows),'domains':{}}
    for d in C['domains']:
        summary['domains'][d]={'minimum_parameter_spread':min(float(np.ptp(v)) for v in base.parameters(d).values()),'final_margins':{p:[r for r in rows if r['domain']==d and r['policy']==p][-1]['task_margin'] for p in C['policies']}}
    summary['claim_boundary']=C['claim_boundary']; (OUT/'qualification_registry.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': mp.set_start_method('spawn'); main()
