#!/usr/bin/env python3
import json,math,multiprocessing as mp,socket,struct,sys
DOMAINS={"uav6dof":(.82,1.23,.10,.27,"second"),"vehicle":(.34,.88,.015,.046,"second"),"motor":(.026,.071,.018,.057,"second"),"robot":(.72,1.31,.11,.29,"second"),"microgrid":(2.4,5.7,.55,1.35,"second"),"circuit":(.78,1.36,.42,.93,"first"),"water":(.76,1.42,.035,.092,"first"),"structure":(.76,1.38,.12,.36,"second")}
POLICIES=("all_coupled","two_layer_gate","physical_filter"); N=5; K=60; DT=.02
def lin(a,b): return [a+(b-a)*i/(N-1) for i in range(N)]
def params(d):
 a,b,c,e,_=DOMAINS[d]; return lin(a,b),lin(c,e),lin(.65,1.05)
def step(d,k,x,v,u):
 p1,p2,_=params(d); bias=[.018*math.sin(.071*k+i)+.002*math.sin(.13*k+list(DOMAINS).index(d)) for i in range(N)]
 if DOMAINS[d][4]=="second":
  acc=[(u[i]-p2[i]*v[i]-bias[i])/p1[i] for i in range(N)]
  v=[v[i]+DT*acc[i] for i in range(N)]; x=[x[i]+DT*v[i] for i in range(N)]
 else:
  flow=[p1[i]*u[i]-p2[i]*x[i]+bias[i] for i in range(N)]
  x=[x[i]+DT*flow[i] for i in range(N)]; v=flow
 return x,v
def control(policy,x,v):
 trusted=[abs(z)<=.25 for z in x]; u=[]
 for i in range(N):
  nei=(x[i-1]-x[i]) if i and (policy=="all_coupled" or trusted[i-1] or i-1==0) else 0.
  z=-1.05*x[i]-.42*v[i]+.34*nei
  if policy=="two_layer_gate": z-=.18*x[i]*trusted[i]
  if policy=="physical_filter": z-=.32*math.tanh(x[i]/.25)
  u.append(max(-1.,min(1.,z)))
 return u
def send(s,o):
 b=json.dumps(o,separators=(",",":")).encode(); s.sendall(struct.pack("!I",len(b))+b)
def recv(s):
 h=b""
 while len(h)<4: h+=s.recv(4-len(h))
 n=struct.unpack("!I",h)[0]; b=b""
 while len(b)<n: b+=s.recv(n-len(b))
 return json.loads(b)
def plant_pipe(c,d):
 x=[.04*(i+1) for i in range(N)]; v=[0.]*N
 for k in range(K):
  c.send((x,v)); u=c.recv(); x,v=step(d,k,x,v,u); c.send((x,v))
def ctrl_pipe(c,p,q):
 for _ in range(K):
  x,v=c.recv(); c.send(control(p,x,v)); x,v=c.recv()
 q.put((x,v))
def plant_tcp(portq,d):
 s=socket.socket(); s.bind(("127.0.0.1",0)); s.listen(1); portq.put(s.getsockname()[1]); c,_=s.accept(); x=[.04*(i+1) for i in range(N)]; v=[0.]*N
 for k in range(K):
  send(c,{"x":x,"v":v}); u=recv(c)["u"]; x,v=step(d,k,x,v,u); send(c,{"x":x,"v":v})
 c.close(); s.close()
def ctrl_tcp(port,p,q):
 s=socket.socket(); s.connect(("127.0.0.1",port))
 for _ in range(K):
  m=recv(s); send(s,{"u":control(p,m["x"],m["v"])}); m=recv(s)
 q.put((m["x"],m["v"])); s.close()
def episode(mode,d,p):
 q=mp.Queue()
 if mode=="pipe":
  a,b=mp.Pipe(); ps=mp.Process(target=plant_pipe,args=(a,d)); pc=mp.Process(target=ctrl_pipe,args=(b,p,q))
 else:
  pq=mp.Queue(); ps=mp.Process(target=plant_tcp,args=(pq,d)); ps.start(); port=pq.get(); pc=mp.Process(target=ctrl_tcp,args=(port,p,q)); pc.start(); out=q.get(); ps.join(); pc.join(); return out
 ps.start(); pc.start(); out=q.get(); ps.join(); pc.join(); return out
def main():
 mode=sys.argv[1]; assert mode in ("pipe","tcp"); out={"backend":mode,"hardware_hil":False,"entity_hil":False,"physical_experiment":False,"domains":{}}
 for d in DOMAINS:
  p1,p2,_=params(d); out["domains"][d]={"parameter_spread":min(max(p1)-min(p1),max(p2)-min(p2)),"policies":{}}
  for p in POLICIES: out["domains"][d]["policies"][p]=episode(mode,d,p)[0]
 assert all(v["parameter_spread"]>0 for v in out["domains"].values())
 open("virtual_"+mode+".json","w").write(json.dumps(out,sort_keys=True,indent=2)+"\n")
 print("PASS",mode,len(out["domains"]))
if __name__=="__main__": mp.set_start_method("spawn"); main()
