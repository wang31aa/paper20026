#!/usr/bin/env python3
"""Build a supplementary simulation-process movie with an evidence-safe label."""
from pathlib import Path
import math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'visualization'/'Nature_Simulation_Workflow.mp4'
W,H,FPS=1280,720,8
BG=(8,12,18); WHITE=(240,244,248); MUTED=(155,169,184)
BLUE=(48,137,206); ORANGE=(230,126,34); GREEN=(57,181,74); RED=(217,70,70); CYAN=(55,190,190)

def font(size,bold=False):
 paths=['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc']
 for p in paths:
  try:return ImageFont.truetype(p,size)
  except:pass
 return ImageFont.load_default()

F18,F22,F28,F38,F54=[font(x) for x in (18,22,28,38,54)]; B22,B28,B42=[font(x,True) for x in (22,28,42)]

def centered(d,text,y,f,color=WHITE):
 box=d.textbbox((0,0),text,font=f);d.text(((W-(box[2]-box[0]))/2,y),text,font=f,fill=color)
def fade(t,a,b): return max(0,min(1,(t-a)/(b-a)))
def panel(d,box,title,color):
 d.rounded_rectangle(box,18,fill=(18,25,34),outline=(51,65,80),width=2)
 d.text((box[0]+18,box[1]+14),title,font=B22,fill=color)
def node(d,x,y,c,r=10,outline=WHITE): d.ellipse((x-r,y-r,x+r,y+r),fill=c,outline=outline,width=2)
def line(d,a,b,c,width=2): d.line((a[0],a[1],b[0],b[1]),fill=c,width=width)

def frame_title(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'Selective coupling in heterogeneous collectives',220,F54)
 centered(d,'Movie S1  ·  Computational experiment workflow',300,F38,MUTED)
 centered(d,'Public-data-constrained models  |  Paired interventions  |  Task margins',390,F22,CYAN)
 d.text((30,H-42),'Simulation evidence — not a hardware experiment',font=F18,fill=MUTED)
 return im

def frame_pipeline(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'From public records to an executable, falsifiable test',45,B42)
 labels=[('1  Source records','trajectories\ncommands · metadata'),('2  Model qualification','one-step checks\nfree-run checks'),
         ('3  Paired policies','same initial state\nsame forcing'),('4  Task endpoints','margin · failure\nrecovery · costs')]
 xs=[70,365,660,955]
 for i,((a,b),x) in enumerate(zip(labels,xs)):
  c=[BLUE,CYAN,ORANGE,GREEN][i];d.rounded_rectangle((x,210,x+250,430),22,fill=(18,25,34),outline=c,width=3)
  d.text((x+18,245),a,font=B22,fill=c);d.multiline_text((x+18,305),b,font=F22,fill=WHITE,spacing=12)
  if i<3:
   d.line((x+255,320,x+285,320),fill=MUTED,width=4);d.polygon([(x+285,320),(x+272,312),(x+272,328)],fill=MUTED)
 d.text((70,500),'Promotion is fail-closed: missing controller execution, physical mapping or sealed outcomes cannot be inferred.',font=F22,fill=MUTED)
 d.text((30,H-42),'Evidence boundary is part of the experiment design',font=F18,fill=MUTED)
 return im

def scene_state(t):
 # t in [0,1]
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'Three independent domain adapters',25,B42)
 boxes=[(25,100,410,625),(447,100,832,625),(869,100,1254,625)]
 titles=[('Robot formation',BLUE),('Vehicle platoon',ORANGE),('Aerial swarm',GREEN)]
 for b,(title,c) in zip(boxes,titles):panel(d,b,title,c)
 # robot, five agents and one mismatched node
 cx,cy=215,370
 pts=[]
 for i in range(5):
  a=2*math.pi*i/5-.5*math.pi;rr=115
  x=cx+rr*math.cos(a)+12*math.sin(6*t+i);y=cy+rr*math.sin(a)+8*math.cos(5*t+i);pts.append((x,y))
 for i in range(5): line(d,pts[i],pts[(i+1)%5],MUTED,2)
 for i,p in enumerate(pts):node(d,*p,RED if i==4 and t>.45 else BLUE,12)
 d.text((55,565),'Formation integrity',font=F22,fill=WHITE)
 # vehicle
 road_y=390;d.line((475,road_y+28,805,road_y+28),fill=(80,89,98),width=4)
 for i,c in enumerate([ORANGE,CYAN,ORANGE]):
  x=745-i*(92+24*math.sin(2*t+i));d.rounded_rectangle((x-35,road_y-18,x+35,road_y+18),6,fill=c)
  if i<2:d.line((x-38,road_y,x-75,road_y),fill=MUTED,width=2)
 d.text((482,565),'Dynamic stopping-distance margin',font=F22,fill=WHITE)
 # uav
 ps=[]
 for i in range(5):
  a=2*math.pi*i/5+2*t;ps.append((1060+105*math.cos(a),370+78*math.sin(a)))
 for i in range(5):line(d,ps[i],ps[(i+1)%5],(70,95,85),2)
 for p in ps:node(d,*p,GREEN,11)
 d.ellipse((1035,325,1085,415),outline=RED,width=4);d.text((900,565),'Archive-defined separation and speed',font=F22,fill=WHITE)
 d.text((30,H-42),'Robot and vehicle: executable computational interventions  ·  UAV: source-controller qualification until replay passes',font=F18,fill=MUTED)
 return im

def scene_gate(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'Residual-triggered selection changes the next control update',25,B42)
 panel(d,(35,100,610,625),'All coupled',BLUE);panel(d,(670,100,1245,625),'Layered selection',GREEN)
 def swarm(x0,gated):
  ps=[]
  for i in range(5):
   a=2*math.pi*i/5;rr=125
   drift=(115*t if i==4 else 0);ps.append((x0+rr*math.cos(a)+drift,350+rr*.65*math.sin(a)))
  for i in range(5):
   c=(70,95,110)
   if gated and (i==4 or (i+1)%5==4):c=(90,55,55)
   line(d,ps[i],ps[(i+1)%5],c,2 if c==(70,95,110) else 1)
  for i,p in enumerate(ps):node(d,*p,RED if i==4 else (GREEN if gated else BLUE),13)
  return ps
 left=swarm(300,False);right=swarm(935,True)
 if t>.35:
  d.text((left[4][0]-55,left[4][1]-55),'high residual',font=F18,fill=RED)
  d.text((right[4][0]-70,right[4][1]-55),'weight ↓',font=F18,fill=RED)
 d.text((90,525),'Mismatch propagates through every edge',font=F22,fill=MUTED)
 d.text((725,500),'1  information qualification',font=F22,fill=CYAN)
 d.text((725,535),'2  target reachability',font=F22,fill=ORANGE)
 d.text((725,570),'3  physical feasibility',font=F22,fill=GREEN)
 return im

def scene_branch(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'Matched counterfactual branches',25,B42)
 d.text((70,105),'Same initial state, reference and disturbance innovations',font=F28,fill=WHITE)
 names=[('All coupled',BLUE),('Residual gate',CYAN),('Connectivity gate',ORANGE),('Physical filter',GREEN),('Constrained baseline',WHITE)]
 for i,(name,c) in enumerate(names):
  y=190+i*82;d.rounded_rectangle((70,y,315,y+52),12,fill=(18,25,34),outline=c,width=2);d.text((88,y+13),name,font=F22,fill=c)
  # margin trace, deliberately includes mixed/adverse effects
  xs=np.linspace(350,1190,100);base=.32-.55*np.exp(-((np.linspace(0,1,100)-.55)/.13)**2)
  offsets=[0,-.02,-.04,.14,.28];arr=base+offsets[i]+.04*np.sin(np.linspace(0,7,100)+i)
  pts=[(float(x),float(y+27-55*a)) for x,a in zip(xs,arr)];d.line(pts,fill=c,width=3)
  d.line((350,y+27,1190,y+27),fill=(95,75,75),width=1)
 d.text((350,630),'signed task margin:  above zero = feasible; below zero = declared failure side',font=F18,fill=MUTED)
 return im

def scene_metrics(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'A recovery claim must pass every endpoint',35,B42)
 labels=[('Failure prediction','sensitivity · specificity · lead time'),('Recovery','20-update dwell · no recurrence'),
         ('Task performance','minimum and tail margin'),('Resources','energy · saturation · communication')]
 for i,(a,b) in enumerate(labels):
  y=150+i*115;c=[RED,GREEN,BLUE,ORANGE][i]
  d.rounded_rectangle((125,y,1155,y+82),16,fill=(18,25,34),outline=c,width=2)
  d.text((155,y+14),a,font=B28,fill=c);d.text((500,y+22),b,font=F22,fill=WHITE)
  progress=min(1,max(0,t*1.5-i*.18));d.rectangle((905,y+56,1125,y+65),fill=(48,58,68));d.rectangle((905,y+56,905+220*progress,y+65),fill=c)
 d.text((125,635),'A longer survival time is not promoted when tail error or control cost becomes unacceptable.',font=F22,fill=MUTED)
 return im

def scene_end(t):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 centered(d,'Scientific test, not a demonstration of guaranteed safety',155,B42)
 centered(d,'Mechanism  →  critical function / viability set  →  intervention  →  task outcome',250,F28,CYAN)
 centered(d,'Positive, null and adverse policy effects are retained.',340,F28,WHITE)
 centered(d,'Hardware recovery requires the proposed gate to be executed on hardware or HIL.',400,F22,MUTED)
 centered(d,'Public-data-constrained computational experiments',520,F28,ORANGE)
 return im

def main():
 OUT.parent.mkdir(exist_ok=True)
 writer=imageio.get_writer(OUT,fps=FPS,codec='libx264',quality=8,macro_block_size=None)
 scenes=[(4,frame_title),(6,frame_pipeline),(7,scene_state),(7,scene_gate),(6,scene_branch),(6,scene_metrics),(4,scene_end)]
 for seconds,fn in scenes:
  for k in range(seconds*FPS): writer.append_data(np.asarray(fn(k/max(seconds*FPS-1,1))))
 writer.close();print(OUT)

if __name__=='__main__':main()
