function run_v63(mss_root,out_dir)
addpath(genpath(mss_root));if ~exist(out_dir,'dir'),mkdir(out_dir);end
h=.05;steps=301;N=5;seeds=[6301 6302];rhos=[.2 .6 1];policies={'all_coupled','independent_tracking','two_layer_gate'};
mp=[0 10 20 30 40];rp=[0 .03 .05 .07 .09;0 0 0 0 0;-.30 -.32 -.35 -.37 -.40];Vc=[.10 .16 .22 .28 .34];bc=[0 .25 -.35 .50 -.60];off=[0 -3 -3 -6 -6;0 -3 3 -3 3];
[~,~,~,Bp,nmin,nmax]=otter();Bi=invQR(Bp);runrows=zeros(18,10);tsrows=zeros(18*steps*N,28);rr=0;tt=0;
for sd=seeds
 for rho=rhos
  for pi=1:3
   rr=rr+1;randn('seed',sd);x=zeros(12,N);n=zeros(2,N);zp=zeros(2,N);zv=zeros(2,N);minc=1e9;energy=0;msgs=0;
   x(7:8,:)=off+.12*randn(2,N);x(12,:)=.03*randn(1,N);
   for k=1:steps
    t=(k-1)*h;tp=[.45*t;1.2*sin(.12*t)];tv=[.45;.144*cos(.12*t)];zp(:,1)=tp;zv(:,1)=tv;
    oldp=zp;oldv=zv;
    for i=2:N,zp(:,i)=zp(:,i)+h*(zv(:,i)+1.5*rho*(oldp(:,i-1)-zp(:,i)));zv(:,i)=zv(:,i)+h*1.5*rho*(oldv(:,i-1)-zv(:,i));msgs=msgs+1;end
    vin=zeros(2,N);nreq=zeros(2,N);for i=1:N,c=cos(x(12,i));s=sin(x(12,i));vin(:,i)=[c -s;s c]*x(1:2,i);end
    for i=1:N
     if pi==2,dp=tp;dv=tv;else,dp=zp(:,i);dv=zv(:,i);end
     vd=dv+.40*(dp+off(:,i)-x(7:8,i));
     if pi~=2
      for j=1:N
       if j==i,continue;end;dist=norm(x(7:8,i)-x(7:8,j));
       if dist<7
        w=rho;res=norm((x(7:8,j)-off(:,j))-(x(7:8,i)-off(:,i)));
        if pi==3 && res>2,w=.15*rho;end
        vd=vd+w*(.10*((x(7:8,j)-off(:,j)+off(:,i))-x(7:8,i))+.08*(vin(:,j)-vin(:,i)));
       end
      end
     end
     hd=atan2(vd(2),vd(1));spe=norm(vd);tauX=min(180,max(0,80+35*(spe-x(1,i))));tauN=-90*ssa(x(12,i)-hd)-28*x(6,i);u=Bi*[tauX;tauN];nc=sign(u).*sqrt(abs(u));nc=satlim(nc,nmin,nmax);nreq(:,i)=nc;
     x(:,i)=rk4(@otter,h,x(:,i),n(:,i),mp(i),rp(:,i),Vc(i),bc(i));n(:,i)=satlim(n(:,i)+h/.1*(nc-n(:,i)),nmin,nmax);energy=energy+h*sum(n(:,i).^2);
    end
    for i=1:N,for j=1:i-1,minc=min(minc,norm(x(7:8,i)-x(7:8,j)));end;end
    for i=1:N,tt=tt+1;tsrows(tt,:)=[rr,k,t,i,x(:,i)',n(1,i),n(2,i),nreq(1,i),nreq(2,i),zp(1,i),zp(2,i),zv(1,i),zv(2,i),norm(zp(:,i)-tp),rho,mp(i),Vc(i)];end
   end
   tp=[.45*15;1.2*sin(.12*15)];er=sqrt(mean(mean((x(7:8,:)-(tp+off)).^2)));ok=(minc>=1.5 && er<=3);runrows(rr,:)=[rr,sd,rho,pi,ok,er,minc,energy,msgs,max(abs(n(:)))];
  end
 end
end
dlmwrite(fullfile(out_dir,'V63_RUNS.csv'),runrows,',','precision',17);dlmwrite(fullfile(out_dir,'V63_NODE_TIMESERIES.csv'),tsrows,',','precision',17);
end
