function run_official_mss_replay(mss_root, out_dir)
% Deterministic, non-interactive execution of the official MSS OTTER plant
% and the PID-heading branch of SIMotter at pinned source commit. This file
% does not alter the official functions.
addpath(genpath(mss_root));
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% Five derivative witnesses spanning rest, motion, attitude, current and
% asymmetric propeller actuation.
X=zeros(12,5); N=zeros(2,5); MP=[0 25 45 10 30]; VC=[0 .3 .6 .15 .45]; BC=[0 pi/6 -pi/4 pi/2 -pi/3];
X(:,2)=[.4;-.1;0;0;0;.05;1;2;0;0;0;.2]; N(:,2)=[50;45];
X(:,3)=[1;.2;-.05;.01;-.02;-.08;-2;3;.1;.03;-.04;-.5]; N(:,3)=[-30;70];
X(:,4)=[.2;.4;.03;-.02;.01;.12;4;-1;-.2;-.08;.06;1.1]; N(:,4)=[100;-90];
X(:,5)=[1.4;-.3;.1;.05;-.03;.2;8;5;.3;.1;-.12;2.2]; N(:,5)=[120;120];
R=zeros(5,14);
for j=1:5
 [xd,U]=otter(X(:,j),N(:,j),MP(j),[.05;0;-.35],VC(j),BC(j));
 R(j,:)=[j,U,xd'];
end
dlmwrite(fullfile(out_dir,'MSS_OTTER_DERIVATIVE_WITNESSES.csv'),R,',','precision',17);

% Official SIMotter PID-heading branch without plotting or menu input.
randn('seed',6201); h=.05; tf=600; steps=round(tf/h)+1; mp=25;rp=[.05;0;-.35];Vc=.3;bc=pi/6;
[~,~,M,Bp,nmin,nmax]=otter(); Binv=invQR(Bp); T=1; K=T/M(6,6);wn=1.5;zeta=1;
Kp=M(6,6)*wn^2;Kd=M(6,6)*(2*zeta*wn-1/T);Td=Kd/Kp;Ti=10/wn;
wnd=1;zetad=1;rmax=10*pi/180;x=zeros(12,1);n=zeros(2,1);zpsi=0;psid=0;rd=0;ad=0;Tn=.1;
trace=zeros(steps,17);
for k=1:steps
 t=(k-1)*h;r=x(6)+.001*randn;psi=x(12)+.001*randn;psiref=0;
 if t>100, psiref=0; end
 if t>500, psiref=-pi/2; end
 [psid,rd,ad]=refModel(psid,rd,ad,psiref,rmax,zetad,wnd,h,1);
 tauX=100;tauN=(T/K)*ad+(1/K)*rd-Kp*(ssa(psi-psid)+Td*(r-rd)+(1/Ti)*zpsi);
 u=Binv*[tauX;tauN];nc=sign(u).*sqrt(abs(u));nc=satlim(nc,nmin,nmax);
 trace(k,:)=[t,x',n',psid,rd];
 x=rk4(@otter,h,x,n,mp,rp,Vc,bc);n=n+h/Tn*(nc-n);n=satlim(n,nmin,nmax);zpsi=zpsi+h*ssa(psi-psid);
end
dlmwrite(fullfile(out_dir,'MSS_OTTER_PID_REPLAY.csv'),trace,',','precision',17);
fid=fopen(fullfile(out_dir,'MSS_RUNTIME.txt'),'w');fprintf(fid,'octave=%s\n',version);fclose(fid);
end
