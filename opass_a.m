function [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=opass_a(x,A,params);
% Runs the OPASS algorithm
% x is of length N
% A is size PxK
% of spikes in the data.
% params passes in a list of parameters:
% params.alph is the parameter of the CRP
% params.kappa_0, prior precision of mean on NW distribution
% params.nu_0, prior precision of Wishart part of NW distribution
% params.Phi_0, prior cluster covariance*nu_0
% params.a_pii and params.b_pii are the hyperparameters on the probability
% of seeing a spike
% params.bet is the time scale of the ar function
%%
apii=params.a_pii;
bpii=params.b_pii;
alph=params.alph;
Phi0=params.Phi_0;
nu_0=params.nu_0;
kappa_0=params.kappa_0;
bet=params.bet;
%% Internal Parameters
Cmax=50;
curndx=0;
lookahead=500;
rang=15;
tau=10;
%%
N=numel(x);
[P,K]=size(A);
%% Calculate precision matrix
[acf] = autocorr(x,1);
if abs(acf(2))<1e-3
    acf(2)=0;
end
% acf(2)=0;
lambi=zeros(P);
for p=1:P
    lambi(p,:)=1-p:P-p;
end
sig=acf(2).^abs(lambi)*cov(x(1:1e5));
sig(1:(P+1):P^2)=cov(x(1:1e5));
lamda=inv(sig);
detlamb=det(lamda);
%%
pii=apii./bpii;
nu=repmat(nu_0,Cmax,1);
Phi=cell(Cmax,1);
lamclus=cell(Cmax,1);
R=cell(Cmax,1);
for c=1:Cmax
    Phi{c}=Phi0;
    lamclus{c}=inv(Phi0)*nu_0;
    R{c}=lamclus{c}*.2;
end
muu00=zeros(K,1);
muu0=zeros(K,Cmax);
kappa=kappa_0*ones(Cmax,1);
muu=zeros(K,Cmax);
tlastspike=zeros(Cmax,1);
itau=1./tau;
muuS=cell(Cmax,1);
lamclusS=cell(Cmax,1);
%%
xpad=[x;zeros(P,1)];
%%
C=0;
nz=0;
z=zeros(N,1);
gam=zeros(N,1);
piip=zeros(N,1);
lthet=zeros(Cmax,1);
ngam=zeros(Cmax,1);
S=zeros(K,N);
lpii=log(pii);
lnpii=log(1-pii);
thr=log(pii./(1-pii));
xm=x;
mT=N;
logdetlamb=log(detlamb);
%%
while curndx<N-P-rang
    %% set up parameters
    ndx=(curndx+1:min(mT-P-rang,curndx+lookahead));n=numel(ndx);
    ndxwind=bsxfun(@plus,ndx,[0:P-1]');
    xwind=xpad(ndxwind);
    pii=(apii+sum(z))./(bpii+curndx);
    thr=log(pii./(1-pii));
    %% set up parameters
    %     ndx=(curndx+1:curndx+lookahead);n=numel(ndx);
    ndx=(curndx+1:min(mT-P-rang,curndx+lookahead));n=numel(ndx);
    ngam2=ngam;
    nz2=nz;
    lthet=log(ngam2./(alph+nz2));lthet(C+1,1)=log(alph./(alph+nz2));
    
    %% calc llk
    lnone=-P/2*log(2*pi)+.5*logdetlamb-.5*dot(xwind,((lamda)*xwind));
    lon=zeros(C+1,n);
    for c=1:C
        %         lon(c,:)=getllk(xwind,muu(:,c),A,lamclus,sig,kappa(c));
        Qt=inv(lamclus{c})+inv(R{c});
        Q=sig+A*(Qt*A');
        xwindm=bsxfun(@minus,xwind,A*muu(:,c));
        tmp=sum(log(diag(chol(Q))));
        Re=(ndx-tlastspike(c))<50;
        lon(c,:)=-P/2*log(2*pi)-tmp-.5*dot(xwindm,(Q\xwindm))-double(Re)*1e5;
    end
    Qt=inv(lamclus{C+1})+inv(R{C+1});
    Q=sig+A*(Qt*A');
    lon(C+1,:)=-P/2*log(2*pi)-sum(log(diag(chol(Q))))-.5*dot(xwind,(Q\xwind));
    %     lon(C+1,:)=getllk(xwind,muu0,A,lamclus,sig,kappa_0);
    lon=bsxfun(@plus,lthet(1:C+1,:),lon);
    H=bsxfun(@minus,lon,max(lon));
    Hadj=log(sum(exp(H)));
    lthr=lnone-max(lon)-Hadj; % Fix this.
    %% Find new spike
    Q=find(lthr<thr,1,'first');
    % no spike
    if (numel(Q)==0) || Q>lookahead-rang
        curndx=curndx+lookahead-rang;
        continue
    end
    % new spike
    [~,offset]=min(lthr(Q:Q+rang));
    Q=Q+offset-1;
    nz=nz+1;
    Qt=Q+curndx;
    z(Qt)=1;
    [~,Cnew]=max(lon(:,Q));
    if Cnew>C
        C=Cnew;
    end
    Qmat=A'*lamda*A+lamclus{Cnew};
    yhat=Qmat\(A'*lamda*xwind(:,Q)+lamclus{Cnew}*muu(:,Cnew));
    gam(Qt)=Cnew;
    ngam(Cnew)=ngam(Cnew)+1;
    deltt=Qt-tlastspike(Cnew);
    tlastspike(Cnew)=Qt;
    ebet=exp(-bet*deltt);
    mhat=muu0(:,Cnew)*(1-ebet)+muu(:,Cnew)*ebet;
    muu0(:,Cnew)=(kappa(Cnew).*muu0(:,Cnew)+yhat)./(kappa(Cnew)+1);
    Qhat=itau*eye(K)*(1-ebet^2)+inv(R{Cnew})*ebet.^2;
    R{Cnew}=inv(Qhat)+lamclus{Cnew};
    muu(:,Cnew)=R{Cnew}\(Qhat\mhat+lamclus{Cnew}*yhat);
    yhat=Qmat\(A'*lamda*xwind(:,Q)+lamclus{Cnew}*muu(:,Cnew));
    Phi{Cnew}=Phi{Cnew}+kappa(Cnew)./(kappa(Cnew)+1)*(yhat-muu(:,Cnew))*(yhat-muu(:,Cnew))'+inv(Qmat);
    kappa(Cnew)=kappa(Cnew)+1;
    nu(Cnew)=nu(Cnew)+1;
    lamclus{Cnew}=inv(Phi{Cnew})*nu(Cnew);
    S(:,Qt)=yhat;
    curndx=Qt+4;
    xpad(Qt:Qt+P-1)=xpad(Qt:Qt+P-1)-A*S(:,Qt);
end


















