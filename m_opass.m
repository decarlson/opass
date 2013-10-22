function [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=m_opass(x,A,params);
% Runs the M_OPASS algorithm
% x is size NxnumCh
% A is size PxK
% of spikes in the data.
% params passes in a list of parameters:
% params.alph is the parameter of the CRP
% params.kappa_0, prior precision of mean on NW distribution
% params.nu_0, prior precision of Wishart part of NW distribution
% params.Phi_0, prior cluster covariance*nu_0
% params.a_pii and params.b_pii are the hyperparameters on the probability
% of seeing a spike
%%
apii=params.a_pii;
bpii=params.b_pii;
alph=params.alph;
Phi0=params.Phi_0;
nu_0=params.nu_0;
kappa_0=params.kappa_0;
%% Internal Parameters
Cmax=50;
curndx=0;
lookahead=500;
rang=40;
%%
[N,numCh]=size(x);
[P,K]=size(A);
%% Calculate precision matrix
[acf] = autocorr(x(:,1),1);
if abs(acf(2))<1e-3
    acf(2)=0;
end
% acf(2)=0;
lambi=zeros(P);
for p=1:P
    lambi(p,:)=1-p:P-p;
end
sig=acf(2).^abs(lambi)*cov(x(1:1e5,1));
sig(1:(P+1):P^2)=cov(x(1:1e5,1));
lamda=inv(sig);
detlamb=det(lamda);
%%
pii=apii./bpii;
lamb=inv(sig);
detlamb=det(lamb);
% lamclus=eye(K)*1e1;
nu=repmat(nu_0,Cmax,1);
Phi=cell(Cmax,numCh);
lamclus=cell(Cmax,numCh);
for ch=1:numCh
    for c=1:Cmax
        Phi{c}{ch}=Phi0;
        lamclus{c}{ch}=inv(Phi0)*nu_0;
    end
end
kappa_0=.1;
muu0=zeros(K,numCh);
kappa=kappa_0*ones(Cmax,1);
muu=zeros(K,numCh,Cmax);
%%
xpad=cell(numCh,1);
for ch=1:numCh
    xpad{ch}=[x(:,ch);zeros(P,1)];
end
xwind=cell(numCh,1);
ywind=cell(numCh,1);
%%

C=0;
nz=0;
z=zeros(N,1);
gam=zeros(N,1);
piip=zeros(N,1);
lthet=zeros(Cmax,1);
ngam=zeros(Cmax,1);
S=zeros(K,numCh,N);
lpii=log(pii);
lnpii=log(1-pii);
thr=log(pii./(1-pii));
xm=x;
tlastspike=zeros(Cmax,1);
muuS=cell(Cmax,numCh);
lamclusS=cell(Cmax,numCh);
mT=N;
sz=0;
%%
tic
while curndx<N-P-rang
    %% set up parameters
%     pii=(apii+sz)./(bpii+curndx);
%     thr=log(pii./(1-pii));
    ndx=(curndx+1:min(mT-P-rang,curndx+lookahead));n=numel(ndx);
    ndxwind=bsxfun(@plus,ndx,[0:P-1]');
%     lthet=log(ngam./(alph+nz));lthet(C+1)=log(alph./(alph+nz));
%     xwind=xpad(ndxwind);
    lthet=log(ngam./(alph+nz));lthet(C+1)=log(alph./(alph+nz));
    for ch=1:numCh
        xwind{ch}=xpad{ch}(ndxwind);
    end
    %% calc llk
    lnone=zeros(1,n,numCh);
    lon=zeros(C+1,n,numCh);
    for ch=1:numCh;
        lnone(1,:,ch)=-P/2*log(2*pi)+.5*log(detlamb)-.5*sum((xwind{ch}.*((lamb)*xwind{ch})));
        
        for c=1:C
            %         lon(c,:)=getllk(xwind,muu(:,c),A,lamclus,sig,kappa(c));
            Q=sig+(1+kappa(c))./kappa(c)*A*(lamclus{c}{ch}\A');
            xwindm=bsxfun(@minus,xwind{ch},A*muu(:,ch,c));
            lon(c,:,ch)=-P/2*log(2*pi)-sum(log(diag(chol(Q))))-.5*sum(xwindm.*(Q\xwindm));
        end
        Q=sig+(1+kappa(C+1))./kappa(C+1)*A*(lamclus{C+1}{ch}\A');
        lon(C+1,:,ch)=-P/2*log(2*pi)-sum(log(diag(chol(Q))))-.5*sum(xwind{ch}.*(Q\xwind{ch}));
    end
    lon=sum(lon,3);
    lon(:,:)=bsxfun(@plus,lthet(1:C+1,:),lon);
    %     lon(C+1,:)=getllk(xwind,muu0,A,lamclus,sig,kappa_0);
    lnone=sum(lnone,3);
    H=bsxfun(@minus,lon,max(lon));
    Hadj=log(sum(exp(H)));
    lthr=lnone-max(lon)-Hadj;
    %% Find new spike
    Q=find(lthr<thr,1,'first');
    % no spike
    if (numel(Q)==0) || Q>lookahead-rang
        curndx=curndx+lookahead-rang;
        continue
    end
    % new spike
    [~,offset]=min(lthr(Q:min(Q+rang,numel(lthr))));
    Q=Q+offset-1;
    nz=nz+1;
    Qt=Q+curndx;
    z(Qt)=1;
    [~,Cnew]=max(lon(:,Q));
    if Cnew>C
        C=Cnew;
    end
    tlastspike(Cnew)=Qt;
%     Qmat=A'*lamda*A+lamclus{Cnew};
        %     yhat=A'*xwind(:,Q);
    ngam(Cnew)=ngam(Cnew)+1;
    gam(Qt)=Cnew;
    nu(Cnew)=nu(Cnew)+1;
     for ch=1:numCh
             curndx=Qt+1;
        Qmat=A'*lamda*A+lamclus{Cnew}{ch};
        yhat=Qmat\(A'*lamda*xwind{ch}(:,Q)+lamclus{Cnew}{ch}*muu(:,ch,Cnew));
        %     yhat=A'*xwind(:,Q);

        muu(:,ch,Cnew)=(muu(:,ch,Cnew)*kappa(Cnew)+yhat)./(kappa(Cnew)+1);
%         muuS{Cnew}=[muuS{Cnew},muu(:,Cnew)];
        Phi{Cnew}{ch}=Phi{Cnew}{ch}+kappa(Cnew)./(kappa(Cnew)+1)*(yhat-muu(:,ch,Cnew))*(yhat-muu(:,ch,Cnew))'+inv(Qmat);
        lamclus{Cnew}{ch}=inv(Phi{Cnew}{ch})*nu(Cnew);
        %     S(:,Qt)=Bci*(A'*lamb*xwind(:,Q)+lamclus*muu(:,gam(Qt)));
        S(:,ch,Qt)=yhat;
%         curndx=Qt+1;
        xpad{ch}(Qt:Qt+P-1)=xpad{ch}(Qt:Qt+P-1)-A*S(:,ch,Qt);
    end
%     yhat=Qmat\(A'*lamda*xwind(:,Q)+lamclus{Cnew}*muu(:,Cnew));

%     muuold=muu(:,Cnew);
%     muu(:,Cnew)=(muu(:,Cnew)*kappa(Cnew)+yhat)./(kappa(Cnew)+1);dmuu=muuold-muu(:,Cnew);
%     S(:,Qt)=yhat;
    %     Phi{Cnew}=Phi0+bsxfun(@minus,S(:,gam==Cnew),muu(:,Cnew))*bsxfun(@minus,S(:,gam==Cnew),muu(:,Cnew))'+ngam(Cnew)*inv(Qmat);
%     Phi{Cnew}=Phi{Cnew}+(kappa(Cnew)+1)./(kappa(Cnew)+1)*(yhat-muu(:,Cnew))*(yhat-muu(:,Cnew))'+inv(Qmat)+ngam(Cnew)*dmuu*dmuu';
    kappa(Cnew)=kappa(Cnew)+1;

%     lamclus{Cnew}=inv(Phi{Cnew})*nu(Cnew);
    %     S(:,Qt)=Bci*(A'*lamb*xwind(:,Q)+lamclus*muu(:,gam(Qt)));
    

%     muuS{Cnew}=[muuS{Cnew},muu(:,Cnew)];
%     lamclusS{Cnew}{size(muuS{Cnew},2)}=lamclus{Cnew};
%     xpad(Qt:Qt+P-1)=xpad(Qt:Qt+P-1)-A*S(:,Qt);
    sz=sz+1;
    %     continue
end
toc

















