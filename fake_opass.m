function [gam,ngam,muu,Lam,nu,kappa,Phi]=fake_opass(y,params);
% Runs the FAKE_OPASS algorithm
% y is KxN, where K is the low-dimensional space and N is the total number
% of spikes in the data.
% params passes in a list of parameters:
% params.alph is the parameter of the CRP
% params.kappa_0, prior precision of mean on NW distribution
% params.nu_0, prior precision of Wishart part of NW distribution
% params.Phi_0, prior cluster covariance*nu_0
%% Rename params:
alph=params.alph;
kappa_0=params.kappa_0;
nu_0=params.nu_0;
Phi_0=params.Phi_0;
%% Internal parameters:
Cmax=200; % upper bound on number of clusters
%%
[K,N]=size(y);
ngam=zeros(Cmax,1);
kappa=repmat(kappa_0,Cmax,1);
nu=repmat(nu_0,Cmax,1);
Phi=cell(Cmax,1);
Lam=cell(Cmax,1);
for c=1:Cmax
    Phi{c}=Phi_0;
    Lam{c}=inv(Phi{c})*nu_0;
end
muu0=zeros(K,1);
gam=zeros(N,1);
muu=zeros(K,Cmax);
ngam=zeros(Cmax,1);

%% Run online DP
gam(1)=1;
muu(:,1)=(y(:,1)+kappa_0*muu(:,1))./(1+kappa_0);
Phi{1}=Phi{1}+(kappa(1)/(kappa(1)+1))*y(:,1)*y(:,1)';
kappa(1)=kappa(1)+1;
nu(1)=nu(1)+1;
Lam{1}=inv(Phi{1})*nu(1);
C=2;
ngam(1)=1;
% tic
for t=2:N
    lthet=log(ngam);lthet(C)=log(alph);
    termllk=zeros(1,C);
    termnorm=zeros(1,C);
    termwish=zeros(1,C);
    for c=1:C
        ya=y(:,t)-muu(:,c);
       termllk(c)=sum(diag(log(chol(Lam{c}))))-.5*(y(:,t)-muu(:,c))'*...
           Lam{c}*(y(:,t)-muu(:,c));
       termnorm(c)=K/2*log(kappa(c)/(1+kappa(c)))+.5/(kappa(c)+1)'*...
           (y(:,t)-muu(:,c))'*Lam{c}*(y(:,t)-muu(:,c));
       termwish(c)=K/2*log(2)-(nu(c)+1)/2*log(1+kappa(c)/(kappa(c)+1)*ya'*Lam{c}*ya/nu(c))-...
           -sum(diag(log(chol(Phi{c}))))+.5*kappa_0*ya'*Lam{c}*ya/(kappa_0+1);
    end
    lprob=lthet(1:C)'+termllk+termnorm+termwish;
    [~,c]=max(lprob);
    gam(t)=c;
    if c==C
        C=C+1;
    end
    Phi{c}=Phi{c}+kappa(c)*(y(:,t)-muu(:,c))*(y(:,t)-muu(:,c))'/(kappa(c)+1);
    muu(:,c)=(kappa(c)*muu(:,c)+y(:,t))/(kappa(c)+1);
    kappa(c)=kappa(c)+1;
    nu(c)=nu(c)+1;
    Lam{c}=inv(Phi{c})*nu(c);
    ngam(c)=ngam(c)+1;
end
% toc