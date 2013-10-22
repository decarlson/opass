%% test all algorithms
% set which algorithms to run
run_FAKEOPASS=true;
run_OPASS=false;
run_OPASS_A=false;
run_M_OPASS=false;
run_M_OPASS_A=false;
% add local data path
% addpath ../data/
% load example data set
% load d5331filt
load toy
ictimes=[];
% what part of the dataset to use
mT=size(X,1);
x=X(1:mT,1); % load channel 1
xa=X;
[N,numCh]=size(x);
samplingrate=1e4;
%% Set paramters
P=round(3e-3*samplingrate); % window size is 3ms
maxpoint=round(1.5e-3*samplingrate); % where to align waveform peaks
K=3; % Number of PCA components to use.
sig=std(x); % Noise standard deviation estimate
thres=3*sig; % Detection voltage threshold
%% Detect spike waveforms
[timepoints,spikes]=detectspikes_thresh(-x,thres,samplingrate,P,maxpoint);
%% Reduce dimensionality
maxtimepoints=30*samplingrate; % limit to first 30s to simulate online system
[U,S,V]=svd(spikes(:,timepoints<maxtimepoints),'econ');
A=U(:,1:K);
y=A'*spikes; % convert all spikes to low-dimensional feature space
% normalize
y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y')');
%% Get IC labels:
xtimes=timepoints;
Q=zeros(size(xtimes));
label=zeros(size(xtimes));
ictimesq=zeros(size(ictimes));
n1=0;
for q=1:numel(ictimes);
    if sum(abs(xtimes-ictimes(q)-1)<10);
        n1=n1+1;
        ictimesq(q)=1;
        label(abs(xtimes-ictimes(q)-1)<11)=1;
    end
end
%% Set parameters:
params.alph=1e-1;
params.kappa_0=.01;
params.nu_0=.1;
params.Phi_0=.1*eye(K);
params.a_pii=1;
params.b_pii=1e7;
params.bet=1./(30*samplingrate);
%% Run fake_opass
if run_FAKEOPASS
    %%
    [gam,ngam,muu,Lam,nu,kappa,Phi]=fake_opass(y,params);    
    %% Plot non-trivial clusters;
    C=sum(ngam>10);
    [~,rendx]=sort(ngam,'descend');
    colors=hsv(C);
    figure(1);clf; hold on
    set(0,'defaulttextinterpreter','latex')
    for c=1:C
        plot3(y(1,gam==rendx(c)),y(2,gam==rendx(c)),y(3,gam==rendx(c)),'.','MarkerSize',15,'color',colors(c,:))
    end
    plot3(y(1,label==1),y(2,label==1),y(3,label==1),'.k','MarkerSize',5) % add IC spikes
    xlabel('pc-1');ylabel('pc-2');zlabel('pc-3');title('\tt FAKE-OPASS','FontSize',20)
    hold off
end
%% run OPASS
if run_OPASS
%     clear params
    [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=opass(x,A,params);
    %% Plot non-trivial clusters
    % figure out which spike detections go with a IC spike
    C=sum(ngam>0);
    zic=zeros(size(z));
    for c=1:C
        xtimes=find((z>0)&(gam==c));
        nic(c)=0;
        for q=1:numel(xtimes);
            if sum(abs(xtimes(q)-ictimes+10)<5);
                nic(c)=nic(c)+1;
                zic(xtimes(q))=1;
            end
        end
        
    end
    % Plot spikes
    C=max(gam);
    col=hsv(C);
    figure(1);clf;hold on
    for c=1:C
        plot(S(1,gam==c),S(2,gam==c),'.','Color',col(c,:),'markersize',20)
    end
    plot(S(1,zic>0),S(2,zic>0),'k.','markersize',10);
    hold off
    xlabel('PCA Component 1','FontSize',16)
    ylabel('PCA Component 2','FontSize',16)
    title('Inferred y_k Values for Detected Spikes','FontSize',18)
    snames=cell(C+1,1);
    for c=1:C
        snames{c}=num2str(c);
    end
    snames{C+1}='IC';
    a=legend(snames);
end
%% run OPASS-A
if run_OPASS_A
%     clear params
    [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=opass_a(x,A,params);
    %% Plot non-trivial clusters
    % figure out which spike detections go with a IC spike
    zic=zeros(size(z));
    for c=1:C
        xtimes=find((z>0)&(gam==c));
        nic(c)=0;
        for q=1:numel(xtimes);
            if sum(abs(xtimes(q)-ictimes+10)<5);
                nic(c)=nic(c)+1;
                zic(xtimes(q))=1;
            end
        end
        
    end
    % Plot spikes
    C=max(gam);
    col=hsv(C);
    figure(1);clf;hold on
    for c=1:C
        plot(S(1,gam==c),S(2,gam==c),'.','Color',col(c,:),'markersize',20)
    end
    plot(S(1,zic>0),S(2,zic>0),'k.','markersize',10);
    hold off
    xlabel('PCA Component 1','FontSize',16)
    ylabel('PCA Component 2','FontSize',16)
    title('Inferred y_k Values for Detected Spikes','FontSize',18)
    snames=cell(C+1,1);
    for c=1:C
        snames{c}=num2str(c);
    end
    snames{C+1}='IC';
    a=legend(snames);
    
end
%%
%% run M_OPASS
if run_M_OPASS
%     clear params
    [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=m_opass(xa,A,params);
    %% Plot non-trivial clusters
    % Plot spikes
    C=max(gam);
    col=hsv(C);
    figure(1);clf;hold on
    for c=1:C
        plot(squeeze(S(1,1,gam==c)),squeeze(S(2,1,gam==c)),'.','Color',col(c,:),'markersize',20)
    end
%     plot(S(1,zic>0),S(2,zic>0),'k.','markersize',10);
    hold off
    xlabel('PCA Component 1','FontSize',16)
    ylabel('PCA Component 2','FontSize',16)
    title('Inferred y_k Values for Detected Spikes','FontSize',18)
    snames=cell(C+1,1);
    for c=1:C
        snames{c}=num2str(c);
    end
    snames{C+1}='IC';
    a=legend(snames);
end

%% run M_OPASS_A
if run_M_OPASS_A
%     clear params
    [z,gam,ngam,muu,lamclus,nu,kappa,Phi,S]=m_opass_a(xa,A,params);
    %% Plot non-trivial clusters
    % Plot spikes
    C=max(gam);
    col=hsv(C);
    figure(1);clf;hold on
    for c=1:C
        plot(squeeze(S(1,1,gam==c)),squeeze(S(2,1,gam==c)),'.','Color',col(c,:),'markersize',20)
    end
%     plot(S(1,zic>0),S(2,zic>0),'k.','markersize',10);
    hold off
    xlabel('PCA Component 1','FontSize',16)
    ylabel('PCA Component 2','FontSize',16)
    title('Inferred y_k Values for Detected Spikes','FontSize',18)
    snames=cell(C+1,1);
    for c=1:C
        snames{c}=num2str(c);
    end
    snames{C+1}='IC';
    a=legend(snames);
end


% % % % %%
% % % % if false
% % % %     nC=ngam;
% % % %     N=size(y,2);
% % % %     C=max(gam);
% % % %     for c=1:C
% % % %         nic(c)=sum((gam==c).*label(1:N)');
% % % %         fprintf('Total in class %d:%d %d %0.2f\n',c, ngam(c),nic(c),nic(c)./ngam(c))
% % % %     end
% % % %     [~,tndx]=max(nic);
% % % %     e1=(sum(nC)-nC(tndx)+2*nic(tndx)-sum(nic))./sum(nC);
% % % %     e2=nic(tndx)./nC(tndx);
% % % %     e3=nic(tndx)./sum(nic);
% % % %     % n1=sum(nic);
% % % %     n2=nic(tndx);
% % % %     [e1,e2,e3,n1,n2]
% % % % end
