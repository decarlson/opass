% makeToy:
load waveforms
[P,K]=size(waveforms);
%% Generate a background signal from a correlated noise model
% Note that there are better ways to do this:
ar1=.7;
numSeconds=60;
sampleRate=10000;
totalSamples=sampleRate*numSeconds;
xbackground=randn(totalSamples,1)*sqrt(1-ar1^2);
for t=2:totalSamples
    xbackground(t)=xbackground(t-1)*(ar1)+xbackground(t);
end
%%
x=xbackground;
scale=7;
for k=1:K
    waveforms(:,k)=waveforms(:,k)./(max(abs(waveforms(:,k)))./scale);
end
%% spike times:
sptimes=cell(K,1);
rate=5;
for k=1:K
    numspikes=poissrnd(numSeconds*rate);
    tmp=randi(totalSamples,numspikes,1);
    tmp=sort(tmp);
    dtmp=tmp(2:end)-tmp(1:end-1);
    for t=numspikes-1:-1:1
        if dtmp(t)<50;
            tmp(t)=[];
        end
    end
    tmp(tmp>totalSamples-P)=[];
    sptimes{k}=tmp;
end
%% Add to X:
% first pass, just the exact waveform, no variability:
for k=1:K
    for t=1:numel(sptimes{k})
        tim=sptimes{k}(t);
        q=tim:tim+P-1;
        x(q)=x(q)+waveforms(:,k);
    end
end
%%
X=x;
save toy X sampleRate sptimes