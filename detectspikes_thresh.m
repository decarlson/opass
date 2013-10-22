function [timepoints,spikes]=detectspikes_thresh(x,thres,samplingrate,P,maxpoint)

timepoints=[];
t=2*P;
N=numel(x);
while t<N-P
    wind=x(t-P:t+P);
    [val,ndx]=max(wind);
    if val>thres
        if ndx<P*3/2;
            timepoints=[timepoints,t-P-1+ndx];
            t=t+round(4e-3*samplingrate);
            
        end
    end
    t=t+P;
end
% get array of detected spikes
spikes=zeros(P,numel(timepoints));
for t=1:numel(timepoints)
    spikes(:,t)=x(timepoints(t)+(-maxpoint+1:P-maxpoint));
end