function r2lmvalue = r2lm( measuredy, estimatedy )
% r^2 based on the latest measured y-values
%
% Calculate r^2 based on the latest measured y-values
% measuredy and estimatedy must be vectors.
%
% --- input ---
% measuredy : measured y-values ( m x 1, m is the number of samples)
% estimated : estimated y-values ( m x 1 )
%
% --- output ---
% r2lmvalue : r2 based on the latest measured y-values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r2lmvalue = 1 - sum( (measuredy-estimatedy).^2 ) ./ sum( (measuredy(1:end-1,:)-measuredy(2:end,:)).^2 );

end

