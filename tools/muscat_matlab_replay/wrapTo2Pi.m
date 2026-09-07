function wrapped = wrapTo2Pi(angle)
%WRAPTO2PI Toolbox-independent equivalent used by the fixed source replay.
wrapped = mod(angle, 2*pi);
wrapped(angle > 0 & wrapped == 0) = 2*pi;
end
