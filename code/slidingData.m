function Sldiingdata=slidingData(data, slidpara)

dim = size(data);
if slidpara >= dim(2)
    fprintf("don't need to slid\n");
    Sldiingdata = data;
end

slidetimes = dim(2)-slidpara+1;

for i = 1:slidetimes
    Sldiingdata(:,:,i) = data(:,i:i+slidpara-1);
end
end