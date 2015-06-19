function [ mask ] = createMask( m,n,sparsity)
 
mask=(rand(m,n)<sparsity);

end

