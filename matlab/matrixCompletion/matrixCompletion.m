 function [ output, U, V, log,objective ] = matrixCompletion( input, mask,maskT, rank,iters,mu,rows,cols,type)
[m,n]=size(input);
U=rand(m,rank)/sqrt(rank) ;
V=rand(rank,n)/sqrt(rank) ;
 
log=zeros(iters,1);
objective=zeros(iters,1);

nnzMask=length(rows);
 
    for it=1:iters
        
        
        if(type==1)
            simpleIterationSelection  
            simpleIteration
        elseif(type==2)
             cleverSelection  
             simpleIteration
        elseif(type==3)
             simpleIterationSelection
             kaczmarzIteration
        elseif(type==4)
            cleverSelection
            kaczmarzIteration
       elseif(type==5)
           
            if (mod(it,2)==0)
                cleverSelection 
               kaczmarzIteration     
            else
               simpleIterationSelection 
               simpleIteration 
            end
            
            
        end
        
       
      
     
      
        output=U*V;
      error = 0.5* norm((output-input).*mask,'fro')^2 ;
      log(it)=error;
      objective(it)=error+mu/2*(norm(U,'fro')^2+ norm(V,'fro')^2);
     
    end
 
end

