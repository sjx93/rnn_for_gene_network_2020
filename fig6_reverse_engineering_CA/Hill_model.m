function f = Hill_model(x,b,K,links,Nx)
%x, current input and state [BATCH,Nx,Gin+Gs]
%b, max_rate for Hill activation term [Gin+Gs,Gs]
%K, Micheales constant [Gin+Gs, Gs]
%links, network topology +1/0/-1, [Gin+Gs, Gs]
BATCH = size(x,1);
Gs = 10;
Gin= Gs;

Hill_n = 2;
f0 = (repmat(reshape(x,[BATCH,Nx,Gin+Gs,1]),[1,1,1,Gs]).^Hill_n)./...
     (repmat(reshape(K,[1,1,Gin+Gs,Gs]),[BATCH,Nx,1,1]).^Hill_n +...
      repmat(reshape(x,[BATCH,Nx,Gin+Gs,1]),[1,1,1,Gs]).^Hill_n);
     %[BATCH,Nx,Gin+Gs,Gs]
f_activation1 = f0.*repmat(reshape(b,[1,1,Gin+Gs,Gs]),[BATCH,Nx,1,1]).*...
                    repmat(reshape(links==1,[1,1,Gin+Gs,Gs]),[BATCH,Nx,1,1]);
f_activation2 = sum(f_activation1,3); %[BATCH,Nx,Gs]

f_inhibition1 = (1-f0).*repmat(reshape(links==-1,[1,1,Gin+Gs,Gs]),[BATCH,Nx,1,1]) +...
                       repmat(reshape(links~=-1,[1,1,Gin+Gs,Gs]),[BATCH,Nx,1,1]);
f_inhibition2 = prod(f_inhibition1,3); %[BATCH,Nx,1,Gs]

f = squeeze(f_activation2.*f_inhibition2); %[BATCH,Nx,Gs]
end