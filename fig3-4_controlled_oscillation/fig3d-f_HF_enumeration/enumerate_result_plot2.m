A = cell(4,1);
px = cell(4,1);
node_size = cell(4,1);
wplot = [2,3,3,3];
for n = 1:4
    N_links = n+4;
    A0 = csvread([ num2str(N_links),'edges/allforscat.csv']);
    A{n} = A0(:,3:end);
    node_size{n} =  A0(:,1);
    %ntmp = A0(:,6)==1; A{n} = A0(ntmp,3:end);
    N1 = size(A{n},1);
    px{n} = wplot(n)*linspace(-1,1,N1)';
end

B = cell(3,1);
for n = 1:3
    N1 = size(A{n},1);
    N2 = size(A{n+1},1);
    B0 = zeros(N1,N2);
    for i = 1:N1
        for j = 1:N2
            B0(i,j) = sum(A{n}(i,:)~=A{n+1}(j,:));
        end
    end
    B{n} = B0;
end

loss0 = 9999;
for i = 1:50000
    px_new = px;
    n = randi(4);
    perm_i = randi(size(px{n},1),2,1);
    px_new{n} = px{n};
    px_new{n}(perm_i(1)) = px{n}(perm_i(2));
    px_new{n}(perm_i(2)) = px{n}(perm_i(1));
    loss = graph_tangle_loss(px_new,B);
    if (loss <= loss0) || rand()<0.0002
        px = px_new;
        loss0 = loss;
        disp(loss);
    end
end

% save 'HF_enumeration_pxforscat.mat' A B node_size px


hold on;
for n = 1:3
    [N1,N2] = size(B{n}); 
     for i = 1:N1
         for j = 1:N2
             if B{n}(i,j)==1
                 plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-k');
             end
             if B{n}(i,j)==2
                 plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[0.7,0.7,0.7]);
             end
         end
     end
end

 for n = 1:4
     scatter(px{n},(4+n)*(1+0*px{n}),4*node_size{n},'k','filled');
 end
    