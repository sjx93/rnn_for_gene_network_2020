pdir ='nets_fig3d-f/'; 
a0_threshold=0.0005;

cmp_threshold = 1;
    
A_nn4 = [];
for ci = 1:201
    fname = ls([pdir,'run',num2str(ci),'.csv']);
    if ~isempty(fname)
        a00= csvread([pdir,'run',num2str(ci),'.csv'])';
        a0 = a00(1,:);
        a1 = a0 .* (abs(a0)>a0_threshold);
        A_nn4 = [A_nn4; sign(a1)];
    end
end

%A_nn4 = A_nn;
redundant_topology = zeros(size(A_nn4,1),1);
for i = 1:size(A_nn4,1)
    % redundent topology
    if (i>=2) && (0 == min(sum(ones(i-1,1)*A_nn4(i,:)~=A_nn4(1:(i-1),:), 2)))
        redundant_topology(i)=1;
    end
    % violating basic connectivity
    if (A_nn4(i,1)==0) && (A_nn4(i,2)==0)
        redundant_topology(i)=1;
    end
    if (A_nn4(i,3)==0) && (A_nn4(i,4)==0)
        redundant_topology(i)=1;
    end
    if (A_nn4(i,6)==0) || (A_nn4(i,7)==0)
        redundant_topology(i)=1;
    end
end

A_nn = A_nn4(redundant_topology==0,:);
fprintf('all NN proposed topology: %d\n', size(A_nn,1));

load fig3d-f_HF_enumeration/HF_enumeration_pxforscat.mat
topo_found = cell(4,1);
for n = 1:4
    A_hill = A{n}; %[A{1};A{2};A{3};A{4}];

    N0 = size(A_hill,1);
    N1 = size(A_nn,1);
    distance0 = 100*ones(N1,N0);
    for i = 1:N1
        for j = 1:N0
            %distance0(i,j) = sum(A_nn(i,:)~=A_hill(j,:));
            distance0(i,j) = sum(abs(A_nn(i,:)-A_hill(j,:)));
        end
    end
    distance = min(distance0,[],1);
    topo_found{n} = find(distance<=cmp_threshold);
end
% plot

figure;
hold on;
if 1
for n = 1:3
    [N1,N2] = size(B{n}); 
     for i = 1:N1
         for j = 1:N2
             if B{n}(i,j)==1
                 if (~isempty(topo_found{n})) && (~isempty(topo_found{n+1}))
                     if max(topo_found{n}==i) && max(topo_found{n+1}==j)
                         plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-r');
                     else
                         plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-','color',[0.65,0.65,0.65]);
                     end
                 else
                     plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-','color',[0.65,0.65,0.65]);
                 end
             end
             if 0 %B{n}(i,j)==2
                 if (~isempty(topo_found{n})) && (~isempty(topo_found{n+1}))
                     if max(topo_found{n}==i) && max(topo_found{n+1}==j)
                         plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[1,0.7,0.7]);
                     else
                         plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[0.7,0.7,0.7]);
                     end
                 else
                     plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[0.7,0.7,0.7]);
                 end
             end
         end
     end
end
end


 for n = 1:4
     %color = 0.3*ones(size(px{n},1),3);
     %color(topo_found{n},1) = 1;
     %color(topo_found{n},2) = 0;
     %color(topo_found{n},3) = 0;
     topo_not_found = 1:size(px{n},1); topo_not_found(topo_found{n}) = [];
     scatter(px{n}(topo_not_found),(4+n)*(1+0*px{n}(topo_not_found)),10+node_size{n}(topo_not_found),[1,1,1],'filled','markeredgecolor','k');
     scatter(px{n}(topo_found{n}),(4+n)*(1+0*px{n}(topo_found{n})),10+node_size{n}(topo_found{n}),'r','filled','markeredgecolor','k');
 end

xlim([-4,4]);  ylim([4,9]); alpha(0.5);
set(gcf,'unit','centimeters','position',[1,2,8,6]);

A_found = [A{1}(topo_found{1},:);A{2}(topo_found{2},:);A{3}(topo_found{3},:);A{4}(topo_found{4},:)];
clustergram(-tanh(100*A_found),'Cluster',1);
fprintf('all NN proposed topology: %d\n', size(A_nn,1));
fprintf('all HE topology: %d\n', size([A{1};A{2};A{3};A{4}],1));
fprintf('overlapping topology in Hill: %d\n', size(topo_found{1},2)+size(topo_found{2},2)+size(topo_found{3},2)+size(topo_found{4},2));
