cmp_threshold = [0,1];
case_name = {'direct hit', 'HF-relevant'};

for case_i = 1:2

pdir ='nets_fig3d-f/'; a0_threshold=0.0005;
rearrange_nodes = 0;
if rearrange_nodes
    A_nn4 = [];
    %for ci = 201:400
    for ci = 1:200
        fname = ls([pdir,'run',num2str(ci),'.csv']);
        if ~isempty(fname)
            a00= csvread([pdir,'run',num2str(ci),'.csv'])';
            %a0 = mean(a00,1);
            a0 = a00(1,:);
            a1 = a0 .* (abs(a0)>a0_threshold);
            A_nn4 = [A_nn4; sign(a1)];
        end
    end

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
    %disp(size(A_nn));

    % count frequency of appearing for each topology 
    A_nn_count = zeros(size(A_nn,1),1);
    for i = 1:size(A_nn4,1)
        n_tmp = sum(ones(size(A_nn,1),1)*A_nn4(i,:)~=A_nn,2)==0;
        A_nn_count(n_tmp) = A_nn_count(n_tmp)+1;
    end

    % pack A_nn into cell(4,1)
    A_NN = cell(4,1);
    node_size = cell(4,1);
    px = cell(4,1);
    wplot = [1.6,3,3,3];

    N_link = sum(A_nn~=0,2);
    for n = 1:4
        A_NN{n} = A_nn(N_link==(4+n),:);
        node_size{n} = A_nn_count(N_link==(4+n),:);
        px{n} = wplot(n)*linspace(-1,1,size(A_NN{n},1))';
    end

    B = cell(3,1);
    for n = 1:3
        N1 = size(A_NN{n},1);
        N2 = size(A_NN{n+1},1);
        B0 = zeros(N1,N2);
        for i = 1:N1
            for j = 1:N2
                %B0(i,j) = sum(A_NN{n}(i,:)~=A_NN{n+1}(j,:));
                B0(i,j) = sum(abs(A_NN{n}(i,:)-A_NN{n+1}(j,:)));
            end
        end
        B{n} = B0;
    end

%     loss0 = 9999;
%     for i = 1:10000 %iteration of rearranging
%         px_new = px;
%         n = randi(4);
%         %n = randi(3)+1;
%         perm_i = randi(size(px{n},1),2,1);
%         px_new{n} = px{n};
%         px_new{n}(perm_i(1)) = px{n}(perm_i(2));
%         px_new{n}(perm_i(2)) = px{n}(perm_i(1));
%         loss = graph_tangle_loss(px_new,B);
%         if (loss <= loss0) || rand()<((4e-6)*(loss-loss0))
%             px = px_new;
%             loss0 = loss;
%             disp(loss);
%         end
%     end
    
else
    load 'nets_fig3d-f/pxforscat_linkko90.mat';
end
    

px_protect = px;
A_protect = A_NN;
B_protect = B;
node_size_protect = node_size;
 
load fig3d-f_HF_enumeration/HF_enumeration_pxforscat.mat
A_hill_ = [A{1};A{2};A{3};A{4}];
A_NN = A_protect;
B = B_protect;
px = px_protect;
node_size =  node_size_protect;

topo_found = cell(4,1);
for n = 1:4
    A_nn_ = A_NN{n}; %[A{1};A{2};A{3};A{4}];

    N0 = size(A_nn_,1);
    N1 = size(A_hill_,1);
    distance0 = 100*ones(N1,N0);
    for i = 1:N1
        for j = 1:N0
            %distance0(i,j) = sum(A_nn(i,:)~=A_hill(j,:));
            distance0(i,j) = sum(abs(A_hill_(i,:)-A_nn_(j,:)));
        end
    end
    distance = min(distance0,[],1);
    topo_found{n} = find(distance<=cmp_threshold(case_i));
end

% figure;
% hold on;
% for n = 1:3
%     [N1,N2] = size(B{n}); 
%      for i = 1:N1
%          for j = 1:N2
%              if B{n}(i,j)==1
%                  if (~isempty(topo_found{n})) && (~isempty(topo_found{n+1}))
%                      if max(topo_found{n}==i) && max(topo_found{n+1}==j)
%                          plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-r');
%                      else
%                          plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'-','color',[0,0,0]);
%                      end
%                  end
%              end
%              if 0 %B{n}(i,j)==2
%                  if (~isempty(topo_found{n})) && (~isempty(topo_found{n+1}))
%                      if max(topo_found{n}==i) && max(topo_found{n+1}==j)
%                          plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[1,0.7,0.7]);
%                      else
%                          plot([px{n}(i),px{n+1}(j)],4+[n,n+1],'color',[0.7,0.7,1]);
%                      end
%                  end
%              end
%          end
%      end
% end

% for n = 1:4
%     topo_not_found = 1:size(px{n},1); topo_not_found(topo_found{n}) = [];
%     scatter(px{n}(topo_not_found),(4+n)*(1+0*px{n}(topo_not_found)),5+8*node_size{n}(topo_not_found),[0.6,0.6,0.6],'filled','markeredgecolor','k');
%     scatter(px{n}(topo_found{n}),(4+n)*(1+0*px{n}(topo_found{n})),5+8*node_size{n}(topo_found{n}),'r','filled','markeredgecolor','k');
% end
% xlim([-7,7]);  ylim([4,9]); alpha(0.6);

fprintf([case_name{case_i}, ': %d\n'], size(topo_found{1},2)+size(topo_found{2},2)+size(topo_found{3},2)+size(topo_found{4},2));
end
fprintf('all NN proposed topology: %d\n', size(A_NN{1},1)+size(A_NN{2},1)+size(A_NN{3},1)+size(A_NN{4},1));
