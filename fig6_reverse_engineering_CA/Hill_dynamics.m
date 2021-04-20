function x_out = Hill_dynamics(CAmodel_num)
paras = csvread('CA_ground_truth_models/para/paras_ground_truth.csv');
n = CAmodel_num;
b = reshape(paras(n,1:200),[10,20])';
K = reshape(paras(n,201:400),[10,20])';
links = reshape(paras(n,401:600),[10,20])';

BATCH = 2;
TIMESTEP = 0.2;
Gs = 10; Gin= Gs;
Nx = 20+1;
time_points = 300;
gamma = 1;

X0 = zeros([BATCH,time_points,Nx,Gs]);

%x0 = rand([BATCH,Nx,Gs]).*repmat(rand([1,Nx,1])<0.5,[BATCH,1,Gs]);
x0 = zeros([BATCH,Nx,Gs]); x0(:,fix(Nx/2)+1,:) = double(rand(BATCH,Gs)<0.5);

for t0 = 1:time_points
    %x0_neighbor = 0.5*np.concatenate([np.ones([BATCH,1,Gs]), x0[:,0:(Nx-1),:]], axis=1) +...
    %              0.5*np.concatenate([x0[:,1:,:], np.ones([BATCH,1,Gs])], axis=1)
    x0_left = zeros(BATCH,Nx,Gs); 
    x0_left(:,2:Nx,:) = x0(:,1:(Nx-1),:);
    x0_left(:,1,:) = x0(:,Nx,:);
    x0_right = zeros(BATCH,Nx,Gs); 
    x0_right(:,1:(Nx-1),:) = x0(:,2:Nx,:);
    x0_right(:,Nx,:) = x0(:,1,:);
    x0_neighbor = 0.5*x0_left + 0.5*x0_right; %[BATCH,Nx,Gin=Gs]
    
    x0_input = zeros([BATCH,Nx,2*Gs]);
    x0_input(:,:,1:Gs) = x0_neighbor;
    x0_input(:,:,(Gs+1):(2*Gs)) = x0;
    f0 = Hill_model(x0_input,b,K,links,Nx);
    x0 = (1-gamma*TIMESTEP)*x0 + f0*TIMESTEP;
    X0(:,t0,:,:) = x0;
end
imshow(squeeze(X0(1,1:3:end,:,1:3)),'initialmagnification',500);
%imshow(squeeze(X0(1,1:3:end,:,[4,5,8])),'initialmagnification',500); % model #24
%imshow(squeeze(X0(1,1:3:end,:,[1,4,10])),'initialmagnification',500); % model #7
x_out = squeeze(X0(1,10:1:end,:,:));
end
    