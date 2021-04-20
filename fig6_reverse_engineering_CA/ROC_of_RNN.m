paras_ground_truth = csvread('CA_ground_truth_models/para/paras_ground_truth.csv');
netdir = 'trained_RNN_models/nets';

figure; plot_case = 2;
ROC_score = zeros(25,2); %[CAmodel_num, AOC]

net_pred1_v = zeros(25,20*10);
net_pred2_v = zeros(25,20*10);
net_ground_truth_v = zeros(25,20*10);
Ks_ground_truth_v = zeros(25,20*10);

for CAmodel_i = 1:25
    fname = ['model_',num2str(CAmodel_i),'.csv'];
    ROC_score(CAmodel_i,1) = CAmodel_i;
    
    net_pred_ = csvread([netdir,'/',fname]);
    net_pred1 = net_pred_(1:20,:);
    %net_pred2 = net_pred_(22:41,:);
    
    net_ground_truth= reshape(paras_ground_truth(CAmodel_i,401:600),[10,20])';
    Ks_ground_truth = reshape(paras_ground_truth(CAmodel_i,201:400),[10,20])';
    
    net_ground_truth_v(CAmodel_i,:) = reshape(net_ground_truth,[],1)';
    Ks_ground_truth_v(CAmodel_i,:) = reshape(Ks_ground_truth,[],1)';
    net_pred1_v(CAmodel_i,:) = reshape(net_pred1,[],1)';
    %net_pred2_v(CAmodel_i,:) = reshape(net_pred2,[],1)';
end


% compute and plot ROC
for CAmodel_i = 1:25
    % predicted sign and strength for true act/inh/null links
    n_act = net_ground_truth_v(CAmodel_i,:)==1;
    true_act = sort(net_pred1_v(CAmodel_i,n_act))';

    n_inh = net_ground_truth_v(CAmodel_i,:)==-1;
    true_inh = sort(net_pred1_v(CAmodel_i,n_inh))';

    n_null= net_ground_truth_v(CAmodel_i,:)==0;
    true_null= sort(net_pred1_v(CAmodel_i,n_null))';

    % compute ROC
    threshold = true_act; N_act = size(true_act,1);
    ROC_Act = zeros(N_act,2);
    for i = 1:N_act
        ROC_Act(i,1) = sum(true_act>=threshold(i)); %true_positive
        ROC_Act(i,2) = sum(true_inh>=threshold(i)) +...
                       sum(true_null>=threshold(i)); %false_positive
    end
    threshold = true_inh; N_inh = size(true_inh,1);
    ROC_Inh = zeros(N_inh,2);
    for i = 1:N_inh
        ROC_Inh(i,1) = sum(true_inh<=threshold(i)); %true_positive
        ROC_Inh(i,2) = sum(true_act<=threshold(i)) +...
                       sum(true_null<=threshold(i)); %false_positive
    end
    % normalize ROC
    ROC_Act(:,1) = ROC_Act(:,1)/max(ROC_Act(:,1));
    ROC_Act(:,2) = ROC_Act(:,2)/max(ROC_Act(:,2));
    ROC_Inh(:,1) = ROC_Inh(:,1)/max(ROC_Inh(:,1));
    ROC_Inh(:,2) = ROC_Inh(:,2)/max(ROC_Inh(:,2));


    % compute area under ROC curve
    score = 0.5*(-trapz(ROC_Act(:,2), ROC_Act(:,1))+...
                  trapz(ROC_Inh(:,2), ROC_Inh(:,1)));
    ROC_score(CAmodel_i,2) = score;
    % plot ROC
    switch plot_case
        case 1
            subplot(5,5,CAmodel_i);
            plot(ROC_Act(:,2), ROC_Act(:,1),'-','linewidth',2,'color',[0.36,0.61,0.84]); hold on;
            plot(ROC_Inh(:,2), ROC_Inh(:,1),'-','linewidth',2,'color',[0.8,0,0]);
            plot([0,1], [0,1], '--k','linewidth',1); hold off;
            axis equal; grid on; box on;
            xlim([0,1]); ylim([0,1]);
            set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
        case 2
            subplot(1,2,1); hold on;
            plot(ROC_Act(:,2), ROC_Act(:,1),'-','linewidth',1,'color',[0.36,0.61,0.84]);
            subplot(1,2,2); hold on;
            plot(ROC_Inh(:,2), ROC_Inh(:,1),'-','linewidth',1,'color',[0.8,0,0]);
    end
  
end

switch plot_case
    case 1
        set(gcf, 'unit', 'centimeters', 'Position', [1,1,11,16]);
    case 2
        subplot(1,2,1);
        plot([0,1], [0,1], '--k','linewidth',1); hold off;
        axis equal; grid on; box on;
        xlim([0,1]); ylim([0,1]);
        set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
        
        subplot(1,2,2);
        plot([0,1], [0,1], '--k','linewidth',1); hold off;
        axis equal; grid on; box on;
        xlim([0,1]); ylim([0,1]);
        set(gca,'xticklabel',[]); set(gca,'yticklabel',[]);
        
        set(gcf, 'unit', 'centimeters', 'Position', [1,0,16,8]);
end
