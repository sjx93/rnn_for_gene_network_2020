pdir = 'trajs_fig4/';

N_nets = 8;

Amp1 = zeros(N_nets,2);
Amp2 = zeros(N_nets,2);
for i = 1:N_nets
    run_num = 60+i;
    g1 = csvread([pdir,'run',num2str(run_num),'_traj_g1.csv']);

    A1 = g1(1:20,:);
    A2 = g1(21:40,:);
    
    Amp1(i,:) = [min(A1(16,150:end)), max(A1(16,150:end))];
    Amp2(i,:) = [min(A2(16,150:end)), max(A2(16,150:end))];
end

hold on;
X1 = [(1:N_nets)'; flipud((1:N_nets)')];
Y1 = [Amp1(1:N_nets,1); flipud(Amp1(1:N_nets,2))];
patch(X1,Y1,[0,0,1],'linestyle','none'); alpha(0.5);
plot((1:N_nets)',Amp1(1:N_nets,:),'-','color',[0,0,0.9],'linewidth',2);

X2 = [(1:N_nets)'; flipud((1:N_nets)')];
Y2 = [Amp2(1:N_nets,1); flipud(Amp2(1:N_nets,2))];
patch(X2,Y2,[0.5,0.7,1],'linestyle','none'); alpha(0.5);
plot((1:N_nets)',Amp2(1:N_nets,:),'-','color',[0.5,0.7,0.9],'linewidth',2);

%plot([0,10],[0.8,0.8],'--k','linewidth',0.5);
xlim([1,N_nets]); ylim([0,1]); grid on; box on;
set(gca,'xtick',[1:N_nets]);
set(gca,'ytick',[0,0.5,1]);
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gcf,'unit','centimeters','position',[1,2,9,3]);

figure;
g1 = csvread([pdir,'run',num2str(61),'_traj_g1.csv']);
a1 = g1(16,:)'; a1(1:59) = a1(60);
a2 = g1(20+16,:)'; a2(1:59) = a2(60);
ts = linspace(-6,24,300);
plot(ts, a2, 'color', [0.5,0.7,1], 'linewidth', 2); hold on;
plot(ts, a1, 'color', [0,0,1], 'linewidth', 2); hold off;
xlim([-5,20]); ylim([0,1]); grid on; box on;
set(gca,'xtick',[0,10,20]);
set(gca,'ytick',[0,0.5,1]);
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gcf,'unit','centimeters','position',[1,2,2.4,3]);