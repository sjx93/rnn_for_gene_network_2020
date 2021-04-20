for i = 1:10
    train_loss = csvread([num2str(i),'/savenet/accuracy.csv']);
    if train_loss(end,2) < 0.0015
        xs = linspace(0.05,0.95,91)';
        X = csvread([num2str(i),'/patterning/1_wt.csv']);
        hold on;
        plot(xs, X(:,1), 'color', [0.3,0.3,1], 'linewidth', 1);
        plot(xs, X(:,2), 'color', [0.3,0.7,0.3], 'linewidth', 1);
        plot(xs, X(:,3), 'color', [1,0.5,0.5], 'linewidth', 1);
        plot(xs, X(:,4), 'color', [0.8,0.8,0.3], 'linewidth', 1);
        xlim([0.05,0.95]); ylim([0,1.3]);
    end
end

xs = linspace(0.05,0.95,91)';
X = csvread('Data_frame/FlyEX_nc14T7_gap-genes.csv');
hold on;
plot(xs, X(:,1),':', 'color', [0,0,0.4], 'linewidth', 3);
plot(xs, X(:,2),':', 'color', [0,0.3,0], 'linewidth', 3);
plot(xs, X(:,3),':', 'color', [0.5,0.1,0.1], 'linewidth', 3);
plot(xs, X(:,4),':', 'color', [0.3,0.3,0], 'linewidth', 3);
xlim([0.05,0.95]); ylim([0,1.3]); box on; grid on;

set(gca,'xtick',[0.25,0.5,0.75]); 
set(gca,'ytick',[0.5,1]);
set(gca,'xticklabel',[]); 
set(gca,'yticklabel',[]);
set(gcf,'unit','centimeters','position',[1,1,7,3.5]);