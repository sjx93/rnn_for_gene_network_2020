num_nets = [6,5];
name_offset = [10,20];

figure;
for i_panel = 1:2
    subplot(1,2,i_panel);
    Z = zeros(num_nets(i_panel),2);
    for n = 1:num_nets(i_panel)
        output_name = name_offset(i_panel)+n;
        g1 = csvread(['trajs/run' num2str(output_name) '_traj_g1.csv']);
        g2 = csvread(['trajs/run' num2str(output_name) '_traj_g2.csv']);

        input_val = 20;
        sensitivity = max(g1(input_val,40:end)) - g1(input_val,40);
        error = abs(g1(input_val,end) - g1(input_val,40));
        Z(n,:) = [sensitivity,error];
    end

    plot(1:num_nets(i_panel), Z(:,1), '-ok','linewidth',1.5,'markersize',4,'markerfacecolor','k'); hold on;
    plot(1:num_nets(i_panel), Z(:,2), ':sk','linewidth',1.5,'markersize',8,'markerfacecolor','none'); hold off;
    xlim([1,num_nets(i_panel)]); ylim([0,0.5]); grid on;
    set(gca,'xtick',1:num_nets(i_panel));
    set(gca,'ytick',[0,0.2,0.4]);
    set(gcf,'unit','centimeters','position',[1,2,18,4]);
end