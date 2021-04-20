load 'HF_enumeration_pxforscat.mat'

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
     scatter(px{n},(4+n)*(1+0*px{n}),1.5*node_size{n},'k','filled');
 end
 
xlim([-4,4]);  ylim([4,9]);