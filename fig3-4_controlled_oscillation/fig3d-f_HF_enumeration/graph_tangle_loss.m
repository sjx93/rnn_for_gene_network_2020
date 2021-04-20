function z = graph_tangle_loss(px,B)

z = 0;
for n = 1:3
    N1 = size(px{n},1);
    N2 = size(px{n+1},1);
    d = abs(px{n}*ones(1,N2)-ones(N1,1)*px{n+1}'); %[N1,N2]
    B1 = B{n}==1; %[N1,N2]
    B2 = B{n}==2;
    z = z + sum(sum(d.*B1 + d.*B2));
end