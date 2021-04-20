%pdir = '8edges/';
%pdir = '7edges/';
%pdir = '6edges/';
pdir = '5edges/';

A = [];
%for nL = 1
%for nL = 1:6
%for nL = 1:13
for nL = 1:12

    links_primary = reshape(csvread([pdir,num2str(nL),'/links_primary.csv'])',1,8);
    N_links = sum(links_primary);
    fname = ls([pdir,num2str(nL),'/para/*.csv']);
    for nT = 1:size(fname,1)
        isnumber = (fname(nT,:)>='0')&(fname(nT,:)<='9');
        topology_num = str2double(fname(nT,isnumber));
        links0= double(char(dec2bin(topology_num)))-'0';
        links00 = [zeros(1,N_links-length(links0)),links0];
        links = links_primary;
        links(links==1) = (-1).^links00;
        
        para = csvread([pdir,num2str(nL),'/para/',fname(nT,:)]);
        if size(para,1)>1
            A = [A;[size(para,1),topology_num,links]];
        end
    end
end
csvwrite([pdir,'allforscat.csv'],A);

        
        