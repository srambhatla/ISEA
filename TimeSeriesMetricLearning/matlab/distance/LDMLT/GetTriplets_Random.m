function Triplet = GetTriplets_Random(y, factor, Y_kind)
Triplet=[];

for k=Y_kind
    k_In=find(y==k);
    k_Out=find(y~=k);
    l_In=length(k_In);
    l_Out=length(k_Out);
    j=1;
    while j<ceil(l_In*factor)
        rand_value=rand(1,3);
        index_i=ceil(rand_value(1)*l_In);
        index_j=ceil(rand_value(2)*l_In);
        index_k=ceil(rand_value(3)*l_Out);
        if index_i==index_j
            continue;
        end
        select_i=k_In(index_i);
        select_j=k_In(index_j);
        select_k=k_Out(index_k);
        dist_ik=(k_Out(index_k)-k_In(index_i))'*(k_Out(index_k)-k_In(index_i));
        dist_jk=(k_Out(index_k)-k_In(index_j))'*(k_Out(index_k)-k_In(index_j));
        
        if dist_jk>dist_ik
            select_j=k_In(index_i);
            select_i=k_In(index_j);
        end
        Triplet=[Triplet;select_i,select_j,select_k];
        j=j+1;
    end
end