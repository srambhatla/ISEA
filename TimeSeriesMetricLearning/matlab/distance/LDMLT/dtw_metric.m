function [Dist,MTS_E1,MTS_E2]=dtw_metric(MTS_1,MTS_2,M)
% MTS_i is t by d
MTS_1=MTS_1';
MTS_2=MTS_2';


[~,col_1]=size(MTS_1); 
[row,col_2]=size(MTS_2); 

d=zeros(col_1,col_2);
D1=MTS_1'*M*MTS_1;
D2=MTS_2'*M*MTS_2;
D3=MTS_1'*M*MTS_2;

for i=1:col_1
    for j=1:col_2
        d(i,j)=D1(i,i)+D2(j,j)-2*D3(i,j);
    end
end
D=zeros(size(d));
D(1,1)=d(1,1);

for m=2:col_1
    D(m,1)=d(m,1)+D(m-1,1);
end
for n=2:col_2
    D(1,n)=d(1,n)+D(1,n-1);
end
for m=2:col_1
    for n=2:col_2
        D(m,n)=d(m,n)+min(D(m-1,n),min(D(m-1,n-1),D(m,n-1))); 
    end
end

Dist=D(col_1,col_2);
n=col_2;
m=col_1;
k=1;
w=[col_1 col_2];

while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0 
        n=n-1;
    else 
      [~,number]=min([D(m-1,n),D(m,n-1),D(m-1,n-1)]);
      switch number
      case 1
        m=m-1;
      case 2
        n=n-1;
      case 3
        m=m-1;
        n=n-1;
      end
  end
    k=k+1;
    w=[m n; w]; 
end
MTS_E1=zeros(row,k);
MTS_E2=zeros(row,k);
for i=1:row
MTS_E1(i,:)=MTS_1(i,w(:,1));
MTS_E2(i,:)=MTS_2(i,w(:,2));
end

