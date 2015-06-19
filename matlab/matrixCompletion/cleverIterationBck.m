


LipU = mu + norm(mask(mb,:).*V(rb,:))^2;
tmp=(U(mb,:)*V(:,:)-input(mb,:)).*mask(mb,:);
gradU = U(mb,rb) *mu + sum(   tmp.*V(rb,:)  ) ; 

LipV = mu + norm(mask(:,nb).*U(:,rb))^2;
tmp=(U(:,:)*V(:,nb)-input(:,nb)).*mask(:,nb);
gradV = V(rb,nb) *mu + sum(   tmp.*U(:,rb)  )  ;



LipUV = (  V(rb,nb) * U(mb,rb) + ...
     U(mb,:)*V(:,nb)  - input(mb,nb))*mask(mb,nb);
 
 
H=[LipU LipUV; LipUV, LipV];

 if(det(H)<0)
     H
     it
     disp('XXXXXXXXXXXXXX');
 end

t=-H\[gradU; gradV];
 

DeltaU=t(1);
DeltaV=t(2);


U(mb,rb)=U(mb,rb)+DeltaU;
V(rb,nb)=V(rb,nb)+DeltaV;


 