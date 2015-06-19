


 
LipU = mu + norm(mask(mb,:).*V(rb,:))^2;
tmp=(U(mb,:)*V(:,:)-input(mb,:)).*mask(mb,:);
grad = U(mb,rb) *mu + sum(   tmp.*V(rb,:)  ) ; 
DeltaU=-grad /LipU;

U(mb,rb)=U(mb,rb)+DeltaU;
 


 
LipV = mu + norm(mask(:,nb).*U(:,rb))^2;
tmp=(U(:,:)*V(:,nb)-input(:,nb)).*mask(:,nb);
grad = V(rb,nb) *mu + sum(   tmp.*U(:,rb)  )  ;
DeltaV=- grad/LipV;
V(rb,nb)=V(rb,nb)+DeltaV;

%disp(sprintf('%d %f  %f    ',it,DeltaU, DeltaV));

 