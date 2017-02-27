%60 points, 70 features 
clear all
n0= 37; 
n1= 23; 

Feature_Name={'G1'	'G2'	'G3'	'LOC51203'	'G4'	'ALDH4'	'G5'	'FGF18'	'G6'	'G7'	'BBC3'	'G8'	'KIAA1442'	'DC13'	'CEGP1'	'EXT1'	'FLT1'	'GNAZ'	'OXCT'	'MMP9'	'G9'	'G10'	'ECT2'	'GMPS'	'HEC'	'WISP1'	'PK428'	'SERF1A'	'G11'	'GSTM3'	'G12'	'RAB6B'	'G13'	'G14'	'UCH37'	'PECI'	'KIAA1067'	'G15'	'TGFB3'	'KIAA0175'	'COL4A2'	'L2DTL'	'HSA250839'	'DCK'	'G16'	'DKFZP564D0462'	'SLC2A3'	'PECI.1'	'ORC6L'	'RFC4'	'G17'	'G18'	'CFFM4'	'MCM6'	'AP2B1'	'G19'	'IGFBP5'	'LOC57110'	'MP1'	'IGFBP5.1'	'NMU'	'AKAP2'	'G20'	'PRC1'	'G21'	'CENPA'	'SM.20'	'CCNE2'	'ESM1'	'FLJ11190'}; 

fid= fopen('Training_Data.txt','r');
fgetl(fid);
train_data=[]; 

while ~feof(fid)
a=fgetl(fid);
temp_str= textscan(a,'%f', 'delimiter', ' ');
temp_double= temp_str{1}; 
temp_double= temp_double'; 
train_data= [train_data ; temp_double] ;
end  

fclose(fid); 

train_data_label= train_data(:,end);
train_data= train_data(:,2:end-1); 

fid= fopen('C:\Users\Home\Downloads\Testing_Data.txt','r');
fgetl(fid);
test_data=[]; 


while ~feof(fid)
a=fgetl(fid);
temp_str= textscan(a,'%f', 'delimiter', ' ');
temp_double= temp_str{1}; 
temp_double= temp_double'; 
test_data= [test_data ; temp_double] ;
end  

fclose(fid); 

test_data_label= test_data(:,end);
test_data= test_data(:,2:end-1); 


Best_Feature_Set= zeros(22,8);
Resub_Error= zeros(22,1); 
True_Error= zeros(22,1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Feature set 1 : using dlda and exhaustive search - 3 features 

for index= 1:3

	comb= combnk(1:70,index);
	comb=sortrows([comb sum(comb,2)], index+1);
	comb=comb(:,1:index);
	
	total_sets= size(comb,1); %total number of combinations 
	error_count1 = zeros (1,total_sets); 

	best_error1= 1; 
	best_features1=[]; %best triplet 
	best_a1=0; 
	best_b1=0; 

	for j=1:total_sets %for each feature,we construct the dlda classifier and check error  

		data0= train_data(1:n0,comb(j,:)); 
		data1= train_data(n0+1:60,comb(j,:));

		%classifier design:
		u0= mean(data0); 
		u1= mean(data1); 
		cov0= cov(data0);
		cov1= cov(data1);
	
		cov_total= zeros(index, index); 

			for new_index=1: index
						cov_total(new_index, new_index)=((n0-1)*cov0(new_index, new_index) + (n1-1)*cov1(new_index, new_index))/ (n0+n1-2);
			end


		a=  pinv(cov_total)*(u1 - u0)'; 
		b= -(1/2)*((u1 - u0)* pinv(cov_total)*(u1 + u0)')+ log((n1)/(n0));

		%Resubstitution error calculation:
		x= [ data0 ; data1]; %vectors for resubstitution error 
		
		for k= 1:60 
			temp = sign(a'*x(k,:)' + b ); 
			
		%	if(temp== label_class_0)	
			if(temp < 0)	
				assign_label= 0; 
			else 
				assign_label= 1; 
			end
			
			if(assign_label ~= train_data_label (k))
				error_count1(j) = error_count1(j) + 1/60; 
			end 
		end

		
		if (error_count1(j)< best_error1)  
			best_error1 = error_count1(j); 
			best_a1= a ; 
			best_b1= b; 
			best_features1= comb(j,:); 			
		end		
	end 
	
	Best_Feature_Set(index,:)= [best_features1 zeros(1, 8-index)]; 
	Resub_Error(index)= best_error1; 
	best_train1= train_data(:,best_features1)	;		

	
	%True Error 
	for l= 1:235 
			temp = sign(best_a1'*test_data(l,best_features1)' + best_b1 ); 
			
			if(temp < 0)	
				assign_label= 0; 
			else 
				assign_label= 1; 
			end
			
			if(assign_label ~= test_data_label (l))
				True_Error(index) = True_Error(index) + 1/235; 
			end 
	end
	
end 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Feature set 2 : using 3nn and exhaustive search - 3 features 
for index=1:3
	comb= combnk(1:70,index);
	comb=sortrows([comb sum(comb,2)], index+1);
	comb=comb(:,1:index);
	
	total_sets= size(comb,1); %total number of combinations 
	error_count2 = zeros (1,total_sets); 

	best_error2= 1; 
	best_features2=[]; %best triplet 


	for j=1:total_sets 
		
		data= train_data(1:60,comb(j,:)); 
		% no explicit classifier construction- knn is 'lazy' classifier

		for k=1:60
			true_label= train_data_label(k); 
			dist= ((data(:,:)- (repmat(data(k,:),60,1))).^2); 
			dist= sum(dist,2); 

			temp= sortrows([dist  train_data_label]); 
			neighbour_index= temp(1:3,2);

			if (sum(neighbour_index) > 1)
				predicted_label= 1 ;
			else 
				predicted_label= 0 ;
			end

			
			%Resubstitution error:
			if(true_label ~= predicted_label)
				error_count2(j) = error_count2(j) + 1/60 ; 
			end 
			
		end
		
		
		if (error_count2(j)< best_error2)
			best_error2 = error_count2(j); 
			best_features2= comb(j,:); 
		end
		
	end 

	
	Best_Feature_Set(3+index,:)= [best_features2 zeros(1, 8-index)]; 
	Resub_Error(3+index)= best_error2; 
	best_train2= train_data(:,best_features2)	;		

	
	%True Error 
	for l= 1:235 
			true_label= test_data_label(l); 
			dist= ((train_data(:,best_features2)- (repmat(test_data(l,best_features2),60,1))).^2); 
			dist= sum(dist,2); 

			temp= sortrows([dist  train_data_label]); 
			neighbour_index= temp(1:3,2);

			if (sum(neighbour_index) > 1)
				assign_label= 1 ;
			else 
				assign_label= 0 ;
			end

			if(assign_label ~= test_data_label (l))
				True_Error(index+3) = True_Error(index+3) + 1/235; 
			end 
	end
	
end 
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Feature set 3 : using dlda and forward search - 3 features 
best_features3=[]; 

for m=1:8
	error_count3 = zeros (1,70); 
	best_error3= 1; 
	best_a3=0; 
	best_b3=0; 
	best_label_class_0= 0 ; 
	best_label_class_1= 0 ; 
	
	for j=1:70 %for each feature,we construct the dlda classifier by combining it with already selected features and check error  
	
		if(sum(best_features3 == j))
			continue; %skip already included feature
		else 
		
		
			feature_set=[best_features3 j]; %combine this index with existing features
			data0= train_data(1:n0, feature_set); 
			data1= train_data(n0+1:60,feature_set);

			%classifier design:
			u0= mean(data0); 
			u1= mean(data1); 
			cov0= cov(data0);
			cov1= cov(data1);
			%doubt? 
			%cov_total=((n0-1)*cov0 + (n1-1)*cov1 )/ (n0+n1-2);
			
			cov_total= zeros(length(feature_set), length(feature_set)); 
			
			for new_index=1: length(feature_set)
				cov_total(new_index, new_index)=((n0-1)*cov0(new_index, new_index) + (n1-1)*cov1(new_index, new_index) )/ (n0+n1-2);
			end
			
			a=  pinv(cov_total)*(u1 - u0)'; 
			b= -(1/2)*((u1 - u0)* pinv(cov_total)*(u1 + u0)') + log((n1)/(n0));
			label_class_0= sign(a'*u0' + b); 
			label_class_1= sign(a'*u1' + b); 

			
			%Resubstitution error:
			x= [ data0 ; data1]; %vectors for resubstitution error 

			
			for k= 1:60 
				temp = sign(a'*x(k,:)' + b ); 
				
				if(temp< 0)	
					assign_label= 0; 
				else 
					assign_label= 1; 
				end
				
				if(assign_label ~= train_data_label (k))
					error_count3(j) = error_count3(j) + 1/60; 
				end 
			end

			
			if (error_count3(j)< best_error3)
				best_error3 = error_count3(j); 
				best_index= j;  
				best_a3= a ; 
				best_b3= b; 
				best_label_class_0= label_class_0; 
				best_label_class_1= label_class_1;
			end
			
		end
	end
	
	
	best_features3=[best_features3 best_index]; 
	Best_Feature_Set(6+m,:) = [best_features3 zeros(1,8-m)] ; 
	Resub_Error(6+m)= best_error3; 

	
	%True Error 
	for l= 1:235 
			temp = sign(best_a3'*test_data(l,best_features3)' + best_b3 ); 
			
			if(temp < 0 )	
				assign_label= 0; 
			else 
				assign_label= 1; 
			end
			
			if(assign_label ~= test_data_label (l))
				True_Error(6+m) = True_Error(6+m) + 1/235; 
			end 
	end
	
end	

	best_train3= train_data(:,best_features3)	;		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Feature set 4 : using 3nn and forward search - 3 features 
best_features4=[]; 

for m=1:8
	error_count4 = zeros (1,70); 
	best_error4= 1; 
	
	for j=1:70 %for each feature,we construct the dlda classifier by combining it with already selected features and check error  
	
		if(sum(best_features4 == j))
			continue; %skip already included feature
		else 
			feature_set=[best_features4 j]; 
			data= train_data(1:60,feature_set);
			% no explicit classifier construction- knn is 'lazy' classifier

			for k=1:60
				true_label= train_data_label(k); 
				dist= ((data(:,:)- (repmat(data(k,:),60,1))).^2); 
				dist= sum(dist,2); 

				temp= sortrows([dist  train_data_label]); 
				neighbour_index= temp(1:3,2);

				if (sum(neighbour_index) > 1)
					predicted_label= 1 ;
				else 
					predicted_label= 0 ;
				end


			%Resubstitution error:				
				if(true_label ~= predicted_label)
					error_count4(j) = error_count4(j) + 1/60 ; 
				end 
			end
			
			
			if (error_count4(j)< best_error4)
				best_error4 = error_count4(j); 
				best_index= j; 
			end
			
		end
	end
	
	best_features4=[best_features4 best_index]; 
	Best_Feature_Set(14+m,:)= [best_features4 zeros(1, 8-m)]; 
	Resub_Error(14+m)= best_error4; 
	
	%True Error 
	for l= 1:235 
			true_label= test_data_label(l); 
			dist= ((train_data(:,best_features4)- (repmat(test_data(l,best_features4),60,1))).^2); 
			dist= sum(dist,2); 

			temp= sortrows([dist  train_data_label]); 
			neighbour_index= temp(1:3,2);

			if (sum(neighbour_index) > 1)
				assign_label= 1 ;
			else 
				assign_label= 0 ;
			end

			if(assign_label ~= test_data_label (l))
				True_Error(m+14) = True_Error(m+14) + 1/235; 
			end 
		end
	
	
end	
			
best_train4= train_data(:,best_features4)	;		


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Printing All 
disp('Feature Sets are-' ); 

for i=1:22
	disp(sprintf('Feature Set No. %d =' , i)); 
	
	for j=1:8 
		if(Best_Feature_Set(i,j)==0)
			last_index= j-1; 
			break; 
		
		elseif (j==8)
			last_index=j; 
			
		end		
	end 
	
	disp(Feature_Name(Best_Feature_Set(i,1:last_index))) ; 
	disp(sprintf('Resubstitution Error= %d', Resub_Error(i))); 
	disp(sprintf('True Error= %d', True_Error(i))); 
	disp(sprintf('\n \n'));
end 
	
	
	
	