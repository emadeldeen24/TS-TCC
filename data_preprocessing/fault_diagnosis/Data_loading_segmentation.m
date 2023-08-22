%% Following This reference: 
    %Retraining Strategy-Based Domain Adaptio Network for Intelligent Fault
    %Diagnosis, IEEE Transaction on industrial Electronics
    % Case_study one:
        % In this case study we train on dataset and test on another, each
        % dataset comprised as follows:
        % Healthy: K001,K002,K003,K0004,K005
        % OR Damage: KA01, KA03,KA04, KA05, KA06,KA07,KA09, ,KA15,KA16, KA22
        % In Damage: KI01, KI03,KI05,KI07,KI08, KI14,KI16,KI17,KI19,KI21

    % Case_study two--> Artifical to Real:
        % This dataset train on artifical faults and test on real faults.
        % Training: 
            % Healthy: K002
            % OR:  KA01, KA05, KA07
            % In: KI01, KI05, KI07
        % Testing: 
            % Healthy: K001
            % OR: KA04, KA15, KA16, KA22, KA30
            % In: KI14, KI16, KI17, KI18, KI21
        % 
        % Health Condition:Artifical: K002, Real: K001,
        % OR Damage: KA01, KA03,KA04, KA05, KA06,KA07,KA09, ,KA15,KA16, KA22
        % In Damage: KI01, KI03,KI05,KI07,KI08, KI14,KI16,KI17,KI19,KI21
    


% Data loading and preparation 
numfiles = 20;
T=256000;
sample_len=5120;
delta=4046;
% following the state of the art technique 
ks_normal_train=["K001", "K002"]; 
ks_real=["KA04", "KA15", "KA16", "KA22", "KA30", "KI14", "KI16","KI17","KI18","KI21"];
ks_artificial =["K002","KA01","KA05","KA07", "KI01", "KI05", "KI07"]; 
for k1 = ks_artificial
    s=1;
    for k = 1:numfiles
        a=0;
        % load as string 4 working conditions
        WK_0 = sprintf('N09_M07_F10_%s_%d.mat', k1, k);
        WK_1 = sprintf('N15_M01_F10_%s_%d.mat', k1, k);
        WK_2 = sprintf('N15_M07_F04_%s_%d.mat', k1, k);
        WK_3 = sprintf('N15_M07_F10_%s_%d.mat', k1, k);
        % load names to indexing
        WK_0_name = sprintf('N09_M07_F10_%s_%d.Y(7).Data', k1, k);
        WK_1_name = sprintf('N15_M01_F10_%s_%d.Y(7).Data', k1, k);
        WK_2_name= sprintf('N15_M07_F04_%s_%d.Y(7).Data', k1, k);
        WK_3_name= sprintf('N15_M07_F10_%s_%d.Y(7).Data', k1, k);
        % data loading
        data_load_0{k} = load(WK_0);
        data_load_1{k} = load(WK_1);
        data_load_2{k} = load(WK_2);
        data_load_3{k} = load(WK_3);
        
        
        % extracting vibrational signals
        vib_0= eval(['data_load_0{k}.' WK_0_name]);
        vib_0=vib_0(:);
        vib_1= eval(['data_load_1{k}.' WK_1_name]);
        vib_1=vib_1(:);
        vib_2= eval(['data_load_2{k}.' WK_2_name]);
        vib_2=vib_2(:);
        vib_3= eval(['data_load_3{k}.' WK_3_name]);
        vib_3=vib_3(:);
        
        
        % limit the number of points to 256000
        nd_0=size(vib_0,1);
        nd_1=size(vib_1,1);
        nd_2=size(vib_2,1);
        nd_3=size(vib_3,1);
        
        %  T=256000;
        %%
        if size(vib_0)<T
            vib_0=[vib_0;zeros(T-nd_0,1)];
        else
            vib_0=vib_0(1:256000);
        end
        if size(vib_1)<T
            vib_1=[vib_1;zeros(T-nd_1,1)];
        else
            vib_1=vib_1(1:256000);
        end
        if size(vib_2)<T
            vib_2=[vib_2;zeros(T-nd_2,1)];
        else
            vib_2=vib_2(1:256000);
        end
        if size(vib_3)<T
            vib_3=[vib_3;zeros(T-nd_3,1)];
        else
            vib_3=vib_3(1:256000);
        end
        %%     % moving window to generate samples
        for i=1:delta:(256000-sample_len)
            a=a+1;
            Mat_0(k,a,:)=vib_0(i:i+sample_len-1);
            Mat_1(k,a,:)=vib_1(i:i+sample_len-1);
            Mat_2(k,a,:)=vib_2(i:i+sample_len-1);
            Mat_3(k,a,:)=vib_3(i:i+sample_len-1);
        end
        
    end
Varname=matlab.lang.makeValidName(strcat(k1));
A_5120L_artificial.(Varname)=reshape(Mat_0,[k*a,sample_len]);
B_5120L.real.(Varname)=reshape(Mat_1,[k*a,sample_len]);
C_5120L.real.(Varname)=reshape(Mat_2,[k*a,sample_len]);
D_5120L.real.(Varname)=reshape(Mat_3,[k*a,sample_len]);
end
% clearvars -except Dataset
