%%% test result
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Setting %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result_dir = './result/test_fold/';  %%% result path
max_epoch_num = 10000;  %%%% max epoch num of the training
thre = 0.5;  %%%% threshold for two class classification

fold_num = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx_all = [];
for fold = 1:fold_num
    
    Result_all = [];
    Gt_all = [];
    for i = 1:max_epoch_num
        result_path = strcat(result_dir, 'fold', num2str(fold), '_result', num2str(i), '.mat');
        
        if ~exist(result_path)
            break;
        end
        
        temp = load(result_path);
        
        result_final = temp.result;
        result_final = reshape(result_final, [2, length(result_final)/2]);
        gt_final = temp.gt;
        
        %result for a video;
        Predict = [];
        Gt = [];
        result_p = exp(result_final);
        result_p = result_p(2,:)./sum(result_p);
        for j = 1:length(gt_final)
            if (result_p(j) < thre)
                Predict = [Predict 0];
            else
                Predict = [Predict 1];
            end
            
            Gt = [Gt gt_final(j)];
        end
        
        cm = confusionmat(Gt, Predict);
        if(cm(2,2) == 0)
            f1f = 0;
            Result_all = [Result_all; 0 0 0];
        else
            p = cm(2,2) / sum(cm(:,2));
            r = cm(2,2) / sum(cm(2,:));
            f1f = 2*p*r / (p+r);
            Result_all = [Result_all; p r f1f];
        end
    end
    
    [a, idx] = max(Result_all(:,3));
    idx_all = [idx_all, idx];
    
    figure(fold)
    clf;
    plot(Result_all(:,3));
end
%% result evaluation
% idx = 160;

Predict = [];
Gt = [];
Result_p = [];
for f = 1:fold_num
    temp = load(strcat(result_dir, 'fold', num2str(f), '_result', num2str(idx_all(f)), '.mat'));
    
    result_final = temp.result;
    result_final = reshape(result_final, [2, length(result_final)/2]);
    gt_final = temp.gt;
    
    result_p = exp(result_final);
    result_p = result_p(2,:)./sum(result_p);
    Result_p = [Result_p, result_p];
    for j = 1:length(gt_final)
        if (result_p(j) < thre)
            Predict = [Predict 0];
        else
            Predict = [Predict 1];
        end
        
        Gt = [Gt gt_final(j)];
    end
end

cm = confusionmat(Gt, Predict);
if(cm(2,2) == 0)
    f1f = 0;
    Result_all = [Result_all; 0 0];
else
    p = cm(2,2) / sum(cm(:,2));
    r = cm(2,2) / sum(cm(2,:));
    f1f = 2*p*r / (p+r);
    Result_all = [Result_all; p r f1f];
end

cm