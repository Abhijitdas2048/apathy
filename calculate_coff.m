%% calculate coff

path = './feature_dom/'; %%  feature dir path which you use for calculation emo/dom/AU_lip etc.
person_num = 72;

gt_file = './motap_apathy_YES-NO.xlsx'; %% gt dir path
% there are some changes in the table you just sent me,  So there may need
% some adjustments.
[label, video_list] = xlsread(gt_file); 

oui_pos = []; % features for oui and postive
oui_neg = [];
non_pos = [];
non_neg = [];

for p = 1:person_num
    person_dir = strcat(path, num2str(p), '/');
    
    if ~exist(person_dir)
        continue;
    end
    temp = load(strcat(person_dir, 'log.mat'));
    
    video_name = temp.video_name;
    
    [gt, pos] = get_gt_pos(video_name, video_list);
    
    fileFolder=fullfile(person_dir);
    dirOutput=dir(fullfile(fileFolder));
    fileNames={dirOutput.name}';
    
    if gt == 1 && pos == 1 
        for i = 3:length(fileNames)-2
            temp = load(strcat(person_dir, fileNames{i}));
            %             feature = temp.feature_all;
            feature = temp.feature_all.feature;
            oui_pos = [oui_pos; feature]; 
        end
    elseif gt == 1 && pos == 0 
        for i = 3:length(fileNames)-2
            temp = load(strcat(person_dir, fileNames{i}));
%             feature = temp.feature_all;
            feature = temp.feature_all.feature;
            oui_neg = [oui_neg; feature];
        end
    elseif gt == 0 && pos == 1
        for i = 3:length(fileNames)-2
            temp = load(strcat(person_dir, fileNames{i}));
            %             feature = temp.feature_all;
            feature = temp.feature_all.feature;
            non_pos = [non_pos; feature];
        end
    elseif gt == 0 && pos == 0
        for i = 3:length(fileNames)-2
            temp = load(strcat(person_dir, fileNames{i}));
            %             feature = temp.feature_all;
            feature = temp.feature_all.feature;
            non_neg = [non_neg; feature];
        end
    end
end

C_oui_pos = corrcoef(oui_pos);
C_oui_neg = corrcoef(oui_neg);
C_non_pos = corrcoef(non_pos);
C_non_neg = corrcoef(non_neg);