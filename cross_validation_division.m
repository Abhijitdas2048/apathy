clear all;
%%%%%%%%%%%%%%%%%% Setting %%%%%%%%%%%%%%%%%%%%%
video_num = 90; %%% how many videos are there in the 
fold_num = 3;
feature_path = './feature_GAP/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileFolder = fullfile(feature_path);
dirOutput = dir(fullfile(fileFolder));
fileNames = {dirOutput.name}';

subject_id = [];
video_id = [];
for i = 3:length(fileNames)
    temp = load(strcat(feature_path, fileNames{i}, '/log.mat'));
    temp = temp.source_path;
    a = find(temp == '_');
    subject_id = [subject_id, str2num(temp(a(1)+1:a(2)-1))];
    video_id = [video_id, str2num(fileNames{i})];
end

final_count = zeros(max(subject_id),1);
final_idx = zeros(max(subject_id));

for i = 1:length(subject_id)
    final_count(subject_id(i)) = final_count(subject_id(i)) + 1;
    final_idx(subject_id(i),final_count(subject_id(i))) = video_id(i);
end

idx = find(final_count ~= 0);
final_idx = final_idx(idx,:);

subject = randperm(length(idx));
subject_num = length(subject);
fold_subject_num = floor(subject_num/fold_num);

for i = 1:fold_num
    begin_idx = max(1, (i-1)*fold_subject_num + 1);
    end_idx = min(i*fold_subject_num, subject_num);
    
    videos = [];
    for j = begin_idx:end_idx
        temp = final_idx(subject(j),:);
        temp = temp(temp~=0);
        
        videos = [videos, temp];
    end
    
    %videos = video_idx(begin_idx:end_idx);
    
    eval(['fold.fold', num2str(i), ' = videos;'])
end

save fold fold