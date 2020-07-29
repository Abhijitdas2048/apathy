%% division for each video
clear;
% addpath('./utils');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Setting %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csv_dir = './OpenFace_CSV/';  % location of the csv files.
gt_file = './Clinical_Data.xlsx'; % location of the ground truth file.
target_dir_list = './video_devision/'; % target location of the generated feature

chip_img_num = 200; % how many frames are included in one clip
chip_num = 50; % how many clip num are there for each video
buff = 1; % how many frames are ignored at the begining and end 
          % in case of outliers frames when the doctors are setting the camera etc.
          
% Output: the begin and the end of each clip each row;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
fileFolder = fullfile(csv_dir);
dirOutput = dir(fullfile(fileFolder));
fileNames = {dirOutput.name}';

for v = 3:length(fileNames) 
    video_idx = v-2;

    source_path = fileNames{v};
    
    tardir = strcat(target_dir_list, num2str(video_idx), '/');
    
    fprintf('%s\n',source_path);
    %% whole feature
    x_all = xlsread(strcat(csv_dir,source_path));
%     x_all = x_all(x_all(:,5) == 1, :);
    
    if size(x_all,1) < 100
        fprintf('%s\n', 'too few frames for feature generation ....');
        continue;
    end
    
    if ~exist(tardir)
        mkdir(tardir);
    end

    %% HeadPose-Gaze-AU
    window_l = floor((size(x_all,1) - buff*2 - chip_img_num)/(chip_num));
    feature_idx = 1;
    
    division = [];
    for i = 1:chip_num
        division = [division; buff+(i-1)*window_l+1, buff+(i-1)*window_l+1+chip_img_num]; 
    end
    
    log_path = strcat(tardir, 'log');
    eval(['save ', log_path, ' source_path']);
    
    division_path = strcat(tardir, 'division');
    eval(['save ', division_path, ' division']);
end
