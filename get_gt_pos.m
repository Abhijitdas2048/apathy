%Input: the video name or the csv file name & the video names in the ground truth file
%Output: the ground truth (oui 1 /non 0/ not found the sbuject -1 ) and is the senerios (postive 1 negetive 0)
function [gt, pos] = get_gt_pos(video_name,video_list)

temp = find(video_name == '_');
person_idx = str2num(video_name(temp(1)+1:temp(2)-1)); 

gt = -1;
for i = 1:size(video_list,1)
    video_idx = video_list(i,1);
    if person_idx == str2num(video_idx{1})
        label = video_list(i,2);
        if strcmp(label{1}, 'Non')
            gt = 0;
        else
            gt = 1;
        end
    end
end

if isempty(find(video_name == 'n'))
    pos = 1;
else
    pos = 0;
end

end