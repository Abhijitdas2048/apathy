for each subjects, there is a folder containing one feature for the whole video (feature_all.mat), 150 features for the clips (feature1.mat - feature150.mat), one label file (label.mat, 1 for Oui and 0 for Non), and one log file (log.mat, indicating which video I used to compute the features).

The semantics of the 117-d feature:
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Action unit information(1-69)
1-51: amplitude, max intensit, and standard deviation of AU intensity for AU 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, and 45.
52-69: detected AU frequency for AU 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, and 45.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lip movement information (70-79)
70-73: amplitude and standard deviation for the mouth open vector (mean of up lip - mean of bottom lip)
74-79: amplitude and standard deviation for the mean of 3D mouth open vector (mean of up lip - mean of bottom lip)