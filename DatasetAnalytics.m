%% DATASET ANALYTICS
clear; clc; close all;

% 1. Load Dataset
datasetPath = fullfile(getenv('HOME'), 'Desktop', 'art_data');
if ~isfolder(datasetPath), error('Dataset folder not found!'); end

disp('Scanning dataset...');
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% PLOT 1: CLASS DISTRIBUTION (Pie Chart)
tbl = countEachLabel(imds);
labels = tbl.Label;
counts = tbl.Count;

figure('Name', 'Dataset Distribution', 'Color', 'w', 'Position', [100 100 800 600]);
p = pie(counts);
legend(string(labels), 'Location', 'eastoutside');
title({'Dataset Class Distribution', ['Total Images: ' num2str(sum(counts))]}, 'FontSize', 14);
colormap(jet(numel(labels)));


%% PLOT 2: MEAN IMAGE ANALYSIS (The "Ghost" Art)
disp('Computing Mean Images per Class...');
classes = categories(imds.Labels);
meanImages = [];

for i = 1:numel(classes)
    className = classes{i};
    subds = subset(imds, find(imds.Labels == className));
    numToAvg = min(50, numel(subds.Files));
    
    avgImg = zeros(224, 224, 3);
    for j = 1:numToAvg
        img = readimage(subds, j);
        img = imresize(img, [224 224]);
        if size(img,3) == 1, img = cat(3, img, img, img); end
        avgImg = avgImg + double(img);
    end
    avgImg = avgImg / numToAvg;
    meanImages{i} = uint8(avgImg);
end

figure('Name', 'Mean Artworks', 'Color', 'w', 'Position', [100 100 1000 600]);
montage(meanImages, 'Size', [3 5], 'ThumbnailSize', [224 224]);
title('Mean Image Analysis (Pixel Averaging per Style)');