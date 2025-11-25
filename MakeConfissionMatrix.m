%% CONFUSION MATRIX GENERATOR
clear; clc; close all;

% Model
modelName = 'UltimateArtModel.mat';
if exist(modelName, 'file')
    disp('Model yükleniyor...');
    load(modelName);
else
    error('UltimateArtModel.mat dosyası bulunamadı!');
end



datasetPath = fullfile(getenv('HOME'), 'Desktop', 'art_data');
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

disp('Test verisi ayrılıyor...');
% Her sınıftan rastgele 50 resim 
[~, imdsTest] = splitEachLabel(imds, 50, 'randomized');

% Ön İşleme (Resize & Renk Düzeltme)

inputSize = trainedNet.Layers(1).InputSize(1:2);

% Siyah-beyaz resimlerin hata vermemesi için 'gray2rgb' kullanıyoruz
augimdsTest = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');


disp('Tahminler yapılıyor ');
[predictedLabels, scores] = classify(trainedNet, augimdsTest);
trueLabels = imdsTest.Labels;

figure('Name', 'Confusion Matrix Analysis', 'Color', 'w', 'Position', [100 100 1000 800]);

% Grafik Oluşturma
cm = confusionchart(trueLabels, predictedLabels);

% Görsel Ayarlar
cm.Title = 'Art Style Classification Performance';
cm.RowSummary = 'row-normalized'; % Yanda doğruluk yüzdelerini göster
cm.ColumnSummary = 'column-normalized'; % Altta hassasiyeti göster
sortClasses(cm, 'descending-diagonal'); % En başarılıları başa al
cm.FontName = 'Helvetica'; % Şık yazı tipi



