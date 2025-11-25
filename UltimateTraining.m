

%% 1. VERİ YÖNETİMİ VE HAZIRLIK
disp('Tüm veri seti taranıyor... (Bu işlem biraz sürebilir)');
datasetPath = fullfile(getenv('HOME'), 'Desktop', 'art_data');

if ~isfolder(datasetPath)
    error('Klasör bulunamadı! Masaüstünde "art_data" olduğundan emin olun.');
end

allImages = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');


% %70 Eğitim, %15 Doğrulama, %15 Test
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(allImages, 0.7, 0.15, 'randomized');

disp(['Toplam Resim Sayısı: ', num2str(numel(allImages.Files))]);
disp(['Eğitim Seti: ', num2str(numel(imdsTrain.Files))]);
disp(['Doğrulama Seti: ', num2str(numel(imdsValidation.Files))]);

%% 2. AĞ MİMARİSİ (ResNet-18)
disp('Ağ mimarisi hazırlanıyor...');
net = resnet18;
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
    
lgraph = replaceLayer(lgraph, 'fc1000', newLearnableLayer);
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

%% 3. VERİ ÇOĞALTMA (DATA AUGMENTATION) 
% MATLAB Image Processing Toolbox kullanarak veriyi yapay olarak zenginleştiriyoruz.


imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...      % Resmi yatay çevir (Ayna etkisi)
    'RandXTranslation', [-10 10], ... % Sağa sola hafif kaydır
    'RandYTranslation', [-10 10]);    % Yukarı aşağı hafif kaydır

% Eğitim verisine Augmenter uyguluyoruz
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb'); % Siyah-beyaz hatasını önle

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

%% 4. EĞİTİM AYARLARI 
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...        
    'MaxEpochs', 8, ...             % Tüm veriyi 8 kere dönecek 
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 50, ...  % Her 50 adımda bir test et
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 5. EĞİTİMİ BAŞLAT
disp('Büyük eğitim başlıyor! Bilgisayarınızın fanları çalışabilir :)');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% Final modelini kaydediyoruz 
save('UltimateArtModel.mat', 'trainedNet');
disp('Eğitim tamamlandı! Model "UltimateArtModel.mat" olarak kaydedildi.');