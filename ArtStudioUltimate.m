function ArtStudioUltimate
    % ART STUDIO ULTIMATE - The Complete MATLAB Show
    % Features: Transfer Learning, Grad-CAM, Histogram, Edge Detection, Entropy, 3D Surface
    
    %% 1. MODEL YÜKLEME
    modelName = 'UltimateArtModel.mat'; 
    net = [];
    
    if exist(modelName, 'file')
        loadedData = load(modelName);
        if isfield(loadedData, 'trainedNet')
            net = loadedData.trainedNet;
        else
            uialert(uifigure, 'Model dosyası bozuk!', 'Hata');
            return;
        end
    else
        f = uifigure;
        uialert(f, 'Model dosyası bulunamadı!', 'Hata');
        return;
    end
    
    %% 2. ARAYÜZ TASARIMI
    fig = uifigure('Name', 'MATLAB Art Style Classifier Lab', ...
        'Position', [50 50 1100 650], ...
        'Color', [0.95 0.95 0.95]);

    % --- SOL PANEL ---
    pnlLeft = uipanel(fig, 'Title', 'Control Center', ...
        'Position', [20 20 280 610], ...
        'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');

    imAx = uiimage(pnlLeft, 'Position', [15 300 250 250]);
    imAx.ImageSource = '';
    imAx.ScaleMethod = 'fit';
    imAx.BackgroundColor = [0.9 0.9 0.9];

    lblPred = uilabel(pnlLeft, 'Text', 'Waiting for Input...', ...
        'Position', [10 240 260 40], ...
        'FontSize', 16, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'FontColor', [0.2 0.2 0.2]);

    btnLoad = uibutton(pnlLeft, 'push', ...
        'Text', '1. Load Art', ...
        'Position', [40 150 200 45], ...
        'BackgroundColor', [0 0.4470 0.7410], 'FontColor', 'white', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(btn,event) loadImage(net));

    btnAnalyze = uibutton(pnlLeft, 'push', ...
        'Text', '2.Run Full Analysis', ...
        'Position', [40 80 200 45], ...
        'BackgroundColor', [0.8500 0.3250 0.0980], 'FontColor', 'white', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Enable', 'off', ...
        'ButtonPushedFcn', @(btn,event) runDeepAnalysis(net));

    % --- SAĞ PANEL (SEKMELER) ---
    tabGroup = uitabgroup(fig, 'Position', [310 20 760 610]);

    % TAB 1: AI Insights
    tabProb = uitab(tabGroup, 'Title', 'AI Probability');
    axProb = uiaxes(tabProb, 'Position', [50 50 650 500]);
    title(axProb, 'Class Prediction Confidence');
    grid(axProb, 'on');

    % TAB 2: Grad-CAM
    tabExplain = uitab(tabGroup, 'Title', 'AI Focus (Grad-CAM)');
    axExplain = uiaxes(tabExplain, 'Position', [50 50 650 500]);
    axis(axExplain, 'off');
    title(axExplain, 'Where is the AI looking?');

    % TAB 3: Colors
    tabColor = uitab(tabGroup, 'Title', 'Color Histogram');
    axColor = uiaxes(tabColor, 'Position', [50 50 650 500]);
    title(axColor, 'RGB Distribution');

    
    % TAB 4: Structural Analysis (Canny Edge)
    tabEdge = uitab(tabGroup, 'Title', 'Structure (Edges)');
    axEdge = uiaxes(tabEdge, 'Position', [50 50 650 500]);
    axis(axEdge, 'off');
    title(axEdge, 'Artist''s Sketch Lines (Canny Filter)');

    % TAB 5: Texture Analysis (Entropy)
    tabTexture = uitab(tabGroup, 'Title', 'Texture (Entropy)');
    axTexture = uiaxes(tabTexture, 'Position', [50 50 650 500]);
    axis(axTexture, 'off');
    title(axTexture, 'Complexity & Brushwork Map');

    % TAB 6: 3D Topography
    tab3D = uitab(tabGroup, 'Title', '3D Light Map');
    ax3D = uiaxes(tab3D, 'Position', [50 50 650 500]);
    title(ax3D, '3D Pixel Intensity Topography');
    grid(ax3D, 'on');

    % Değişkenler
    currentImg = [];
    currentLabel = '';

    %% FONKSİYONLAR
    function loadImage(net)
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files'});
        if isequal(file, 0), return; end
        
        fullPath = fullfile(path, file);
        currentImg = imread(fullPath);
        imAx.ImageSource = fullPath;
        
        % Hızlı Tahmin
        inputSize = net.Layers(1).InputSize(1:2);
        imgResized = imresize(currentImg, inputSize);
        [label, scores] = classify(net, imgResized);
        
        currentLabel = char(label);
        lblPred.Text = currentLabel;
        lblPred.FontColor = [0 0.5 0];
        
        btnAnalyze.Enable = 'on';
        
        % TAB 1: Probabilities
        [sortedScores, sortedIdx] = sort(scores, 'descend');
        top5Scores = sortedScores(1:5) * 100;
        try
             classes = net.Layers(end).Classes;
        catch
             classes = categories(label); 
        end
        top5Labels = classes(sortedIdx(1:5));
        
        barh(axProb, top5Scores, 'FaceColor', [0 0.4470 0.7410]);
        yticklabels(axProb, top5Labels);
        xlabel(axProb, 'Confidence (%)');
        title(axProb, ['Prediction: ' currentLabel]);
        
        tabGroup.SelectedTab = tabProb;
    end

    function runDeepAnalysis(net)
        btnAnalyze.Text = 'Processing...';
        btnAnalyze.Enable = 'off';
        drawnow;
        
        inputSize = net.Layers(1).InputSize(1:2);
        imgResized = imresize(currentImg, inputSize);
        imgGray = rgb2gray(imgResized); % Gri tonlama (Analizler için)
        
        % --- TAB 2: Grad-CAM ---
        try
            layerName = 'res5b_relu'; 
            lgraph = layerGraph(net);
            layers = {lgraph.Layers.Name};
            if ~any(strcmp(layers, layerName))
                idx = find(contains(layers, 'relu'), 1, 'last');
                layerName = layers{idx};
            end
            scoreMap = gradCAM(net, imgResized, categorical(cellstr(currentLabel)), 'FeatureLayer', layerName);
            
            imshow(currentImg, 'Parent', axExplain);
            hold(axExplain, 'on');
            bigMap = imresize(scoreMap, size(currentImg, [1 2]));
            imagesc(axExplain, bigMap, 'AlphaData', 0.5);
            colormap(axExplain, 'jet');
            hold(axExplain, 'off');
        catch
            title(axExplain, 'Grad-CAM Unavailable');
        end
        
        % --- TAB 3: RGB Histogram ---
        cla(axColor); hold(axColor, 'on');
        histogram(axColor, currentImg(:,:,1), 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        histogram(axColor, currentImg(:,:,2), 'FaceColor', 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        histogram(axColor, currentImg(:,:,3), 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        hold(axColor, 'off');
        
        % --- TAB 4: Edge Detection (Canny) ---
        
        edges = edge(imgGray, 'Canny');
        imshow(edges, 'Parent', axEdge);
        title(axEdge, 'Structural Edges (Canny Filter)');
        
        % --- TAB 5: Texture Analysis (Entropy) ---
        % Dokunun karmaşıklığını ölçer (Pürüzlü yüzeyler vs Düz yüzeyler)
        ent = entropyfilt(imgGray);
        imagesc(axTexture, ent);
        colormap(axTexture, 'parula');
        colorbar(axTexture);
        axis(axTexture, 'off');
        title(axTexture, 'Texture Entropy Map');
        
        % --- TAB 6: 3D Surface Plot ---
        % Resim3D topografya 
        
        smallImg = imresize(imgGray, 0.25); 
        surf(ax3D, double(smallImg));
        shading(ax3D, 'interp');
        colormap(ax3D, 'jet');
        view(ax3D, [-45 60]); 
        title(ax3D, '3D Light Intensity Map');
        zlabel(ax3D, 'Intensity');
        
        btnAnalyze.Text = 'Run Full Analysis';
        btnAnalyze.Enable = 'on';
        

    end
end