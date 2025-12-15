// Neural Network Image Trainer Application
// Main application logic

class ImageTrainerApp {
    constructor() {
        this.images = [];
        this.network = null;
        this.labels = new Set();
        this.labelToIndex = {};
        this.indexToLabel = {};
        this.isTraining = false;
        this.generatorNetwork = null;
        this.categoryAverages = {};
        this.pendingFiles = null;
        
        this.initializeElements();
        this.initializeEventListeners();
        this.initializeTheme();
    }

    initializeElements() {
        // Upload elements
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.imageGrid = document.getElementById('imageGrid');
        
        // Training elements
        this.trainBtn = document.getElementById('trainBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.progressBar = document.getElementById('progressBar');
        this.progressFill = document.getElementById('progressFill');
        
        // AI Settings
        this.learningRate = document.getElementById('learningRate');
        this.epochs = document.getElementById('epochs');
        this.batchSize = document.getElementById('batchSize');
        this.optimizer = document.getElementById('optimizer');
        this.dropout = document.getElementById('dropout');
        this.activation = document.getElementById('activation');
        
        // Image Settings
        this.imageSize = document.getElementById('imageSize');
        this.colorMode = document.getElementById('colorMode');
        
        // Recognition elements
        this.recognizeUploadZone = document.getElementById('recognizeUploadZone');
        this.recognizeFileInput = document.getElementById('recognizeFileInput');
        this.recognitionResult = document.getElementById('recognitionResult');
        this.resultImage = document.getElementById('resultImage');
        this.confidenceBars = document.getElementById('confidenceBars');
        
        // Generation elements
        this.genWidth = document.getElementById('genWidth');
        this.genHeight = document.getElementById('genHeight');
        this.genCategory = document.getElementById('genCategory');
        this.genTemperature = document.getElementById('genTemperature');
        this.generateBtn = document.getElementById('generateBtn');
        this.generatedCanvas = document.getElementById('generatedCanvas');
        
        // Save/Load elements
        this.saveBtn = document.getElementById('saveBtn');
        this.loadBtn = document.getElementById('loadBtn');
        this.loadFileInput = document.getElementById('loadFileInput');
        
        // Status message
        this.statusMessage = document.getElementById('statusMessage');
        
        // Modals
        this.loadingModal = document.getElementById('loadingModal');
        this.bulkNameModal = document.getElementById('bulkNameModal');
        this.bulkNameInput = document.getElementById('bulkNameInput');
        this.bulkNameConfirm = document.getElementById('bulkNameConfirm');
        this.bulkNameCancel = document.getElementById('bulkNameCancel');
    }

    initializeEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Upload zone
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });
        
        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });
        
        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            this.handleFileSelect({ target: { files: e.dataTransfer.files } });
        });
        
        // Training
        this.trainBtn.addEventListener('click', () => this.trainNetwork());
        this.clearBtn.addEventListener('click', () => this.clearImages());
        
        // Recognition
        this.recognizeUploadZone.addEventListener('click', () => this.recognizeFileInput.click());
        this.recognizeFileInput.addEventListener('change', (e) => this.recognizeImage(e));
        
        // Generation
        this.generateBtn.addEventListener('click', () => this.generateImage());
        
        // Save/Load
        this.saveBtn.addEventListener('click', () => this.saveModel());
        this.loadBtn.addEventListener('click', () => this.loadFileInput.click());
        this.loadFileInput.addEventListener('change', (e) => this.loadModel(e));
        
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());
        
        // Bulk naming modal
        this.bulkNameConfirm.addEventListener('click', () => this.applyBulkName());
        this.bulkNameCancel.addEventListener('click', () => this.closeBulkNameModal());
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            document.getElementById('themeToggle').textContent = '☀️ Light Mode';
        }
    }

    toggleTheme() {
        const isLight = document.body.classList.toggle('light-theme');
        const themeToggle = document.getElementById('themeToggle');
        themeToggle.textContent = isLight ? '☀️ Light Mode' : '🌙 Dark Mode';
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    async handleFileSelect(event) {
        const files = Array.from(event.target.files);
        
        if (files.length === 0) return;
        
        // Show bulk naming modal if multiple files
        if (files.length > 3) {
            this.pendingFiles = files;
            this.showBulkNameModal();
            return;
        }
        
        await this.processFiles(files);
    }
    
    showBulkNameModal() {
        this.bulkNameInput.value = '';
        this.bulkNameModal.classList.add('active');
        this.bulkNameInput.focus();
    }
    
    closeBulkNameModal() {
        this.bulkNameModal.classList.remove('active');
        this.fileInput.value = '';
    }
    
    async applyBulkName() {
        const bulkLabel = this.bulkNameInput.value.trim();
        
        if (!bulkLabel) {
            this.showStatus('Please enter a label', 'error');
            return;
        }
        
        this.closeBulkNameModal();
        await this.processFiles(this.pendingFiles, bulkLabel);
        this.pendingFiles = null;
    }
    
    async processFiles(files, bulkLabel = null) {
        this.showLoadingModal();
        
        for (const file of files) {
            if (!file.type.startsWith('image/')) continue;
            
            const imageData = await this.loadImage(file);
            const label = bulkLabel || file.name.split('.')[0].toLowerCase();
            
            this.images.push({ data: imageData, label: label, file: file });
            
            // Parse multiple labels
            this.parseAndAddLabels(label);
            
            this.addImageToGrid(imageData, label, this.images.length - 1);
        }
        
        this.hideLoadingModal();
        this.updateUI();
        this.fileInput.value = '';
        
        this.showStatus(`${files.length} image(s) uploaded successfully`, 'success');
    }
    
    parseAndAddLabels(labelString) {
        // Split by comma, pipe, slash, or hyphen
        const separators = /[,|/\-]/;
        const labels = labelString.split(separators)
            .map(l => l.trim().toLowerCase())
            .filter(l => l.length > 0);
        
        labels.forEach(label => this.labels.add(label));
    }
    
    showLoadingModal() {
        this.loadingModal.classList.add('active');
    }
    
    hideLoadingModal() {
        this.loadingModal.classList.remove('active');
    }

    loadImage(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    resolve({ img, url: e.target.result });
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    addImageToGrid(imageData, label, index) {
        const div = document.createElement('div');
        div.className = 'image-item';
        div.innerHTML = `
            <img src="${imageData.url}" alt="${label}">
            <div class="image-label">
                <input type="text" value="${label}" data-index="${index}">
            </div>
            <button class="remove-btn" data-index="${index}">×</button>
        `;
        
        const input = div.querySelector('input');
        input.addEventListener('change', (e) => this.updateLabel(index, e.target.value));
        
        const removeBtn = div.querySelector('.remove-btn');
        removeBtn.addEventListener('click', () => this.removeImage(index));
        
        this.imageGrid.appendChild(div);
    }

    updateLabel(index, newLabel) {
        this.images[index].label = newLabel.toLowerCase();
        
        // Rebuild labels set from all images
        this.labels.clear();
        this.images.forEach(img => this.parseAndAddLabels(img.label));
        
        this.updateUI();
    }

    removeImage(index) {
        this.images.splice(index, 1);
        this.rebuildImageGrid();
        this.updateUI();
    }

    rebuildImageGrid() {
        this.imageGrid.innerHTML = '';
        this.labels.clear();
        
        this.images.forEach((img, index) => {
            this.addImageToGrid(img.data, img.label, index);
            this.parseAndAddLabels(img.label);
        });
    }

    clearImages() {
        if (this.images.length === 0) return;
        
        if (confirm('Are you sure you want to clear all images?')) {
            this.images = [];
            this.labels.clear();
            this.imageGrid.innerHTML = '';
            this.updateUI();
            this.showStatus('All images cleared', 'info');
        }
    }

    updateUI() {
        const hasImages = this.images.length > 0;
        const hasLabels = this.labels.size > 1;
        
        this.trainBtn.disabled = !hasImages || !hasLabels;
        this.generateBtn.disabled = !this.generatorNetwork;
        this.saveBtn.disabled = !this.network;
        
        // Update generation category dropdown
        this.updateCategoryDropdown();
        
        // Update button text
        if (hasImages && !hasLabels) {
            this.trainBtn.textContent = '⚠️ Need at least 2 different labels';
        } else {
            this.trainBtn.textContent = `🚀 Start Training (${this.images.length} images, ${this.labels.size} labels)`;
        }
    }

    updateCategoryDropdown() {
        this.genCategory.innerHTML = '';
        
        if (this.labels.size === 0) {
            this.genCategory.innerHTML = '<option>No categories available</option>';
            return;
        }
        
        Array.from(this.labels).sort().forEach(label => {
            const option = document.createElement('option');
            option.value = label;
            option.textContent = label.charAt(0).toUpperCase() + label.slice(1);
            this.genCategory.appendChild(option);
        });
    }

    async trainNetwork() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.trainBtn.disabled = true;
        
        // Show loading modal for data collection
        this.showLoadingModal();
        
        try {
            // Create label mappings
            this.createLabelMappings();
            
            // Prepare training data
            const { inputs, targets, processedImages } = await this.prepareTrainingData();
            
            // Close loading modal, show progress bar
            this.hideLoadingModal();
            this.progressBar.classList.add('active');
            
            // Create classifier network
            const imageSize = parseInt(this.imageSize.value);
            const inputSize = imageSize * imageSize;
            const numClasses = this.labels.size;
            
            this.network = new NeuralNetwork(
                inputSize,
                [512, 256, 128],
                numClasses,
                {
                    learningRate: parseFloat(this.learningRate.value),
                    activation: this.activation.value,
                    outputActivation: 'sigmoid',
                    optimizer: this.optimizer.value,
                    dropout: parseFloat(this.dropout.value)
                }
            );
            
            // Create generator network (reverse architecture)
            this.generatorNetwork = new NeuralNetwork(
                numClasses + 100, // category one-hot + noise vector
                [256, 512, 1024],
                inputSize,
                {
                    learningRate: parseFloat(this.learningRate.value) * 0.5,
                    activation: 'leaky_relu',
                    outputActivation: 'tanh',
                    optimizer: this.optimizer.value
                }
            );
            
            // Calculate category averages for generation
            this.calculateCategoryAverages(processedImages, targets);
            
            this.showStatus('Networks created. Training classifier...', 'info');
            
            // Train classifier network
            const epochs = parseInt(this.epochs.value);
            
            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;
                
                // Shuffle data
                const indices = Array.from({ length: inputs.length }, (_, i) => i);
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                
                // Train on shuffled data
                for (const idx of indices) {
                    const output = this.network.train(inputs[idx], targets[idx]);
                    totalLoss += this.calculateLoss(output, targets[idx]);
                }
                
                // Update progress (60% for classifier)
                const progress = ((epoch + 1) / epochs) * 60;
                this.updateProgress(progress);
                
                if (epoch % 5 === 0) {
                    const avgLoss = totalLoss / inputs.length;
                    console.log(`Classifier Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`);
                }
                
                await this.sleep(10);
            }
            
            this.showStatus('Training generator...', 'info');
            
            // Train generator network
            const genEpochs = Math.floor(epochs * 0.5);
            for (let epoch = 0; epoch < genEpochs; epoch++) {
                let totalGenLoss = 0;
                
                for (let i = 0; i < inputs.length; i++) {
                    // Create generator input (category + noise)
                    const noise = Array(100).fill(0).map(() => Math.random() * 2 - 1);
                    const genInput = [...targets[i], ...noise];
                    
                    // Train generator to recreate the image
                    const output = this.generatorNetwork.train(genInput, inputs[i]);
                    totalGenLoss += this.calculateLoss(output, inputs[i]);
                }
                
                // Update progress (40% for generator)
                const progress = 60 + ((epoch + 1) / genEpochs) * 40;
                this.updateProgress(progress);
                
                if (epoch % 5 === 0) {
                    const avgLoss = totalGenLoss / inputs.length;
                    console.log(`Generator Epoch ${epoch}: Loss = ${avgLoss.toFixed(4)}`);
                }
                
                await this.sleep(10);
            }
            
            this.showStatus('Training complete! Networks ready.', 'success');
            
        } catch (error) {
            console.error('Training error:', error);
            this.showStatus('Training failed: ' + error.message, 'error');
        } finally {
            this.isTraining = false;
            this.updateUI();
            setTimeout(() => {
                this.progressBar.classList.remove('active');
                this.updateProgress(0);
            }, 2000);
        }
    }

    calculateLoss(output, target) {
        let sum = 0;
        for (let i = 0; i < output.length; i++) {
            sum += Math.pow(target[i] - output[i], 2);
        }
        return sum / output.length;
    }

    createLabelMappings() {
        this.labelToIndex = {};
        this.indexToLabel = {};
        
        Array.from(this.labels).sort().forEach((label, index) => {
            this.labelToIndex[label] = index;
            this.indexToLabel[index] = label;
        });
    }

    async prepareTrainingData() {
        const inputs = [];
        const targets = [];
        const processedImages = [];
        const imageSize = parseInt(this.imageSize.value);
        
        for (const imgData of this.images) {
            const processedImage = await this.processImage(imgData.data.img, imageSize);
            
            // Parse all labels from the image label string
            const separators = /[,|/\-]/;
            const imgLabels = imgData.label.split(separators)
                .map(l => l.trim().toLowerCase())
                .filter(l => l.length > 0);
            
            // Create target with all labels activated
            const target = new Array(this.labels.size).fill(0);
            imgLabels.forEach(label => {
                if (this.labelToIndex[label] !== undefined) {
                    target[this.labelToIndex[label]] = 1;
                }
            });
            
            inputs.push(processedImage);
            targets.push(target);
            processedImages.push({ pixels: processedImage, labels: imgLabels });
        }
        
        return { inputs, targets, processedImages };
    }

    calculateCategoryAverages(processedImages, targets) {
        this.categoryAverages = {};
        const categoryCounts = {};
        
        // Initialize
        for (let label of this.labels) {
            this.categoryAverages[label] = new Array(processedImages[0].pixels.length).fill(0);
            categoryCounts[label] = 0;
        }
        
        // Sum up all images per category
        for (let i = 0; i < processedImages.length; i++) {
            const img = processedImages[i];
            for (let label of img.labels) {
                if (this.categoryAverages[label]) {
                    for (let j = 0; j < img.pixels.length; j++) {
                        this.categoryAverages[label][j] += img.pixels[j];
                    }
                    categoryCounts[label]++;
                }
            }
        }
        
        // Calculate averages
        for (let label of this.labels) {
            if (categoryCounts[label] > 0) {
                for (let j = 0; j < this.categoryAverages[label].length; j++) {
                    this.categoryAverages[label][j] /= categoryCounts[label];
                }
            }
        }
    }

    async processImage(img, targetSize) {
        const canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext('2d');
        
        // Draw and resize image
        ctx.drawImage(img, 0, 0, targetSize, targetSize);
        
        // Get image data
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
        const pixels = [];
        
        // Convert to grayscale and normalize to [-1, 1]
        for (let i = 0; i < imageData.data.length; i += 4) {
            const r = imageData.data[i];
            const g = imageData.data[i + 1];
            const b = imageData.data[i + 2];
            const gray = (r + g + b) / 3;
            pixels.push((gray / 255) * 2 - 1); // Normalize to [-1, 1]
        }
        
        return pixels;
    }

    createOneHot(index, size) {
        const arr = new Array(size).fill(0);
        arr[index] = 1;
        return arr;
    }

    async recognizeImage(event) {
        if (!this.network) {
            this.showStatus('Please train the network first!', 'error');
            return;
        }
        
        const file = event.target.files[0];
        if (!file) return;
        
        const imageData = await this.loadImage(file);
        const imageSize = parseInt(this.imageSize.value);
        const processedImage = await this.processImage(imageData.img, imageSize);
        
        // Make prediction
        this.network.setTrainingMode(false);
        const prediction = this.network.predict(processedImage);
        
        // Display results
        this.resultImage.src = imageData.url;
        this.recognitionResult.style.display = 'block';
        
        // Sort predictions by confidence
        const predictions = prediction.map((conf, idx) => ({
            label: this.indexToLabel[idx],
            confidence: conf
        })).sort((a, b) => b.confidence - a.confidence);
        
        // Display confidence bars
        this.confidenceBars.innerHTML = '';
        predictions.forEach(pred => {
            const percentage = (pred.confidence * 100).toFixed(1);
            const div = document.createElement('div');
            div.className = 'confidence-item';
            div.innerHTML = `
                <div class="confidence-label">
                    <span>${pred.label.charAt(0).toUpperCase() + pred.label.slice(1)}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${percentage}%">${percentage}%</div>
                </div>
            `;
            this.confidenceBars.appendChild(div);
        });
        
        this.recognizeFileInput.value = '';
    }

    async generateImage() {
        if (!this.generatorNetwork) {
            this.showStatus('Please train the network first!', 'error');
            return;
        }
        
        const width = parseInt(this.genWidth.value);
        const height = parseInt(this.genHeight.value);
        const category = this.genCategory.value;
        const temperature = parseFloat(this.genTemperature.value);
        
        this.showStatus('Generating image...', 'info');
        this.generateBtn.disabled = true;
        
        try {
            // Create category one-hot vector
            const categoryOneHot = new Array(this.labels.size).fill(0);
            categoryOneHot[this.labelToIndex[category]] = 1;
            
            // Create noise vector with temperature
            const noise = Array(100).fill(0).map(() => 
                (Math.random() * 2 - 1) * temperature
            );
            
            // Combine category and noise
            const genInput = [...categoryOneHot, ...noise];
            
            // Generate image
            this.generatorNetwork.setTrainingMode(false);
            let generated = this.generatorNetwork.predict(genInput);
            
            // Blend with category average for better results
            const categoryAvg = this.categoryAverages[category];
            if (categoryAvg) {
                const blendFactor = 0.3; // 30% average, 70% generated
                generated = generated.map((val, idx) => 
                    val * (1 - blendFactor) + categoryAvg[idx] * blendFactor
                );
            }
            
            // Convert from [-1, 1] to [0, 1]
            generated = generated.map(v => (v + 1) / 2);
            
            // Apply post-processing for better quality
            generated = this.enhanceGenerated(generated);
            
            // Draw to canvas
            this.drawGeneratedImage(generated, width, height);
            
            this.showStatus(`Image generated for category: ${category}`, 'success');
        } catch (error) {
            console.error('Generation error:', error);
            this.showStatus('Generation failed: ' + error.message, 'error');
        } finally {
            this.generateBtn.disabled = false;
        }
    }

    enhanceGenerated(pixels) {
        // Apply contrast enhancement
        const enhanced = [...pixels];
        
        // Find min and max
        let min = Math.min(...enhanced);
        let max = Math.max(...enhanced);
        
        // Stretch contrast
        const range = max - min;
        if (range > 0) {
            for (let i = 0; i < enhanced.length; i++) {
                enhanced[i] = (enhanced[i] - min) / range;
            }
        }
        
        // Apply slight sharpening by emphasizing edges
        const size = Math.sqrt(enhanced.length);
        const sharpened = [...enhanced];
        
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                const idx = y * size + x;
                const center = enhanced[idx];
                const avg = (
                    enhanced[(y-1)*size + x] +
                    enhanced[(y+1)*size + x] +
                    enhanced[y*size + (x-1)] +
                    enhanced[y*size + (x+1)]
                ) / 4;
                
                // Sharpen
                sharpened[idx] = center + (center - avg) * 0.5;
                sharpened[idx] = Math.max(0, Math.min(1, sharpened[idx]));
            }
        }
        
        return sharpened;
    }

    drawGeneratedImage(pixels, width, height) {
        const imageSize = Math.sqrt(pixels.length);
        
        this.generatedCanvas.width = width;
        this.generatedCanvas.height = height;
        this.generatedCanvas.style.display = 'block';
        const ctx = this.generatedCanvas.getContext('2d');
        
        // Create temporary canvas at original size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imageSize;
        tempCanvas.height = imageSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        const tempImageData = tempCtx.createImageData(imageSize, imageSize);
        
        // Fill temp canvas with generated pixels
        for (let i = 0; i < pixels.length; i++) {
            const value = Math.floor(pixels[i] * 255);
            const idx = i * 4;
            tempImageData.data[idx] = value;
            tempImageData.data[idx + 1] = value;
            tempImageData.data[idx + 2] = value;
            tempImageData.data[idx + 3] = 255;
        }
        
        tempCtx.putImageData(tempImageData, 0, 0);
        
        // Scale up to desired size with smoothing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(tempCanvas, 0, 0, imageSize, imageSize, 0, 0, width, height);
    }

    saveModel() {
        if (!this.network) return;
        
        try {
            const modelData = {
                network: this.network.save(),
                generatorNetwork: this.generatorNetwork ? this.generatorNetwork.save() : null,
                labelToIndex: this.labelToIndex,
                indexToLabel: this.indexToLabel,
                labels: Array.from(this.labels),
                categoryAverages: this.categoryAverages,
                imageSize: parseInt(this.imageSize.value),
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `neural-model-${Date.now()}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
            
            this.showStatus('Model saved successfully!', 'success');
        } catch (error) {
            console.error('Save error:', error);
            this.showStatus('Failed to save model: ' + error.message, 'error');
        }
    }

    async loadModel(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const text = await file.text();
            const modelData = JSON.parse(text);
            
            // Recreate classifier network
            const networkData = JSON.parse(modelData.network);
            const inputSize = networkData.layers[0].weights[0].length;
            const hiddenSizes = networkData.layers.slice(0, -1).map(l => l.weights.length);
            const outputSize = networkData.layers[networkData.layers.length - 1].weights.length;
            
            this.network = new NeuralNetwork(inputSize, hiddenSizes, outputSize);
            this.network.load(modelData.network);
            
            // Recreate generator network if available
            if (modelData.generatorNetwork) {
                const genData = JSON.parse(modelData.generatorNetwork);
                const genInputSize = genData.layers[0].weights[0].length;
                const genHiddenSizes = genData.layers.slice(0, -1).map(l => l.weights.length);
                const genOutputSize = genData.layers[genData.layers.length - 1].weights.length;
                
                this.generatorNetwork = new NeuralNetwork(genInputSize, genHiddenSizes, genOutputSize);
                this.generatorNetwork.load(modelData.generatorNetwork);
            }
            
            // Restore label mappings
            this.labelToIndex = modelData.labelToIndex;
            this.indexToLabel = modelData.indexToLabel;
            this.labels = new Set(modelData.labels);
            this.categoryAverages = modelData.categoryAverages || {};
            
            // Update image size
            if (modelData.imageSize) {
                this.imageSize.value = modelData.imageSize;
            }
            
            this.updateUI();
            this.showStatus(`Model loaded successfully! (${modelData.labels.length} categories)`, 'success');
            
            this.loadFileInput.value = '';
        } catch (error) {
            console.error('Load error:', error);
            this.showStatus('Failed to load model: ' + error.message, 'error');
        }
    }

    updateProgress(percent) {
        this.progressFill.style.width = percent + '%';
        this.progressFill.textContent = Math.round(percent) + '%';
    }

    showStatus(message, type) {
        this.statusMessage.textContent = message;
        this.statusMessage.className = 'status-message ' + type;
        
        if (type !== 'info') {
            setTimeout(() => {
                this.statusMessage.className = 'status-message';
            }, 5000);
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ImageTrainerApp();
});