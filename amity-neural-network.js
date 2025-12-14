// ============================================================================
// Advanced Neural Network Library v2.1
// NEW: Web Workers for parallel training
// ============================================================================

class NeuralNetwork {
    constructor(inputSize, hiddenSizes, outputSize, options = {}) {
        this.layers = [];
        this.experienceBuffer = [];
        this.learningRate = options.learningRate || 0.01;
        this.activation = options.activation || 'relu';
        this.outputActivation = options.outputActivation || 'sigmoid';
        this.bufferSize = options.bufferSize || 10000;
        this.batchSize = options.batchSize || 32;
        this.gamma = options.gamma || 0.95;
        this.epsilon = options.epsilon || 0.1;

        // Advanced optimization options
        this.optimizer = options.optimizer || 'sgd';
        this.momentum = options.momentum || 0.9;
        this.beta1 = options.beta1 || 0.9;
        this.beta2 = options.beta2 || 0.999;
        this.adamEpsilon = 1e-8;
        this.timeStep = 0;

        // Regularization
        this.dropout = options.dropout || 0;
        this.l2Lambda = options.l2Lambda || 0;
        this.training = true;

        // Memory management
        this.maxMemoryMB = options.maxMemoryMB || 100;
        this.autoCleanup = options.autoCleanup !== false;

        // Web Workers configuration
        this.useWorkers = options.useWorkers || false;
        this.numWorkers = options.numWorkers || navigator.hardwareConcurrency || 4;
        this.workers = [];
        this.workerBusy = [];

        // Build network architecture
        const sizes = [inputSize, ...hiddenSizes, outputSize];
        for (let i = 0; i < sizes.length - 1; i++) {
            const isOutputLayer = (i === sizes.length - 2);
            this.layers.push({
                weights: this.initializeWeights(sizes[i + 1], sizes[i]),
                biases: this.randomArray(sizes[i + 1]),
                activations: [],
                inputs: [],
                activationType: isOutputLayer ? this.outputActivation : this.activation,

                velocityW: this.createZeroMatrix(sizes[i + 1], sizes[i]),
                velocityB: Array(sizes[i + 1]).fill(0),
                mW: this.createZeroMatrix(sizes[i + 1], sizes[i]),
                mB: Array(sizes[i + 1]).fill(0),
                vW: this.createZeroMatrix(sizes[i + 1], sizes[i]),
                vB: Array(sizes[i + 1]).fill(0),

                dropoutMask: []
            });
        }

        // Initialize workers if enabled
        if (this.useWorkers && typeof Worker !== 'undefined') {
            this.initializeWorkers();
        }
    }

    // ========================================================================
    // Web Workers Initialization
    // ========================================================================

    initializeWorkers() {
        const workerCode = `
            // Worker code for neural network training
            self.onmessage = function(e) {
                const { type, data } = e.data;
                
                if (type === 'train_batch') {
                    const results = trainBatchWorker(data);
                    self.postMessage({ type: 'train_complete', results });
                } else if (type === 'predict_batch') {
                    const results = predictBatchWorker(data);
                    self.postMessage({ type: 'predict_complete', results });
                }
            };
            
            function trainBatchWorker(data) {
                const { inputs, targets, weights, biases, config } = data;
                const results = [];
                
                for (let i = 0; i < inputs.length; i++) {
                    const gradients = computeGradients(
                        inputs[i], 
                        targets[i], 
                        weights, 
                        biases, 
                        config
                    );
                    results.push(gradients);
                }
                
                return results;
            }
            
            function predictBatchWorker(data) {
                const { inputs, weights, biases, config } = data;
                const results = [];
                
                for (let i = 0; i < inputs.length; i++) {
                    const output = forward(inputs[i], weights, biases, config);
                    results.push(output);
                }
                
                return results;
            }
            
            function activate(x, type) {
                switch(type) {
                    case 'sigmoid':
                        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
                    case 'tanh':
                        return Math.tanh(x);
                    case 'relu':
                        return Math.max(0, x);
                    case 'leaky_relu':
                        return x > 0 ? x : 0.01 * x;
                    case 'linear':
                        return x;
                    default:
                        return Math.max(0, x);
                }
            }
            
            function activateDerivative(preActivation, activation, type) {
                switch(type) {
                    case 'sigmoid':
                        return activation * (1 - activation);
                    case 'tanh':
                        return 1 - activation * activation;
                    case 'relu':
                        return preActivation > 0 ? 1 : 0;
                    case 'leaky_relu':
                        return preActivation > 0 ? 1 : 0.01;
                    case 'linear':
                        return 1;
                    default:
                        return preActivation > 0 ? 1 : 0;
                }
            }
            
            function forward(input, weights, biases, config) {
                let current = input;
                const layerOutputs = [];
                const preActivations = [];
                
                for (let i = 0; i < weights.length; i++) {
                    const output = [];
                    const preAct = [];
                    const activationType = i === weights.length - 1 ? 
                        config.outputActivation : config.activation;
                    
                    for (let j = 0; j < weights[i].length; j++) {
                        let sum = biases[i][j];
                        for (let k = 0; k < current.length; k++) {
                            sum += current[k] * weights[i][j][k];
                        }
                        preAct.push(sum);
                        output.push(activate(sum, activationType));
                    }
                    
                    layerOutputs.push(output);
                    preActivations.push(preAct);
                    current = output;
                }
                
                return { outputs: layerOutputs, preActivations };
            }
            
            function computeGradients(input, target, weights, biases, config) {
                const { outputs, preActivations } = forward(input, weights, biases, config);
                const gradients = [];
                
                // Calculate output layer gradients
                const lastLayer = outputs.length - 1;
                const outputGrads = [];
                for (let i = 0; i < outputs[lastLayer].length; i++) {
                    const error = target[i] - outputs[lastLayer][i];
                    const activationType = config.outputActivation;
                    const grad = error * activateDerivative(
                        preActivations[lastLayer][i],
                        outputs[lastLayer][i],
                        activationType
                    );
                    outputGrads.push(grad);
                }
                
                gradients.push(outputGrads);
                
                // Backpropagate through hidden layers
                let errors = outputGrads;
                for (let i = lastLayer - 1; i >= 0; i--) {
                    const layerGrads = [];
                    const newErrors = [];
                    
                    for (let j = 0; j < outputs[i].length; j++) {
                        let error = 0;
                        for (let k = 0; k < weights[i + 1].length; k++) {
                            error += weights[i + 1][k][j] * errors[k];
                        }
                        
                        const activationType = config.activation;
                        const grad = error * activateDerivative(
                            preActivations[i][j],
                            outputs[i][j],
                            activationType
                        );
                        layerGrads.push(grad);
                        newErrors.push(error);
                    }
                    
                    gradients.unshift(layerGrads);
                    errors = layerGrads;
                }
                
                return { gradients, outputs };
            }
        `;

        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const workerUrl = URL.createObjectURL(blob);

        for (let i = 0; i < this.numWorkers; i++) {
            this.workers.push(new Worker(workerUrl));
            this.workerBusy.push(false);
        }
    }

    terminateWorkers() {
        this.workers.forEach(w => w.terminate());
        this.workers = [];
        this.workerBusy = [];
    }

    // ========================================================================
    // Parallel Training Methods
    // ========================================================================

    async trainBatchParallel(batchInputs, batchTargets) {
        if (!this.useWorkers || this.workers.length === 0) {
            return this.trainBatch(batchInputs, batchTargets);
        }

        const batchesPerWorker = Math.ceil(batchInputs.length / this.numWorkers);
        const promises = [];

        for (let i = 0; i < this.numWorkers; i++) {
            const start = i * batchesPerWorker;
            const end = Math.min(start + batchesPerWorker, batchInputs.length);

            if (start >= batchInputs.length) break;

            const workerBatch = batchInputs.slice(start, end);
            const workerTargets = batchTargets.slice(start, end);

            promises.push(this.trainWorkerBatch(i, workerBatch, workerTargets));
        }

        const results = await Promise.all(promises);

        // Aggregate gradients and update weights
        let totalLoss = 0;
        for (const result of results) {
            totalLoss += result.loss;
            this.applyGradients(result.gradients);
        }

        return totalLoss / batchInputs.length;
    }

    trainWorkerBatch(workerIndex, inputs, targets) {
        return new Promise((resolve) => {
            const worker = this.workers[workerIndex];
            this.workerBusy[workerIndex] = true;

            const config = {
                activation: this.activation,
                outputActivation: this.outputActivation,
                learningRate: this.learningRate
            };

            const weights = this.layers.map(l => l.weights);
            const biases = this.layers.map(l => l.biases);

            worker.onmessage = (e) => {
                if (e.data.type === 'train_complete') {
                    this.workerBusy[workerIndex] = false;

                    const gradients = e.data.results;
                    let loss = 0;

                    for (let i = 0; i < gradients.length; i++) {
                        const outputs = gradients[i].outputs;
                        const lastOutput = outputs[outputs.length - 1];
                        for (let j = 0; j < lastOutput.length; j++) {
                            loss += Math.pow(targets[i][j] - lastOutput[j], 2);
                        }
                    }

                    resolve({ gradients, loss: loss / inputs.length });
                }
            };

            worker.postMessage({
                type: 'train_batch',
                data: { inputs, targets, weights, biases, config }
            });
        });
    }

    applyGradients(gradientBatch) {
        // Average and apply gradients from workers
        for (const item of gradientBatch) {
            const { gradients } = item;

            for (let i = 0; i < gradients.length; i++) {
                const layer = this.layers[i];
                const layerGrads = gradients[i];

                for (let j = 0; j < layerGrads.length; j++) {
                    layer.biases[j] += this.learningRate * layerGrads[j];

                    for (let k = 0; k < layer.weights[j].length; k++) {
                        const gradient = layerGrads[j] * layer.inputs[k];
                        layer.weights[j][k] += this.learningRate * gradient;
                    }
                }
            }
        }
    }

    async trainTurboParallel(batchInputs, batchTargets, options = {}) {
        const epochs = options.epochs || 100;
        const validationSplit = options.validationSplit || 0.1;
        const earlyStoppingPatience = options.patience || 10;
        const verbose = options.verbose || false;

        const splitIdx = Math.floor(batchInputs.length * (1 - validationSplit));
        const trainInputs = batchInputs.slice(0, splitIdx);
        const trainTargets = batchTargets.slice(0, splitIdx);
        const valInputs = batchInputs.slice(splitIdx);
        const valTargets = batchTargets.slice(splitIdx);

        let bestLoss = Infinity;
        let patience = 0;
        const history = { trainLoss: [], valLoss: [] };

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Parallel training
            const trainLoss = await this.trainBatchParallel(trainInputs, trainTargets);

            // Validation
            this.training = false;
            let valLoss = 0;
            for (let i = 0; i < valInputs.length; i++) {
                const outputs = this.predict(valInputs[i]);
                valLoss += NeuralNetworkUtils.mse(outputs, valTargets[i]);
            }
            valLoss /= valInputs.length;
            this.training = true;

            history.trainLoss.push(trainLoss);
            history.valLoss.push(valLoss);

            if (verbose && epoch % 10 === 0) {
                console.log(`Epoch ${epoch}: Train Loss = ${trainLoss.toFixed(4)}, Val Loss = ${valLoss.toFixed(4)}`);
            }

            if (valLoss < bestLoss) {
                bestLoss = valLoss;
                patience = 0;
            } else {
                patience++;
                if (patience >= earlyStoppingPatience) {
                    if (verbose) console.log(`Early stopping at epoch ${epoch}`);
                    break;
                }
            }

            if (this.autoCleanup && epoch % 50 === 0) {
                this.cleanupMemory();
            }
        }

        return history;
    }

    // ========================================================================
    // Original Methods (keeping all existing functionality)
    // ========================================================================

    initializeWeights(rows, cols) {
        const isReLU = this.activation === 'relu' || this.activation === 'leaky_relu';
        const limit = isReLU ? Math.sqrt(2 / cols) : Math.sqrt(6 / (rows + cols));
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() => (Math.random() * 2 - 1) * limit)
        );
    }

    randomArray(size) {
        return Array(size).fill(0).map(() => (Math.random() * 2 - 1) * 0.01);
    }

    createZeroMatrix(rows, cols) {
        return Array(rows).fill(0).map(() => Array(cols).fill(0));
    }

    activate(x, type) {
        switch (type) {
            case 'sigmoid':
                return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
            case 'tanh':
                return Math.tanh(x);
            case 'relu':
                return Math.max(0, x);
            case 'leaky_relu':
                return x > 0 ? x : 0.01 * x;
            case 'softplus':
                return Math.log(1 + Math.exp(Math.min(500, x)));
            case 'linear':
                return x;
            default:
                return Math.max(0, x);
        }
    }

    activateDerivative(preActivation, activation, type) {
        switch (type) {
            case 'sigmoid':
                return activation * (1 - activation);
            case 'tanh':
                return 1 - activation * activation;
            case 'relu':
                return preActivation > 0 ? 1 : 0;
            case 'leaky_relu':
                return preActivation > 0 ? 1 : 0.01;
            case 'softplus':
                return 1 / (1 + Math.exp(-preActivation));
            case 'linear':
                return 1;
            default:
                return preActivation > 0 ? 1 : 0;
        }
    }

    applyDropout(values, layer) {
        if (!this.training || this.dropout === 0) {
            return values;
        }

        layer.dropoutMask = values.map(() => Math.random() > this.dropout ? 1 : 0);
        return values.map((v, i) => v * layer.dropoutMask[i] / (1 - this.dropout));
    }

    predict(inputs) {
        let current = inputs;

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            layer.inputs = current;
            const output = [];
            const preActivations = [];

            for (let j = 0; j < layer.weights.length; j++) {
                let sum = layer.biases[j];
                for (let k = 0; k < current.length; k++) {
                    sum += current[k] * layer.weights[j][k];
                }
                preActivations.push(sum);
                output.push(this.activate(sum, layer.activationType));
            }

            layer.preActivations = preActivations;
            layer.activations = output;

            if (i < this.layers.length - 1) {
                current = this.applyDropout(output, layer);
            } else {
                current = output;
            }
        }

        return current;
    }

    train(inputs, targets) {
        this.training = true;
        const outputs = this.predict(inputs);
        let errors = targets.map((t, i) => t - outputs[i]);

        this.timeStep++;

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];

            const gradients = layer.activations.map((a, j) =>
                errors[j] * this.activateDerivative(layer.preActivations[j], a, layer.activationType)
            );

            this.updateWeights(layer, gradients, i);

            if (i > 0) {
                const prevLayer = this.layers[i - 1];
                const newErrors = Array(layer.inputs.length).fill(0);
                for (let k = 0; k < layer.inputs.length; k++) {
                    for (let j = 0; j < layer.weights.length; j++) {
                        newErrors[k] += layer.weights[j][k] * errors[j];
                    }
                    if (prevLayer.dropoutMask.length > 0 && prevLayer.dropoutMask[k] === 0) {
                        newErrors[k] = 0;
                    }
                }
                errors = newErrors;
            }
        }

        return outputs;
    }

    updateWeights(layer, gradients, layerIndex) {
        switch (this.optimizer) {
            case 'momentum':
                this.updateMomentum(layer, gradients);
                break;
            case 'adam':
                this.updateAdam(layer, gradients);
                break;
            default:
                this.updateSGD(layer, gradients);
        }
    }

    updateSGD(layer, gradients) {
        for (let j = 0; j < layer.weights.length; j++) {
            for (let k = 0; k < layer.weights[j].length; k++) {
                const gradient = gradients[j] * layer.inputs[k];
                const l2Penalty = this.l2Lambda * layer.weights[j][k];
                layer.weights[j][k] += this.learningRate * (gradient - l2Penalty);
            }
            layer.biases[j] += this.learningRate * gradients[j];
        }
    }

    updateMomentum(layer, gradients) {
        for (let j = 0; j < layer.weights.length; j++) {
            for (let k = 0; k < layer.weights[j].length; k++) {
                const gradient = gradients[j] * layer.inputs[k];
                const l2Penalty = this.l2Lambda * layer.weights[j][k];
                layer.velocityW[j][k] = this.momentum * layer.velocityW[j][k] +
                    this.learningRate * (gradient - l2Penalty);
                layer.weights[j][k] += layer.velocityW[j][k];
            }
            layer.velocityB[j] = this.momentum * layer.velocityB[j] +
                this.learningRate * gradients[j];
            layer.biases[j] += layer.velocityB[j];
        }
    }

    updateAdam(layer, gradients) {
        const lr_t = this.learningRate * Math.sqrt(1 - Math.pow(this.beta2, this.timeStep)) /
            (1 - Math.pow(this.beta1, this.timeStep));

        for (let j = 0; j < layer.weights.length; j++) {
            for (let k = 0; k < layer.weights[j].length; k++) {
                const gradient = gradients[j] * layer.inputs[k];
                const l2Penalty = this.l2Lambda * layer.weights[j][k];
                const g = gradient - l2Penalty;

                layer.mW[j][k] = this.beta1 * layer.mW[j][k] + (1 - this.beta1) * g;
                layer.vW[j][k] = this.beta2 * layer.vW[j][k] + (1 - this.beta2) * g * g;

                layer.weights[j][k] += lr_t * layer.mW[j][k] /
                    (Math.sqrt(layer.vW[j][k]) + this.adamEpsilon);
            }

            const g_b = gradients[j];
            layer.mB[j] = this.beta1 * layer.mB[j] + (1 - this.beta1) * g_b;
            layer.vB[j] = this.beta2 * layer.vB[j] + (1 - this.beta2) * g_b * g_b;
            layer.biases[j] += lr_t * layer.mB[j] / (Math.sqrt(layer.vB[j]) + this.adamEpsilon);
        }
    }

    trainBatch(batchInputs, batchTargets) {
        let totalLoss = 0;
        for (let i = 0; i < batchInputs.length; i++) {
            const outputs = this.train(batchInputs[i], batchTargets[i]);
            totalLoss += NeuralNetworkUtils.mse(outputs, batchTargets[i]);
        }

        if (this.autoCleanup && batchInputs.length > 100) {
            this.cleanupMemory();
        }

        return totalLoss / batchInputs.length;
    }

    trainTurbo(batchInputs, batchTargets, options = {}) {
        const epochs = options.epochs || 100;
        const validationSplit = options.validationSplit || 0.1;
        const earlyStoppingPatience = options.patience || 10;
        const verbose = options.verbose || false;

        const splitIdx = Math.floor(batchInputs.length * (1 - validationSplit));
        const trainInputs = batchInputs.slice(0, splitIdx);
        const trainTargets = batchTargets.slice(0, splitIdx);
        const valInputs = batchInputs.slice(splitIdx);
        const valTargets = batchTargets.slice(splitIdx);

        let bestLoss = Infinity;
        let patience = 0;
        const history = { trainLoss: [], valLoss: [] };

        for (let epoch = 0; epoch < epochs; epoch++) {
            let trainLoss = 0;
            for (let i = 0; i < trainInputs.length; i++) {
                const outputs = this.train(trainInputs[i], trainTargets[i]);
                trainLoss += NeuralNetworkUtils.mse(outputs, trainTargets[i]);
            }
            trainLoss /= trainInputs.length;

            this.training = false;
            let valLoss = 0;
            for (let i = 0; i < valInputs.length; i++) {
                const outputs = this.predict(valInputs[i]);
                valLoss += NeuralNetworkUtils.mse(outputs, valTargets[i]);
            }
            valLoss /= valInputs.length;
            this.training = true;

            history.trainLoss.push(trainLoss);
            history.valLoss.push(valLoss);

            if (verbose && epoch % 10 === 0) {
                console.log(`Epoch ${epoch}: Train Loss = ${trainLoss.toFixed(4)}, Val Loss = ${valLoss.toFixed(4)}`);
            }

            if (valLoss < bestLoss) {
                bestLoss = valLoss;
                patience = 0;
            } else {
                patience++;
                if (patience >= earlyStoppingPatience) {
                    if (verbose) console.log(`Early stopping at epoch ${epoch}`);
                    break;
                }
            }

            if (this.autoCleanup && epoch % 50 === 0) {
                this.cleanupMemory();
            }
        }

        return history;
    }

    // RL and other methods remain unchanged
    storeExperience(state, action, reward, nextState, done) {
        this.experienceBuffer.push({ state, action, reward, nextState, done });
        if (this.experienceBuffer.length > this.bufferSize) {
            this.experienceBuffer.shift();
        }
        if (this.autoCleanup) {
            this.checkMemoryUsage();
        }
    }

    trainRL(miniBatchSize = null) {
        const batchSize = miniBatchSize || this.batchSize;
        if (this.experienceBuffer.length < batchSize) return null;

        const batch = [];
        const indices = new Set();
        while (indices.size < batchSize) {
            indices.add(Math.floor(Math.random() * this.experienceBuffer.length));
        }
        indices.forEach(idx => batch.push(this.experienceBuffer[idx]));

        let totalLoss = 0;
        for (let exp of batch) {
            const { state, action, reward, nextState, done } = exp;
            const currentQ = this.predict(state);
            let target = reward;
            if (!done) {
                const nextQ = this.predict(nextState);
                target = reward + this.gamma * Math.max(...nextQ);
            }
            const targets = [...currentQ];
            targets[action] = target;
            this.train(state, targets);
            totalLoss += Math.pow(targets[action] - currentQ[action], 2);
        }
        return totalLoss / batchSize;
    }

    selectAction(state, explore = true) {
        if (explore && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.layers[this.layers.length - 1].weights.length);
        }
        const qValues = this.predict(state);
        return qValues.indexOf(Math.max(...qValues));
    }

    decayEpsilon(decayRate = 0.995, minEpsilon = 0.01) {
        this.epsilon = Math.max(minEpsilon, this.epsilon * decayRate);
    }

    // ========================================================================
    // Network Visualization
    // ========================================================================

    drawNetwork(canvas, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Options
        const nodeRadius = options.nodeRadius || 8;
        const nodeColor = options.nodeColor || '#4CAF50';
        const activeNodeColor = options.activeNodeColor || '#FF5722';
        const lineColor = options.lineColor || '#90A4AE';
        const activeLineColor = options.activeLineColor || '#FFC107';
        const lineWidth = options.lineWidth || 1;
        const activeLineWidth = options.activeLineWidth || 2;
        const showLabels = options.showLabels !== false;
        const fontSize = options.fontSize || 10;
        const activationThreshold = options.activationThreshold || 0.3;
        const weightThreshold = options.weightThreshold || 0.1;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Calculate layer sizes including input
        const layerSizes = [
            this.layers[0].weights[0].length, // input size
            ...this.layers.map(l => l.weights.length)
        ];

        // Calculate positions for each node
        const positions = [];
        const padding = 60;
        const usableWidth = width - 2 * padding;
        const usableHeight = height - 2 * padding;
        const layerSpacing = usableWidth / (layerSizes.length - 1);

        for (let i = 0; i < layerSizes.length; i++) {
            const layerPositions = [];
            const nodeCount = layerSizes[i];
            const maxNodes = Math.max(...layerSizes);
            const nodeSpacing = usableHeight / (Math.max(nodeCount, maxNodes) + 1);
            const layerHeight = nodeCount * nodeSpacing;
            const startY = (height - layerHeight) / 2 + nodeSpacing / 2;

            for (let j = 0; j < nodeCount; j++) {
                layerPositions.push({
                    x: padding + i * layerSpacing,
                    y: startY + j * nodeSpacing
                });
            }
            positions.push(layerPositions);
        }

        // Get activation values if available
        const activations = [];
        if (this.layers[0].inputs && this.layers[0].inputs.length > 0) {
            activations.push(this.layers[0].inputs);
            this.layers.forEach(layer => {
                if (layer.activations && layer.activations.length > 0) {
                    activations.push(layer.activations);
                }
            });
        }

        // Draw connections first (so they appear behind nodes)
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const fromPositions = positions[i];
            const toPositions = positions[i + 1];

            for (let j = 0; j < layer.weights.length; j++) {
                for (let k = 0; k < layer.weights[j].length; k++) {
                    const weight = layer.weights[j][k];
                    const from = fromPositions[k];
                    const to = toPositions[j];

                    // Determine if connection is active
                    const isActive = activations.length > 0 &&
                        Math.abs(activations[i][k]) > activationThreshold &&
                        Math.abs(weight) > weightThreshold;

                    // Draw line
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);

                    // Color based on weight strength and activation
                    if (isActive) {
                        const intensity = Math.min(1, Math.abs(activations[i][k]));
                        ctx.strokeStyle = activeLineColor;
                        ctx.lineWidth = activeLineWidth;
                        ctx.globalAlpha = 0.3 + intensity * 0.7;
                    } else {
                        const weightStrength = Math.min(1, Math.abs(weight));
                        ctx.strokeStyle = lineColor;
                        ctx.lineWidth = lineWidth;
                        ctx.globalAlpha = 0.1 + weightStrength * 0.2;
                    }

                    ctx.stroke();
                    ctx.globalAlpha = 1;
                }
            }
        }

        // Draw nodes
        for (let i = 0; i < positions.length; i++) {
            const layerPositions = positions[i];

            for (let j = 0; j < layerPositions.length; j++) {
                const pos = layerPositions[j];

                // Determine if node is active
                const activation = activations.length > i ? activations[i][j] : 0;
                const isActive = Math.abs(activation) > activationThreshold;

                // Draw node circle
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);

                if (isActive) {
                    const intensity = Math.min(1, Math.abs(activation));
                    ctx.fillStyle = activeNodeColor;
                    ctx.globalAlpha = 0.5 + intensity * 0.5;
                } else {
                    ctx.fillStyle = nodeColor;
                    ctx.globalAlpha = 0.6;
                }

                ctx.fill();
                ctx.globalAlpha = 1;

                // Draw node border
                ctx.strokeStyle = isActive ? activeNodeColor : '#37474F';
                ctx.lineWidth = isActive ? 2 : 1;
                ctx.stroke();
            }
        }

        // Draw layer labels
        if (showLabels) {
            ctx.fillStyle = '#263238';
            ctx.font = `${fontSize}px Arial`;
            ctx.textAlign = 'center';

            const labels = ['Input', ...Array(this.layers.length - 1).fill('Hidden'), 'Output'];

            for (let i = 0; i < positions.length; i++) {
                const x = positions[i][0].x;
                const y = padding / 2;
                ctx.fillText(labels[i], x, y);

                // Draw node count
                ctx.font = `${fontSize - 2}px Arial`;
                ctx.fillStyle = '#546E7A';
                ctx.fillText(`(${layerSizes[i]})`, x, y + fontSize + 2);
                ctx.font = `${fontSize}px Arial`;
                ctx.fillStyle = '#263238';
            }
        }
    }

    // Animate network with input
    /* 
        // Create and train network
        const nn = new NeuralNetwork(4, [8, 6], 3, {
            activation: 'relu',
            learningRate: 0.01
        });

        // Draw static network
        nn.drawNetwork(canvas, {
            nodeRadius: 10,
            nodeColor: '#4CAF50',
            activeNodeColor: '#FF5722',
            lineColor: '#90A4AE',
            activeLineColor: '#FFC107',
            showLabels: true
        });

        // After making a prediction, draw with activations highlighted
        const input = [0.5, 0.8, 0.2, 0.9];
        nn.predict(input);
        nn.drawNetwork(canvas, {
            activationThreshold: 0.3,  // Nodes above this value highlight
            weightThreshold: 0.1       // Connections above this show
        });

        // Or animate the forward pass
        nn.animateNetwork(canvas, input, {
            animate: true,
            duration: 1000,
            fps: 30
        });
    */
    animateNetwork(canvas, inputs, options = {}) {
        const fps = options.fps || 30;
        const duration = options.duration || 1000; // ms

        // Predict to get activations
        this.predict(inputs);

        // Draw initial frame
        this.drawNetwork(canvas, options);

        // Optional: Create animation by gradually "flowing" through network
        if (options.animate) {
            let frame = 0;
            const totalFrames = (duration / 1000) * fps;

            const animate = () => {
                if (frame < totalFrames) {
                    const progress = frame / totalFrames;
                    const threshold = options.activationThreshold || 0.3;

                    this.drawNetwork(canvas, {
                        ...options,
                        activationThreshold: threshold * (1 - progress)
                    });

                    frame++;
                    setTimeout(animate, 1000 / fps);
                }
            };

            animate();
        }
    }

    // Get network topology info
    getTopology() {
        const layerSizes = [
            this.layers[0].weights[0].length,
            ...this.layers.map(l => l.weights.length)
        ];

        return {
            layers: layerSizes.length,
            sizes: layerSizes,
            totalNodes: layerSizes.reduce((a, b) => a + b, 0),
            totalWeights: this.layers.reduce((sum, l) =>
                sum + l.weights.flat().length, 0)
        };
    }

    getMemoryUsageMB() {
        let totalSize = 0;
        for (let layer of this.layers) {
            totalSize += layer.weights.flat().length * 8;
            totalSize += layer.biases.length * 8;
            totalSize += (layer.velocityW.flat().length + layer.velocityB.length) * 8;
            totalSize += (layer.mW.flat().length + layer.mB.length) * 8;
            totalSize += (layer.vW.flat().length + layer.vB.length) * 8;
        }
        if (this.experienceBuffer.length > 0) {
            const sampleExp = this.experienceBuffer[0];
            const expSize = (sampleExp.state.length + sampleExp.nextState.length + 3) * 8;
            totalSize += this.experienceBuffer.length * expSize;
        }
        return totalSize / (1024 * 1024);
    }

    checkMemoryUsage() {
        const usage = this.getMemoryUsageMB();
        if (usage > this.maxMemoryMB) {
            console.warn(`Memory usage (${usage.toFixed(2)}MB) exceeds limit. Cleaning...`);
            this.cleanupMemory();
        }
        return usage;
    }

    cleanupMemory() {
        for (let layer of this.layers) {
            layer.activations = [];
            layer.inputs = [];
            layer.preActivations = [];
            layer.dropoutMask = [];
        }
        if (this.experienceBuffer.length > this.bufferSize * 0.8) {
            const removeCount = Math.floor(this.bufferSize * 0.2);
            this.experienceBuffer.splice(0, removeCount);
        }
        if (typeof global !== 'undefined' && global.gc) {
            global.gc();
        }
    }

    setTrainingMode(training) {
        this.training = training;
    }

    getExperienceCount() {
        return this.experienceBuffer.length;
    }

    clearExperience() {
        this.experienceBuffer = [];
    }

    setLearningRate(rate) {
        this.learningRate = rate;
    }

    setEpsilon(epsilon) {
        this.epsilon = epsilon;
    }

    getStats() {
        return {
            memoryUsageMB: this.getMemoryUsageMB(),
            experienceCount: this.experienceBuffer.length,
            timeStep: this.timeStep,
            epsilon: this.epsilon,
            learningRate: this.learningRate,
            optimizer: this.optimizer,
            workersEnabled: this.useWorkers,
            numWorkers: this.workers.length
        };
    }

    createCheckpoint() {
        return {
            data: this.save(),
            generation: this.generation || 0,
            score: this.lastScore || 0,
            timestamp: Date.now()
        };
    }
    
    loadCheckpoint(checkpoint) {
        this.load(checkpoint.data);
        return {
            generation: checkpoint.generation,
            score: checkpoint.score,
            timestamp: new Date(checkpoint.timestamp).toISOString()
        };
    }

    save() {
        return JSON.stringify({
            // Network architecture
            layers: this.layers.map(l => ({
                weights: l.weights,
                biases: l.biases,
                activationType: l.activationType,
                // CRITICAL: Save optimizer states
                velocityW: l.velocityW,
                velocityB: l.velocityB,
                mW: l.mW,
                mB: l.mB,
                vW: l.vW,
                vB: l.vB
            })),
            // Training state
            experienceBuffer: this.experienceBuffer,
            learningRate: this.learningRate,
            activation: this.activation,
            outputActivation: this.outputActivation,
            epsilon: this.epsilon,
            gamma: this.gamma,
            optimizer: this.optimizer,
            momentum: this.momentum,
            beta1: this.beta1,
            beta2: this.beta2,
            timeStep: this.timeStep,
            // Additional metadata
            version: '2.1',
            savedAt: new Date().toISOString()
        });
    }
    
    load(data) {
        const obj = JSON.parse(data);
        
        // Load network structure
        obj.layers.forEach((saved, i) => {
            this.layers[i].weights = saved.weights;
            this.layers[i].biases = saved.biases;
            this.layers[i].activationType = saved.activationType;
            
            // CRITICAL: Restore optimizer states
            if (saved.velocityW) this.layers[i].velocityW = saved.velocityW;
            if (saved.velocityB) this.layers[i].velocityB = saved.velocityB;
            if (saved.mW) this.layers[i].mW = saved.mW;
            if (saved.mB) this.layers[i].mB = saved.mB;
            if (saved.vW) this.layers[i].vW = saved.vW;
            if (saved.vB) this.layers[i].vB = saved.vB;
        });
        
        // Restore training state
        this.experienceBuffer = obj.experienceBuffer || [];
        this.learningRate = obj.learningRate || 0.01;
        this.activation = obj.activation || 'relu';
        this.outputActivation = obj.outputActivation || 'sigmoid';
        this.epsilon = obj.epsilon || 0.1;
        this.gamma = obj.gamma || 0.95;
        this.optimizer = obj.optimizer || 'sgd';
        this.momentum = obj.momentum || 0.9;
        this.beta1 = obj.beta1 || 0.9;
        this.beta2 = obj.beta2 || 0.999;
        this.timeStep = obj.timeStep || 0;
        
        console.log(`Loaded network from ${obj.savedAt || 'unknown time'}`);
        console.log(`Time step: ${this.timeStep}, Epsilon: ${this.epsilon.toFixed(4)}`);
    }
}

// ============================================================================
// Network Templates
// ============================================================================

class NetworkTemplates {
    static classifier(inputSize, numClasses, options = {}) {
        return new NeuralNetwork(
            inputSize,
            [64, 32],
            numClasses,
            {
                activation: 'relu',
                outputActivation: 'sigmoid',
                learningRate: 0.01,
                optimizer: 'adam',
                dropout: 0.2,
                ...options
            }
        );
    }

    static deepClassifier(inputSize, numClasses, options = {}) {
        return new NeuralNetwork(
            inputSize,
            [128, 64, 32],
            numClasses,
            {
                activation: 'relu',
                outputActivation: 'sigmoid',
                learningRate: 0.001,
                optimizer: 'adam',
                dropout: 0.3,
                ...options
            }
        );
    }

    static regressor(inputSize, outputSize, options = {}) {
        return new NeuralNetwork(
            inputSize,
            [64, 32],
            outputSize,
            {
                activation: 'relu',
                outputActivation: 'linear',
                learningRate: 0.01,
                optimizer: 'adam',
                ...options
            }
        );
    }

    static qNetwork(stateSize, actionSize, options = {}) {
        return new NeuralNetwork(
            stateSize,
            [128, 64],
            actionSize,
            {
                activation: 'relu',
                outputActivation: 'linear',
                learningRate: 0.001,
                optimizer: 'adam',
                epsilon: 0.1,
                gamma: 0.99,
                bufferSize: 10000,
                batchSize: 64,
                ...options
            }
        );
    }

    static autoencoder(inputSize, encodingSize, options = {}) {
        const hiddenSize = Math.floor((inputSize + encodingSize) / 2);
        return new NeuralNetwork(
            inputSize,
            [hiddenSize, encodingSize, hiddenSize],
            inputSize,
            {
                activation: 'tanh',
                outputActivation: 'sigmoid',
                learningRate: 0.01,
                optimizer: 'adam',
                ...options
            }
        );
    }

    static patternRecognizer(inputSize, options = {}) {
        return new NeuralNetwork(
            inputSize,
            [32, 16],
            2,
            {
                activation: 'leaky_relu',
                outputActivation: 'sigmoid',
                learningRate: 0.01,
                optimizer: 'momentum',
                ...options
            }
        );
    }

    static timeSeriesPredictor(windowSize, options = {}) {
        return new NeuralNetwork(
            windowSize,
            [64, 32, 16],
            1,
            {
                activation: 'tanh',
                outputActivation: 'linear',
                learningRate: 0.005,
                optimizer: 'adam',
                ...options
            }
        );
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

class NeuralNetworkUtils {
    static textToInput(text, maxLength = 10) {
        const normalized = text.toLowerCase().padEnd(maxLength, ' ').slice(0, maxLength);
        return Array.from(normalized).map(c => c.charCodeAt(0) / 127);
    }

    static normalize(values, min = null, max = null) {
        const actualMin = min !== null ? min : Math.min(...values);
        const actualMax = max !== null ? max : Math.max(...values);
        const range = actualMax - actualMin;
        return values.map(v => range === 0 ? 0 : (v - actualMin) / range);
    }

    static standardize(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        return values.map(v => std === 0 ? 0 : (v - mean) / std);
    }

    static mse(outputs, targets) {
        return outputs.reduce((sum, out, i) =>
            sum + Math.pow(targets[i] - out, 2), 0) / outputs.length;
    }

    static accuracy(predictions, targets) {
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            const predicted = predictions[i].indexOf(Math.max(...predictions[i]));
            const actual = targets[i].indexOf(Math.max(...targets[i]));
            if (predicted === actual) correct++;
        }
        return correct / predictions.length;
    }

    static oneHot(label, numClasses) {
        const encoded = Array(numClasses).fill(0);
        encoded[label] = 1;
        return encoded;
    }

    static trainTestSplit(data, labels, testSize = 0.2) {
        const indices = Array.from({ length: data.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const splitIndex = Math.floor(data.length * (1 - testSize));
        const trainIndices = indices.slice(0, splitIndex);
        const testIndices = indices.slice(splitIndex);

        return {
            trainData: trainIndices.map(i => data[i]),
            trainLabels: trainIndices.map(i => labels[i]),
            testData: testIndices.map(i => data[i]),
            testLabels: testIndices.map(i => labels[i])
        };
    }

    static createBatches(data, labels, batchSize) {
        const batches = [];
        for (let i = 0; i < data.length; i += batchSize) {
            batches.push({
                data: data.slice(i, i + batchSize),
                labels: labels.slice(i, i + batchSize)
            });
        }
        return batches;
    }
}

// ============================================================================
// Export
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralNetwork, NetworkTemplates, NeuralNetworkUtils };
}

if (typeof window !== 'undefined') {
    window.NeuralNetwork = NeuralNetwork;
    window.NetworkTemplates = NetworkTemplates;
    window.NeuralNetworkUtils = NeuralNetworkUtils;
}