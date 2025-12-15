
<div align="center">
	<h1>Amity Neural Network</h1>
	<p><b>Universal JavaScript Neural Network & AI Library</b></p>
	<p>
		<img src="https://img.shields.io/badge/AI-JavaScript-blue" alt="AI JS">
		<img src="https://img.shields.io/badge/License-MIT-brightgreen" alt="MIT">
	</p>
</div>

---

## Overview

**Amity Neural Network** is a powerful, flexible, and easy-to-use JavaScript library for building, training, and deploying neural networks and AI models in the browser or Node.js. It supports a wide range of AI paradigms, including deep learning, reinforcement learning, NLP, computer vision, audio processing, GANs, transfer learning, and more. GPU acceleration (WebGL/WebGPU) and Web Workers are supported for high performance.

---

## Features

- Feedforward, convolutional, recurrent (LSTM/GRU) neural networks
- Text generation, image & audio classification, sequence-to-sequence (translation)
- GANs, transfer learning, ensemble methods, curriculum learning
- Attention mechanisms, advanced training (mixup, label smoothing)
- GPU acceleration (WebGL/WebGPU) & parallel training with Web Workers
- Built-in utilities for NLP, image, and audio processing
- Visualization and diagnostic tools
- Easy extensibility and modern API

---

## Installation

**Browser:**
```html
<script src="amity-neural-network.js"></script>
```

**Node.js:**
```js
const { NeuralNetwork, NetworkTemplates, NeuralNetworkUtils } = require('./amity-neural-network');
```

---

## Quick Start Example

```js
// Create a simple classifier
const nn = new NeuralNetwork(4, [8, 6], 3, {
	activation: 'relu',
	learningRate: 0.01
});

// Train on a batch
const inputs = [ [0.1, 0.2, 0.3, 0.4], ... ];
const targets = [ [1,0,0], [0,1,0], ... ];
for (let i = 0; i < 100; i++) {
	nn.trainBatch(inputs, targets);
}

// Predict
const prediction = nn.predict([0.1, 0.2, 0.3, 0.4]);
console.log(prediction);
```

---

## Example Use Cases

### 1. Text Generation (LSTM)
```js
const textGen = NetworkTemplates.textGenerator(5000, 128, 256, { useGPU: true, dropout: 0.3 });
textGen.buildVocabulary([ 'Hello world', 'Machine learning is amazing', ... ]);
await textGen.trainOnTextBatch(trainingTexts, targetVectors, { epochs: 50 });
const generated = textGen.generateText('Hello', 100, 0.8);
console.log(generated);
```

### 2. Image Classification (with GPU)
```js
const imgClassifier = NetworkTemplates.imageClassifier(28, 10, { useGPU: true, gpuBackend: 'webgpu' });
await imgClassifier.trainOnImageBatch(images, labels, { epochs: 50, augment: true });
const prediction = imgClassifier.predict(testImage);
```

### 3. Audio Classification
```js
const audioClassifier = NetworkTemplates.audioClassifier(13, 100, 10);
const history = await audioClassifier.trainOnAudioBatch(audioBuffers, labels, { epochs: 30, numMFCCs: 13 });
const mfccs = audioClassifier.extractMFCC(newAudioBuffer);
audioClassifier.resetRecurrentStates();
for (let frame of mfccs) {
	var output = audioClassifier.predict(frame);
}
console.log('Prediction:', output);
```

### 4. Transfer Learning
```js
const pretrained = new NeuralNetwork(784, [256, 128], 10);
pretrained.load(pretrainedModelData);
pretrained.freezeLayers([0, 1]);
pretrained.replaceOutputLayer(5, 'sigmoid');
const history = pretrained.trainBatch(newData, newLabels);
```

### 5. GAN Image Generation
```js
const generator = NetworkTemplates.imageGenerator(100, 28);
await generator.trainGAN(realImages, { epochs: 200, batchSize: 64 });
const latent = generator.sampleLatentVector(100);
const generatedImage = generator.generateImage(latent);
```

### 6. Sequence-to-Sequence (Translation)
```js
const seq2seq = NetworkTemplates.sequenceToSequence(10000, 8000, 128);
await seq2seq.trainSeq2Seq(sourceTexts, targetTexts, { epochs: 100 });
const translation = seq2seq.translateSequence('Hello world');
```

### 7. Reinforcement Learning (Q-Learning)
```js
const qNet = NetworkTemplates.qNetwork(8, 4);
// Store experience: state, action, reward, nextState, done
qNet.storeExperience(state, action, reward, nextState, done);
qNet.trainRL();
const action = qNet.selectAction(state);
```

### 8. Attention Mechanism
```js
const model = new NeuralNetwork(128, [256], 128);
const attentionLayer = model.createAttentionLayer(256, 8);
const attended = model.multiHeadAttention(attentionLayer, query, keys, values);
```

### 9. Convolutional Processing
```js
const convLayer = model.createConvLayer(3, 32, 3, 1, 1);
const feature = model.convolve2D(image, convLayer.filters[0][0]);
const pooled = model.maxPool2D(feature, 2, 2);
```

### 10. Visualization
```js
const canvas = document.getElementById('networkCanvas');
model.drawNetwork(canvas, { showLabels: true });
```

---

## API Reference

- **NeuralNetwork**: Core class for custom networks
- **NetworkTemplates**: Factory for common architectures (classifier, regressor, GAN, etc.)
- **NeuralNetworkUtils**: Utility functions (normalization, batching, accuracy, etc.)

See [amity-neural-network.js](amity-neural-network.js) for full API details and advanced usage.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- Inspired by modern deep learning frameworks
- Built for education, research, and creative AI projects

---

## Contributing

Pull requests, issues, and suggestions are welcome!
