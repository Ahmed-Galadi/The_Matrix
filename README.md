# Zero-Dependency Digit Recognizer

A browser-based neural network inference app powered by a model trained **completely from scratch in C++**. No machine learning frameworks, no external libraries, no magic—just raw matrix operations, backpropagation, and clean engineering.

---

## How the Magic Works

### 🔹 1. Pixel Preprocessing (Exact MNIST Pipeline)
When you draw on the canvas, your input undergoes the exact same transformation pipeline used during training:
1. **Bounding Box Detection** → Finds the outermost dark pixels (`gray < 200`)
2. **20% Padding** → Expands the crop slightly to preserve stroke context
3. **Square Crop & Center** → Maintains aspect ratio and centers the digit
4. **Bilinear Resize → 20×20** → Smooth downscaling using native canvas interpolation
5. **Center in 28×28** → Placed on a white background with `(4,4)` offset
6. **Normalization** → `pixel = 1.0 - (gray / 255.0)` → Ink becomes `1.0`, background `0.0`

### 🔹 2. Forward Pass (Pure Math)
The normalized 784-pixel array flows through a lightweight feedforward network:
```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```
- **Hidden Layer:** `Z = X·W₁ + B₁` → `A = ReLU(Z)`
- **Output Layer:** `Z = A·W₂ + B₂` → `A = Softmax(Z)`
- **Precision:** JavaScript `Number` (64-bit IEEE 754) matches C++ `double` exactly.
- **Probabilities:** Real-time animated progress bars update instantly on mouse/finger release, showing confidence for all 10 digits.

---

## Requirements

- A modern web browser (Chrome, Firefox, Safari, Edge)
- **A local HTTP server** (required due to browser security blocking `file://` fetch requests)
- The 4 trained parameter files (see below)

---

## Model Files & Training

The trained weights and biases are **not included in this repository**. You have two ways to get them:

### Option 1: Download Pre-trained Files
Download the following files and place them in the same folder as `index.html`:
```
hidden_weights.txt   (128 × 784)
hidden_biases.txt    (128 × 1)
output_weights.txt   (10 × 128)
output_biases.txt    (10 × 1)
```
 *[download link here]*

### Option 2: Train It Yourself (Pure C++, Zero Libraries)
The C++ training code is completely self-contained. It uses a custom `Matrix` class, manual backpropagation, and reads the official Kaggle MNIST CSV format.

1. Download `train.csv` and `test.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)
2. Place them alongside the C++ source files (`main.cpp`, `Matrix.hpp`, `HiddenLayer.hpp`, `helpers.h`)
3. Compile & run:
   ```bash
   g++ main.cpp -o digit_trainer -std=c++17 -O3
   ./digit_trainer
   ```
4. After ~200 epochs, the program automatically exports the 4 `.txt` files in the exact format the web app expects.

---

## Quick Start

1. Clone or download this repository
2. Place the 4 `.txt` weight/bias files in the root directory
3. Start a local server:
   ```bash
   python3 -m http.server 8000
   ```
4. Open your browser to: `http://localhost:8000`
5. Draw a digit. Lift your pen/finger. Watch the probabilities animate in real-time.

---

## Project Structure
```
digit-recognizer-web/
├── index.html          # Main UI layout
├── style.css           # Responsive styling & animated progress bars
├── script.js           # Canvas drawing, preprocessing, NN inference, auto-loading
├── hidden_weights.txt  # (User-provided) Hidden layer weights
├── hidden_biases.txt   # (User-provided) Hidden layer biases
├── output_weights.txt  # (User-provided) Output layer weights
└── output_biases.txt   # (User-provided) Output layer biases
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Loading Model...` never finishes | You opened `index.html` directly. Use a local server instead. |
| `Missing model files` error | Ensure all 4 `.txt` files are in the same directory and named exactly. |
| Predictions look inaccurate | Verify preprocessing matches MNIST. Draw centered digits with clear strokes. |
| Lag on mobile | `touch-action: none` is enabled. Use a modern mobile browser. |

---

## Notes & References

- **Training Pipeline:** Pure C++ (`main.cpp`, `Matrix.hpp`, `HiddenLayer.hpp`, `helpers.h`) → Custom matrix math, manual gradient descent, no third-party libraries.
- **Inference Engine:** Pure JavaScript → Matches the C++ forward pass and pixel preprocessing exactly.
- **Original Prototype:** The preprocessing logic and inference structure were first validated in `DrawPredict.java`. The web app strictly follows the same mathematical pipeline to guarantee identical results across platforms.
- **Dataset Architecture:** MNIST (28×28 grayscale handwritten digits)

---
*Built for speed, transparency, and dependency-free machine learning directly in the browser.*