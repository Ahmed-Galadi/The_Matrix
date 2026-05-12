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
Download the following files and place them in the `Datasets/` folder:
```
Datasets/hidden_weights.txt   (128 × 784)
Datasets/hidden_biases.txt    (128 × 1)
Datasets/output_weights.txt   (10 × 128)
Datasets/output_biases.txt    (10 × 1)
```
 You can download all four model files from the [**latest release**](https://github.com/Ahmed-Galadi/The_Matrix/releases/latest).
### Option 2: Train It Yourself (Pure C++, Zero Libraries)
The C++ training code is completely self-contained. It uses a custom `Matrix` class, manual backpropagation, and reads the official Kaggle MNIST CSV format.

1. Download `train.csv` and `test.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data)
2. Place them inside the `Train+Finetune/` directory.
3. Compile & run from within the `Train+Finetune/` directory:
   ```bash
   g++ train.cpp HiddenLayer.cpp Matrix.cpp helpers.cpp -o digit_trainer -std=c++17 -O3
   ./digit_trainer
   ```
4. After ~200 epochs, the program automatically exports the 4 `.txt` files into the `Datasets/` folder in the exact format the web app expects.

---

### Option 3: Add Your Own Samples & Fine-Tune

You can improve the model by adding your own handwritten digits and fine-tuning it on the new data.

#### Part 1: Add Handwritten Samples

The `Hand-writing-input/` directory contains a Java Swing application that lets you draw digits and save them in the correct format.

1.  **Compile and Run the Java App:**
    Open a terminal in the root of the project and run:
    ```bash
    # Compile the Java source file
    javac Hand-writing-input/DrawPredict.java

    # Run the application (make sure you are in the project's root directory)
    java -cp . Hand-writing-input/DrawPredict
    ```
    *Note: You must have a JDK (Java Development Kit) installed.*

2.  **Draw and Save:**
    - A window will appear. Draw a single digit in the black box.
    - Enter the correct digit (0-9) in the "Label" text field at the bottom.
    - Click the "Save" button.
    - Your drawing will be processed and appended as a new row to `Datasets/mydata.csv`.
    - Click "Clear" and repeat for as many new samples as you want.

#### Part 2: Fine-Tune the Model

**Important:** For the fine-tuning process to be effective, it is highly recommended to add at least **50 new samples for each digit (0 through 9)** to the `mydata.csv` file. A smaller number of samples may not be sufficient to meaningfully improve the model.

Once you have added your custom samples to `mydata.csv`, you can fine-tune the existing model using the `fine_tune.cpp` program. This will update the model's weights and biases based on your new data.

1.  **Compile the C++ Fine-Tuning Code:**
    Navigate to the `Train+Finetune/` directory and compile the `fine_tune.cpp` file:
    ```bash
    cd Train+Finetune
    g++ fine_tune.cpp Matrix.cpp HiddenLayer.cpp helpers.cpp -o fine_tuner -std=c++17 -O3
    ```

2.  **Run the Fine-Tuner:**
    From the `Train+Finetune/` directory, execute the compiled program:
    ```bash
    ./fine_tuner
    ```
    The program will load the existing model weights from the `Datasets/` folder, train for a few epochs on your new data in `Datasets/mydata.csv`, and automatically save the updated weights back to the `Datasets/` folder.

3.  **Test Your Changes:**
    Go back to the root directory and restart your local web server to see the predictions from your newly fine-tuned model.
    ```bash
    cd ..
    python3 -m http.server 8000
    ```

---

## Quick Start

1. Clone or download this repository
2. Place the 4 `.txt` weight/bias files in the `Datasets/` directory
3. Start a local server from the root of the project:
   ```bash
   python3 -m http.server 8000
   ```
4. Open your browser to: `http://localhost:8000`
5. Draw a digit. Lift your pen/finger. Watch the probabilities animate in real-time.

---

## Project Structure
```
The_Matrix/
├── index.html              # Main UI layout
├── script.js               # Canvas drawing, preprocessing, NN inference
├── style.css               # Responsive styling & animated progress bars
├── Datasets/
│   ├── hidden_weights.txt  # (User-provided or trained)
│   ├── hidden_biases.txt   # (User-provided or trained)
│   ├── output_weights.txt  # (User-provided or trained)
│   └── output_biases.txt   # (User-provided or trained)
└── Train+Finetune/
    ├── train.cpp           # Main training logic
    ├── Matrix.hpp          # Custom Matrix class header
    ├── Matrix.cpp          # Custom Matrix class implementation
    ├── HiddenLayer.hpp     # HiddenLayer class header
    ├── HiddenLayer.cpp     # HiddenLayer class implementation
    ├── helpers.h           # Helper functions header
    └── helpers.cpp         # Helper functions implementation
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Loading Model...` never finishes | You opened `index.html` directly. Use a local server instead. |
| `Missing model files` error | Ensure all 4 `.txt` files are in the `Datasets/` directory and named exactly. |
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