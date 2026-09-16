# MNIST Classification from Scratch (NumPy)

A fully custom multilayer neural network built from scratch using only **NumPy** to classify handwritten digits from the classic Kaggle MNIST dataset (Digit Recognizer).

Instead of relying on high-level deep learning frameworks (PyTorch, TensorFlow/Keras), all forward and backward passes (backpropagation), loss functions, and optimization routines are implemented manually.

## What it does

* **Architecture**: 3-layer Multi-Layer Perceptron (MLP):
  * Dense (784 $\rightarrow$ 128) $\rightarrow$ ReLU
  * Dense (128 $\rightarrow$ 64) $\rightarrow$ ReLU
  * Dense (64 $\rightarrow$ 10) $\rightarrow$ Softmax
* **Custom Engine**:
  * Implements `DenseLayer`, `ReLu`, `Softmax`, and `CrossEntropyLoss` with full manual forward and backward (vectorized matrix calculus) passes.
  * Custom **Adam Optimizer** featuring first ($m$) and second ($v$) moment estimation with bias correction.
* **Data Processing**:
  * Normalizes pixel values ($[0, 255] \rightarrow [0.0, 1.0]$).
  * Encodes categorical labels into One-Hot vectors.
  * Splits dataset into training ($90\%$) and validation ($10\%$) sets.
* **Visualization**: Plots the cross-entropy loss curve over iterations to observe convergence.

## Results

| Metric | Details |
| :--- | :--- |
| **Model** | 3-Layer MLP (Scratch NumPy) |
| **Optimizer** | Adam ($lr=0.001$, $\beta_1=0.9$, $\beta_2=0.999$) |
| **Loss Function** | Categorical Cross-Entropy |
| **Validation Accuracy** | ~91% *(run output)* |

## How to run

1. Download `train.csv` and `test.csv` from the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place them in the project directory.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib jupyter