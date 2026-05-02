# ShuffleFAC: Ultra-Lightweight Ship-Radiated Sound Classification for Real-time Embedded Inference

## 📌 Overview

ShuffleFAC is an ultra-lightweight neural network designed for **ship-radiated sound classification** in resource-constrained environments.
It integrates **Frequency-Aware Convolution (FAC)** into an efficient backbone structure using channel shuffle and depthwise separable convolution.

This model is optimized for **real-time embedded deployment**, achieving high performance with minimal computational cost.

---

## 🧠 Key Features

* 🔹 Frequency-aware feature extraction
* 🔹 Lightweight architecture (low parameters & MACs)
* 🔹 Channel shuffle for efficiency
* 🔹 Suitable for embedded systems (e.g., Raspberry Pi)

---

## 🏗️ Model Architecture

ShuffleFAC combines:

* Depthwise separable convolution
* Pointwise group convolution
* Channel shuffle
* Frequency-Aware Convolution (FAC)

This design allows the model to capture **frequency-specific acoustic patterns** while maintaining efficiency.

---

## 📊 Performance

* **Dataset**: DeepShip
* **Macro F1-score**: ~71.45%
* **Parameters**: ~39K
* **MACs**: ~3.06M
* **Latency**: ~6 ms (Raspberry Pi)

Compared to baseline models:

* Higher performance
* ~9.7× smaller model size
* ~2.5× faster inference

---

## ⚙️ Installation

```bash
git clone https://github.com/KNU-LMAP/ShuffleFAC.git
cd ShuffleFAC

pip install -r requirements.txt
```

---

## 🚀 Usage

### Train & Evaluation

```bash
python main.py
```
---

## 📁 Project Structure

```bash
ShuffleFAC/
├── model/          # Model architecture
├── main.py         # Training & Evaluation script
├── utils/          # Utility functions
├── default.yaml    # Configurations
```

---

## 🔬 Research Background

Ship-radiated noise exhibits distinct **frequency-domain characteristics** depending on vessel types.
Traditional CNNs have limited sensitivity to such variations.

ShuffleFAC addresses this by explicitly modeling frequency information, improving classification performance under low-resource constraints.

---

## 📎 Paper

If you use this work, please cite:

```
@article{shufflefac2026,
  title={Ultra-Lightweight Ship-Radiated Sound Classification for Real-time Embedded Inference},
  author={Park, Sangwon et al.},
  year={2026}
}
```

---

## 🤝 Acknowledgements

This work is supported by KNU LMAP and related research programs.

---

## 📬 Contact

For questions or collaboration, please open an issue or contact the authors.
