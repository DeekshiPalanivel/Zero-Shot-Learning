# 🐾 Zero-Shot Learning on Oxford-IIIT Pet Dataset

This project demonstrates a **Zero-Shot Learning (ZSL)** pipeline using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) by leveraging **ResNet18** feature extraction and **Sentence-BERT** embeddings of class labels. The goal is to classify **unseen classes** by learning a mapping from image features to semantic embeddings without training on those specific categories.

---

## 📂 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [License](#license)

---

## 🧠 Overview

Zero-shot learning enables classification of classes **not seen during training**. Instead of learning class-specific models, it learns a semantic space (using text embeddings of class names) and maps image features to this space.

---

## 🐶 Dataset

- **Oxford-IIIT Pet Dataset** contains 37 pet breeds.
- Each image has a category label representing the breed.
- The dataset is split into **Seen Classes** (60%) for training and **Unseen Classes** (40%) for testing.

---

## 🔍 Approach

1. **Feature Extraction** using pre-trained ResNet18 on input images.
2. **Class Embedding** using `sentence-transformers` to encode class names (seen and unseen).
3. **Linear Mapping Model** is trained to learn a transformation from image features to the semantic space.
4. **Evaluation** is done on unseen classes via cosine similarity matching in the embedding space.

---

## 🧱 Model Architecture

- **Feature Extractor:** ResNet18 (pre-trained, last FC layer removed).
- **Mapper Model:** Linear layer that maps 512-dim image features → 384-dim sentence embedding space.

---

## ⚙️ Training Details

- **Loss**: MSE Loss between predicted and actual sentence embeddings.
- **Optimizer**: Adam (`lr=1e-4`)
- **Epochs**: 20
- **Batch Size**: 32

---

## 🧪 Evaluation

- Evaluated using top-1 accuracy on the **unseen classes only**.
- Cosine similarity is used to compare predicted embeddings with unseen class embeddings.

### 🔎 Sample Output

✅ True: American Bulldog | 🔮 Predicted: Bombay
✅ True: American Bulldog | 🔮 Predicted: Japanese Chin


---

## 💻 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/zero-shot-pet-classification.git
cd zero-shot-pet-classification

# Install dependencies
pip install torch torchvision sentence-transformers matplotlib seaborn
```

Run the notebook:
Open the .ipynb file in Jupyter or VS Code and run each cell step-by-step.

## 📊 Results
Zero-Shot Accuracy on Unseen Classes: ~15.57%

Indicates that while the model captures some semantic mapping, performance could be improved with more sophisticated mapping (e.g., MLP, contrastive loss, etc.)

## 🧠 Possible Improvements
* Replace linear mapper with a multi-layer neural network.

* Use contrastive or triplet loss instead of MSE.

* Leverage prompt-based class descriptions instead of raw class names for richer sentence embeddings.




