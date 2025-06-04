# 🧠 Zero-Shot Learning with Semantic Embeddings

This project implements a Zero-Shot Learning (ZSL) framework using deep features from a pretrained ResNet model and semantic embeddings generated via a Sentence-BERT model. The aim is to classify unseen classes (not present during training) by leveraging textual descriptions of each class.

---

## 📌 Objective

To recognize both seen and unseen classes by:
- Extracting visual features from images using a CNN (ResNet-18).
- Mapping those features into a semantic embedding space using a learned transformation.
- Comparing mapped features with class description embeddings for classification.

---

## 📂 Dataset

- A custom dataset containing both **cat and dog breeds**.
- The dataset is split into:
  - **70% Seen Classes** (used for training)
  - **30% Unseen Classes** (used for testing ZSL performance)

---

## 🔁 Pipeline Overview

### Step 1: Class Split
- Randomly shuffle class labels and split into `seen` and `unseen`.
- Store in `class_split.json`.

### Step 2: Feature Extraction
- Use pretrained `ResNet18` with its final classification layer removed.
- Extract 512-dimensional features for all seen class images.
- Save features and corresponding class labels to `seen_features.pt`.

### Step 3: Semantic Embedding
- Use `sentence-transformers` (e.g., `'all-MiniLM-L6-v2'`) to embed class descriptions (from `class_descriptions.json`).
- Normalize both image features and text embeddings for cosine similarity.

### Step 4: Learn Mapping
- Train a simple `Linear` mapping model from ResNet feature space to semantic space.
- Loss function: `MSELoss`.
- Save model as `mapping_model.pth`.

### Step 5: Evaluate on Seen Classes
- Map features to semantic space and compare with seen class embeddings.
- Compute accuracy using cosine similarity.
- Evaluate predictions and print sample outputs.

### Step 6: Zero-Shot Prediction (Unseen)
- Extract features of unseen images.
- Compare mapped features with **unseen class embeddings**.
- Predict label with highest cosine similarity.
- Evaluate ZSL accuracy and print predictions.

---

## 🔧 Requirements

```bash
torch
torchvision
scikit-learn
sentence-transformers
tqdm
```

## 📊 Sample Results
### ✅ Seen Class Accuracy:
Seen class accuracy: 94.44%

### 🚫 Unseen Class Accuracy (ZSL):
Unseen class accuracy: 16.64%

## 📁 Files
* class_split.json – Randomly generated seen/unseen class split.
* class_descriptions.json – Textual descriptions for all classes.
* seen_features.pt – Extracted ResNet features + labels.
* mapping_model.pth – Trained linear mapping model.

## 📌 To Run
Make sure you execute each step in order:
* Split classes.
* Extract seen features.
* Generate and store semantic embeddings.
* Train mapping model.
* Evaluate on seen and unseen classes.

## 💡 Future Improvements
* Use triplet loss or contrastive loss for better semantic alignment.
* Replace ResNet18 with stronger backbones (ResNet50, ViT, etc.).
* Improve textual descriptions with richer context.
* Try Top-K accuracy for better insights.
