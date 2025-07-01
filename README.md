# 🐱 Cat Breed Classifier (TensorFlow / Keras)

This project is a deep learning image classification model that identifies the **breed of a cat** from an uploaded image. It is trained using **transfer learning (MobileNetV2)** on a filtered version of the Oxford-IIIT Pet Dataset containing only **12 cat breeds**.

---

## 📂 Dataset

- 📦 Source: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- ✅ Only cat breeds were selected (12 in total)
    ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll',
    'Russian_Blue', 'Siamese', 'Sphynx']
- ✅ Data loading and preprocessing handled via `tensorflow_datasets` with custom filtering

---

## 🏗️ Model Architecture

- 🔧 Base Model: [MobileNetV2](https://arxiv.org/abs/1801.04381) (pre-trained on ImageNet)
- 🧠 Custom Top Layers:
  - GlobalAveragePooling2D
  - Dense(128, relu) + Dropout(0.3)
  - Dense(12, softmax)

---

## 🔁 Training Workflow

- 🔹 Base model frozen and trained for 5 epochs
- 🔹 Then fine-tuned (unfrozen last layers) for better accuracy
- 🧪 Evaluation done using `sparse_categorical_crossentropy`

---

## 🚀 Prediction

- Upload an image via Google Colab
- Get the predicted **cat breed name** and confidence score
- Works with unseen real-world cat images

---

## 📦 Files Included

- `cat_breed_model.keras` — final trained model
- `cat_breed_predict.py` — image prediction code
- `notebook.ipynb` — full training + fine-tuning pipeline
- `README.md` — project documentation

---

## ✅ Requirements

```bash
tensorflow
tensorflow-datasets
matplotlib
numpy
