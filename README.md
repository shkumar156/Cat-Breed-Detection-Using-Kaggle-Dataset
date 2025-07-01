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
- 
---

## 📦 Files Included

- `cat_breed_model.keras` — final trained model
- `cat_breed_predict.py` — image prediction code
- `notebook.ipynb` — full training + fine-tuning pipeline
- `README.md` — project documentation

---

## ✅ Results
![Test6](https://github.com/user-attachments/assets/73c9ed0a-da4f-4fbe-bb3a-c47a3f2cfbac)
![Test5](https://github.com/user-attachments/assets/b309f223-938e-4c26-a8d2-fb10bcd8e9b8)
![Test4](https://github.com/user-attachments/assets/b58711b5-0a2b-4c82-9e5b-75deb67e3113)
![Test3](https://github.com/user-attachments/assets/598cb741-7cd7-403b-89c7-fd83ed86623a)
![Test2](https://github.com/user-attachments/assets/13aef4a4-5639-496b-b845-0beeef7d9fc0)
![Test1](https://github.com/user-attachments/assets/4f7eb20b-6a1f-42d0-9e26-e63876edb63a)


---

## ✅ Requirements

```bash
tensorflow
tensorflow-datasets
matplotlib
numpy

