# 🐾 Cat vs Dog Image Classifier
### End-to-End Deep Learning Project | Transfer Learning | Flask Deployment

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-95.62%25-22c55e?style=for-the-badge)

---

## 📌 Project Overview

> 🏢 **This project was built as part of my Remote Internship at [HEX Software](https://hexsoftware.com/).**  
> It represents my hands-on work during the internship, where I applied real-world Machine Learning concepts  
> to build a production-ready deep learning solution from scratch.

This project is a complete, end-to-end **image classification system** that can distinguish between
cats and dogs with **95.62% accuracy**. It was built from scratch — starting from raw data collection,
all the way to a **live Flask REST API** where anyone can upload an image and get an instant prediction.

This is not just a model — it's a full ML pipeline that mirrors what data scientists build in the real world.

> **"A cat or dog photo goes in. The model says what it is — with confidence."**

---

## 🎯 What Problem Does It Solve?

Binary image classification is a foundational computer vision task used in:
- 🏥 **Medical imaging** — detecting tumors vs healthy tissue
- 🔒 **Security systems** — face recognition, threat detection
- 🛒 **E-commerce** — automatic product categorization
- 🚗 **Self-driving cars** — object recognition on the road

This project solves it for cats vs dogs, but the **exact same pipeline** works for any 2-class image problem.

---

## 📊 Results at a Glance

| Phase               | Method                      | Val Accuracy |
|---------------------|-----------------------------|--------------|
| Baseline            | Basic CNN (no augmentation) | ~72%         |
| + Data Augmentation | Flip, rotate, zoom          | ~82%         |
| + Dropout +BatchNorm| Regularization              | ~91%         |
| Phase 1             | VGG16 Feature Extraction    | 90.83%       |
| Phase 2             | VGG16 Fine Tuning           | **95.62%**   |

---

## 🛠️ Tech Stack — What I Used & Why

| Tool                | Purpose                                     |
|---------------------|---------------------------------------------|
| Python              | Main programming language                   |
| TensorFlow + Keras  | Build and train the deep learning model     |
| VGG16               | Pretrained model used for Transfer Learning |
| TensorFlow Datasets | Load the Cats vs Dogs dataset easily        |
| Flask               | Deploy the model as a web API               |
| Matplotlib + Seaborn| Plot training graphs and confusion matrix   |
| NumPy + Pillow      | Handle image data and array operations      |

---

### 🐍 Python
I used Python because it is the most popular language for Machine Learning.
It is simple to read and write, which means I could focus on building the model
instead of worrying about complex code. Almost every ML job in the world uses Python.

---

### 🧠 TensorFlow + Keras
TensorFlow is the engine that does all the heavy math behind the scenes (like
calculating errors and updating weights). Keras sits on top of TensorFlow and
gives simple commands like `model.fit()` to train the model without writing
hundreds of lines of code.

**Important tools I used from Keras:**

| Tool                  | What it does                                                  |
|-----------------------|---------------------------------------------------------------|
| `ImageDataGenerator`  | Creates extra training images on the fly (flip, rotate, zoom) |
| `VGG16`               | Loads the pretrained VGG16 model with ImageNet weights        |
| `EarlyStopping`       | Stops training automatically when the model stops improving   |
| `ModelCheckpoint`     | Saves the best version of the model during training           |
| `ReduceLROnPlateau`   | Slows down learning when the model gets stuck                 |

---

### 🏗️ VGG16 — Transfer Learning
VGG16 is a deep neural network that was already trained on **1.4 million images**
across 1000 categories. Instead of training a new model from zero (which would take
weeks and millions of images), I reused VGG16's existing knowledge and only taught
it the difference between cats and dogs.

Think of it like this: **VGG16 is an experienced artist who already knows how to draw.
I just taught them to draw cats and dogs specifically.**

I trained in two steps:
1. **Phase 1** — Keep VGG16 frozen, only train the new layers I added → **90.83% accuracy**
2. **Phase 2** — Unfreeze the last 4 VGG16 layers and fine-tune carefully → **95.62% accuracy**

---

### 📦 TensorFlow Datasets (TFDS)
This library lets you download famous datasets with just one line of code.
It automatically splits the data into train, validation, and test sets,
and loads images in batches so the computer does not run out of memory.

---

### 🔄 Data Augmentation
With only 16,000 images, the model could start memorizing the training photos
instead of actually learning. Augmentation solves this by slightly changing
each image every time it is shown — flipping it, rotating it, zooming in.
The model then sees a "new" version every epoch, which helps it learn better.

---

### 🛡️ Dropout + Batch Normalization
These two techniques prevent the model from over-learning on training data:

- **Dropout(0.5)** — During training, randomly switches off 50% of neurons so
  the model cannot rely too heavily on any single path through the network.
  This forces it to learn more robust patterns.
- **BatchNormalization** — After each layer, it adjusts the numbers to stay
  in a healthy range. This makes training faster and more stable.

---

### 🌐 Flask
Flask is a small and simple Python web framework. I used it to turn the trained
model into a REST API — meaning anyone can send a photo to a web address and
get a prediction back. This is the standard way companies deploy ML models in
real products.

---

### 📊 Matplotlib + Seaborn
After training, I used these libraries to draw graphs that show how the model
improved over time (accuracy and loss curves), and to create a confusion matrix
that shows exactly where the model made correct and incorrect predictions.

---

### 🔢 NumPy + Pillow
Every image is stored as a 3D grid of numbers (height × width × 3 colors).
NumPy handles all the math on these numbers. Pillow opens the image files and
converts them into the format the model can understand.

---

## 🏛️ Model Architecture

```
Input Image (150×150×3)
        │
┌───────▼────────┐
│   VGG16 Base   │  ← 14.7M params, FROZEN
│  (13 Conv Layers│    ImageNet knowledge
│  + 3 MaxPools) │
└───────┬────────┘
        │
┌───────▼─────────────┐
│ GlobalAveragePooling │  ← smarter than Flatten
└───────┬─────────────┘
        │
┌───────▼──────┐
│  Dense(256)  │  ← 131K trainable params
│  + BatchNorm │
│  + Dropout   │
└───────┬──────┘
        │
┌───────▼──────┐
│   Dense(1)   │  ← Sigmoid: 0=Cat, 1=Dog
│   Sigmoid    │
└──────────────┘
        │
   Prediction
(0.94 → "Dog, 94% confident")
```

**Total params:** 14,847,297  
**Trainable (Phase 1):** 132,097 (only the head)  
**Trainable (Phase 2):** 1,849,345 (head + last 4 VGG16 layers)

---

## 🤔 Known Limitations

- Model confidence can be low (~54%) for unusual poses or lighting
- Trained only on standard cat/dog photos — may struggle with cartoon images, drawings, or extreme close-ups
- VGG16 is a relatively old architecture — EfficientNet or ViT would give better results
- No handling for images that contain neither a cat nor a dog

---

## 📚 Resources That Helped

- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Applications — VGG16](https://keras.io/api/applications/vgg/)
- [TensorFlow Datasets — Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- [CS231n — Convolutional Neural Networks](https://cs231n.github.io/)

---

## 👩‍💻 Author

**[Rafia Gul]**  
Machine Learning Enthusiast | Deep Learning | Computer Vision

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**⭐ If this project helped you learn something, please give it a star! ⭐**

*Built with passion for learning Deep Learning* 🧠


