# IMDB Sentiment Analysis using RNN (PyTorch)

## 📌 Project Overview

This project implements **Sentiment Analysis on the IMDB Movie Reviews Dataset** using a **Recurrent Neural Network (RNN)** built from scratch with **PyTorch**.

The goal is to classify movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and Deep Learning.

The pipeline includes:

* Text preprocessing
* Tokenization
* Stopword removal
* Stemming
* TF-IDF Vectorization
* RNN model training
* Sentiment prediction

This project demonstrates an **end-to-end NLP workflow** from raw text data to a trained deep learning model.

---

# 📊 Dataset

Dataset used: **IMDB Movie Reviews Dataset**

* 50,000 movie reviews
* Binary sentiment classification
* Labels: `positive` / `negative`

Dataset columns:

| Column    | Description       |
| --------- | ----------------- |
| review    | Movie review text |
| sentiment | Sentiment label   |

---

# ⚙️ Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* NLTK
* Scikit-Learn

---

# 🧠 NLP Preprocessing Pipeline

The following preprocessing steps were applied:

### 1️⃣ Lowercase Conversion

All text converted to lowercase.

### 2️⃣ URL Removal

Removed URLs using Regular Expressions.

### 3️⃣ Punctuation Removal

Removed punctuation characters.

### 4️⃣ HTML Tag Removal

Removed HTML tags from reviews.

### 5️⃣ Stopword Removal

Removed common English stopwords using NLTK.

### 6️⃣ Stemming

Applied **Porter Stemmer** to reduce words to root form.

Example:

```
running → run
played → play
```

---

# 🔢 Feature Engineering

### TF-IDF Vectorization

Text is converted into numerical form using:

```
TfidfVectorizer(max_features=5000)
```

This converts text into **5000 numerical features** representing word importance.

---

# 🧠 Deep Learning Model

The model used is a **Recurrent Neural Network (RNN)**.

### Architecture

Input Layer
↓
RNN Layer
↓
Fully Connected Layer
↓
Sigmoid Activation
↓
Binary Sentiment Output

### Model Implementation

```
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out
```

---

# 🏋️ Model Training

Loss Function:

```
Binary Cross Entropy Loss
```

Optimizer:

```
Adam Optimizer
```

Training Parameters:

| Parameter    | Value |
| ------------ | ----- |
| Epochs       | 10    |
| Batch Size   | 64    |
| Hidden Size  | 128   |
| Max Features | 5000  |

---

# 📈 Model Evaluation

Accuracy is computed on the test dataset.

Example:

```
accuracy = (correct_predictions / total_predictions) * 100
```

---

# 🚀 Project Pipeline

```
Raw Text
   ↓
Preprocessing
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming
   ↓
TF-IDF Vectorization
   ↓
Train/Test Split
   ↓
PyTorch Dataset
   ↓
RNN Model Training
   ↓
Prediction
   ↓
Accuracy Evaluation

# 📌 Key Learning Outcomes

This project demonstrates:

* NLP preprocessing techniques
* Feature engineering with TF-IDF
* RNN architecture implementation
* PyTorch training pipeline
* Binary classification with deep learning

---

# 📬 Connect With Me

If you liked this project, feel free to connect with me on LinkedIn.

Let's collaborate on **AI, Deep Learning, and NLP projects**.
