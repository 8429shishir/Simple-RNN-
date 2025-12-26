# 🧠 Sentiment Analysis Web App (IMDB Reviews)

An end-to-end **Natural Language Processing (NLP)** web application that predicts the **sentiment (Positive / Negative)** of movie reviews using a **Simple Recurrent Neural Network (RNN)** trained on the **IMDB dataset**.  
The application provides **real-time predictions**, **confidence scores**, and **probability bar visualizations** through an interactive **Streamlit** interface.

---

## 🚀 Features

- ✍️ Enter any movie review and get instant sentiment prediction  
- 🤖 Deep Learning based sentiment classification using RNN  
- 📊 Confidence score with **Positive vs Negative probability bars**  
- ⚡ Fast inference with cached model loading  
- 🎨 Clean and professional Streamlit UI  

---

## 🧠 Model Details

- **Dataset:** IMDB Movie Reviews  
- **Vocabulary Size:** 10,000 most frequent words  
- **Text Processing:** Tokenization and sequence padding  
- **Model Architecture:**
  - Embedding Layer  
  - Simple RNN Layer  
  - Dense Output Layer with Sigmoid activation  
- **Loss Function:** Binary Cross-Entropy  
- **Output:** Sentiment probability (0–1)

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **NLP:** IMDB Dataset  
- **Frontend:** Streamlit  
- **Visualization:** Streamlit progress bars  

---


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/8429shishir/sentiment-analysis-rnn.git
cd sentiment-analysis-rnn
