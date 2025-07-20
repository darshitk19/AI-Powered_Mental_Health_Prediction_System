
# 🧠 AI-Powered Mental Health Prediction System

A multimodal deep learning–based system that predicts a user's mental state from **text**, **audio**, or **image** inputs. It leverages **CNN**, **RNN with MFCC**, and **TF-IDF with traditional ML** to assess emotions and detect conditions like stress, anxiety, or calmness.

🚀 Built for research, awareness, and future integration with a **Bhagavad Gita–based chatbot** for spiritual guidance.

---

## 📌 Key Features

- 🎧 **Audio Input** – Voice emotion classification using MFCC + LSTM  
- 📝 **Text Input** – Emotion detection using TF-IDF + traditional ML classifier  
- 🖼️ **Image Input** – Facial emotion recognition using a custom CNN  
- 🧠 **Multimodal Prediction** – Combines results for robust output  
- 📜 **Future Scope** – Gita Shloka chatbot offering advice based on emotion

---

## 🛠 Technologies Used

| Input Type | Model/Method            | Tools/Libraries                          |
|------------|-------------------------|------------------------------------------|
| Image      | CNN (custom layers)     | `tensorflow`, `keras`, `pillow`          |
| Audio      | MFCC + RNN (LSTM)       | `librosa`, `tensorflow`, `pandas`        |
| Text       | TF-IDF + ML Classifier  | `nltk`, `scikit-learn`, `joblib`         |
| Interface  | Streamlit Web App       | `streamlit`                              |

---

## 📦 Installation Instructions (Module-wise)

### 🖼️ Image (CNN Model)

```bash
pip install tensorflow keras pillow numpy matplotlib pickle5
```

### 🔊 Audio (MFCC + LSTM Model)

```bash
pip install librosa tensorflow numpy pandas scikit-learn
```

### 📝 Text (TF-IDF + ML Classifier)

```bash
pip install pandas numpy nltk scikit-learn joblib imbalanced-learn
```

### 🌐 Streamlit (Web App UI)

```bash
pip install streamlit
```

### ✅ Install Everything at Once (Recommended)

```bash
pip install tensorflow keras pillow matplotlib pickle5 librosa pandas numpy scikit-learn nltk joblib imbalanced-learn streamlit
```

---

## ▶️ Run the Project (Streamlit Command)

After setting everything up and placing your models, run:

```bash
streamlit run app.py
```

Replace `app.py` with your actual Streamlit file name if different.

---

## 📁 Dataset Used

| Dataset Name | Use Case | Source |
|--------------|----------|--------|
| FER 2013 | Facial emotions | Kaggle – Face Dataset |
| RAVDESS | Audio emotions | Zenodo – RAVDESS |
| Custom Text Dataset | Text-based emotions | Manually curated and labeled paragraphs with mental state tags |

---

## ⚙️ Project Workflow

1. User uploads image/audio or types a message.

2. Model processes input using:
   - **CNN** for image input
   - **MFCC + LSTM** for audio input
   - **TF-IDF + Classifier** for text input

3. Prediction result is shown on Streamlit UI.

4. (Coming Soon) A Bhagavad Gita–based shloka + advice is recommended based on emotion.

---

## 📊 Model Performance

| Modality | Model Used | Accuracy |
|----------|------------|----------|
| Image | CNN | 88.81% |
| Audio | MFCC + LSTM | ~79% |
| Text | TF-IDF + Classifier | ~89% |
| Combined | All | ~80%+ |

---

## 📈 Metrics (Average)

- **Precision:** 0.84
- **Recall:** 0.83
- **F1-Score:** 0.835

---

## 🔮 Future Enhancements

- 📲 Real-time mobile/wearable deployment
- 🧘 Integration with Gita Shloka chatbot
- 🌐 Multilingual support
- 🔍 Explainable AI (XAI) for trustworthiness
- 📊 Personalized mental health tracking

---

## 🙏 Acknowledgements

- **Datasets:** FER-2013, RAVDESS
- **Libraries:** TensorFlow, Keras, scikit-learn, Streamlit
- **Inspiration** from the Bhagavad Gita and AI for social good

---

## 🤝 Contributions

Feel free to fork, raise issues, or submit pull requests. Collaboration is encouraged for:

- Expanding dataset support
- Improving UI/UX in Streamlit
- Bhagavad Gita chatbot module

---

## 📬 Contact

**Darshit Kachhadiya**  
📧 darshitkachhadiya19@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/darshit-kachhadiya/)

---

*"Understanding the mind through AI. Healing it with wisdom."*
