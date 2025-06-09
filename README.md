
# 🧠 Personality Predictor (Introvert/Extrovert)

A beginner-friendly Machine Learning project to classify a user's personality type — Introvert or Extrovert — based on their behavior. Built using Python, trained in Jupyter/Colab, and deployed using Streamlit (mobile-friendly).

---

## 📊 Features Used for Prediction

| Feature                    | Description                                         |
|---------------------------|-----------------------------------------------------|
| `Time_spent_Alone`        | Hours spent alone daily (0–11)                      |
| `Stage_fear`              | Has stage fright (Yes/No)                           |
| `Social_event_attendance` | Frequency of attending social events (0–10)         |
| `Going_outside`           | Frequency of going outside (0–7)                    |
| `Drained_after_socializing` | Feels tired after socializing (Yes/No)           |
| `Friends_circle_size`     | Number of close friends (0–15)                      |
| `Post_frequency`          | Frequency of social media posts (0–10)              |
| `Personality`             | Target label — Extrovert or Introvert               |

---

## 🧪 How to Train the Model

1. Upload your dataset ZIP to **Google Drive**
2. Use **Google Colab** or Jupyter to run the following:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip and Load
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/personality_dataset_ready.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/personality_dataset')

import pandas as pd
df = pd.read_csv('/content/personality_dataset/personality_dataset.csv')

# Handle Missing Data
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
df['Personality_Label'] = le.fit_transform(df['Personality'])

# Train Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop(['Personality', 'Personality_Label'], axis=1)
y = df['Personality_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save Model
import joblib
joblib.dump(model, 'personality_model.pkl')
```

---

## 🌐 How to Deploy on Streamlit Cloud

1. Upload the following files to a **GitHub repo**:
    - `app.py`
    - `requirements.txt`
    - `personality_model.pkl`
    - `README.md`

2. Go to [Streamlit Cloud](https://share.streamlit.io)

3. Click **"New App"**, select your repo, and deploy!

✅ Done! You now have a mobile-friendly AI web app that predicts personality type.

---

## 🗂 Files Included

- `app.py` – Streamlit frontend
- `requirements.txt` – Required libraries
- `personality_model.pkl` – Trained ML model
- `README.md` – This file (documentation)

---
