
# 🕵️‍♂️ Fake Job Posting Detection  
Detect whether a job posting is **Real** or **Fake** using NLP + Machine Learning (BERT embeddings + Scikit-learn).

## 📌 Project Overview
This project builds an intelligent system that identifies fraudulent job postings using Natural Language Processing (NLP).  
We use **BERT sentence embeddings** for feature extraction and a **Traditional ML classifier** (Logistic Regression / SVM / Random Forest — whichever performs best).

A **Streamlit web app** is included for real-time prediction.

## 📂 Project Structure
```
fake-job-detection/
│── data/
│     ├── fake_job_postings.csv        # Dataset  
│     └── fake_jobs_cleaned.csv                          
│── models/
│     └── best_bert_model.pkl          # Trained ML model                   
│── notebooks/
│     ├── 1_data_cleaning_eda.ipynb    # Data Cleaning + EDA  
│     ├── 2_feature_engineering.ipynb  # BERT Feature Engineering  
│     └── 3_model_building.ipynb       # Model Training & Evaluation  
│── streamlit_app/
│     └── app.py                       # Streamlit UI 
│── README.md
│── requirements.txt                   # Dependencies
```

## 🚀 Features
✔ Data Cleaning & EDA  
✔ BERT Embedding Generation  
✔ ML Model Training (LR / SVM / RF)  
✔ Fake/Real Prediction  
✔ Streamlit Web App  
✔ GitHub-ready Project Structure  

## 📊 Dataset
The dataset contains real and fake job postings with fields like:

- Title  
- Location  
- Company  
- Job Description  
- Telecommuting  
- Fraudulent label (0 = Real, 1 = Fake)

## 🧠 Model Pipeline
1. Load & Clean Dataset  
2. Combine text fields into one string  
3. Generate **BERT embeddings** using:  
   ```
   all-MiniLM-L6-v2
   ```  
4. Train multiple ML models  
5. Select best model → save as  
   ```
   models/best_bert_model.pkl
   ```

## 🧪 Streamlit Web App  
Run the app:

```bash
streamlit run app.py
```

Enter:
✔ Job Title  
✔ Company  
✔ Location  
✔ Job Description  

The app predicts:  
### **🟢 REAL**  
or  
### **🔴 FAKE**

## 📦 Installation

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/tashi-ck/fake-job-detection.git
cd fake-job-detection
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit  
```bash
streamlit run app.py
```

## 📘 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- SentenceTransformers (BERT)
- Matplotlib, Seaborn
- Streamlit

## 👨‍💻 Author  
**Tashen Weerasinghe** – Data Science Undergraduate  
Feel free to ⭐ the repo if you found it useful!
