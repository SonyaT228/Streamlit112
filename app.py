import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Cat Health Predictor")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    return df

df = load_data()

# Показываем успешную загрузку
st.success(f"✅ Data loaded! {len(df)} records found")

# Показываем первые строки
st.subheader("📊 Data Preview")
st.dataframe(df.head(10))

# Создаем модель
@st.cache_resource
def train_model():
    # Определяем здоровье
    def is_healthy(row):
        problems = 0
        if row['Age'] > 15:
            problems += 1
        if row['Weight'] < 2 or row['Weight'] > 8:
            problems += 1
        if row['Playing (min.)'] < 20:
            problems += 1
        if row['Sleeps (hours)'] < 10 or row['Sleeps (hours)'] > 20:
            problems += 1
        return 'Yes' if problems < 2 else 'No'
    
    df['Healthy'] = df.apply(is_healthy, axis=1)
    
    features = ['Age', 'Weight', 'Playing (min.)', 'Sleeps (hours)']
    X = df[features]
    y = df['Healthy']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)
    
    return model, le, features

model, le, features = train_model()

st.success("✅ Model trained successfully!")

# --- ПРОГНОЗИРОВАНИЕ ---
st.subheader("🔮 Predict your cat's health")
st.markdown("Enter your cat's information:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0.0, max_value=30.0, value=3.0)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=15.0, value=4.0)

with col2:
    play = st.number_input("Playing time (minutes per day)", min_value=0, max_value=120, value=30)
    sleep = st.number_input("Sleep hours per day", min_value=5.0, max_value=24.0, value=14.0)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame([[age, weight, play, sleep]], columns=features)
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    result = le.inverse_transform([pred])[0]
    
    if result == "Yes":
        st.success(f"✅ Your cat is HEALTHY! (Confidence: {prob[1]*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"⚠️ Your cat needs ATTENTION! (Risk: {prob[0]*100:.1f}%)")

# --- СТАТИСТИКА ---
st.subheader("📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Cats", len(df))
with col2:
    st.metric("Breeds", df['Breed'].nunique())
with col3:
    st.metric("Avg Weight", f"{df['Weight'].mean():.1f} kg")
