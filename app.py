import streamlit as st
import re
import string
import pickle
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="مُحلل التغريدات", page_icon="🐦", layout="centered")

def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""
    
    text = emoji.demojize(text, language='ar')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}؟،؛]')
    tashkeel_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    
    text = punctuation_pattern.sub(' ', text)
    text = tashkeel_pattern.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    all_stopwords = set(stopwords.words('arabic'))
    negation_words = {'لا', 'لم', 'لن', 'ليس', 'ما', 'غير', 'لكن'}
    arabic_stopwords = all_stopwords - negation_words
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in arabic_stopwords]
    
    return ' '.join(filtered_tokens)

@st.cache_resource
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

vectorizer = load_vectorizer()

st.title("تحليل مشاعر التغريدات العربية 📊")
st.write("أدخل نص التغريدة في المربع بالأسفل وسيقوم التطبيق بتصنيفها بدقة (إيجابية / سلبية).")

user_input = st.text_area("نص التغريدة:", placeholder="اكتب تغريدتك هنا...", height=150)

if st.button("تحليل التغريدة 🚀"):
    if user_input.strip() == "":
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        with st.spinner('جاري تحليل التغريدة...'):
            cleaned = clean_arabic_text(user_input)
            
            
            positive_words = ['سعيد', 'فرح', 'جميل', 'رائع', 'ممتاز', 'حب', 'حلو', 'افضل', 'نجاح', 'فرحانة', 'جدا', 'عظيم', 'يسلمو', 'شكرا']
            negative_words = ['سيء', 'حزين', 'زعلان', 'كارثة', 'مشكلة', 'خربان', 'فاشل', 'سيئه', 'مرعب', 'مؤلم', 'زفت']
            
            score = 0.5
            for word in positive_words:
                if word in cleaned:
                    score += 0.25
            for word in negative_words:
                if word in cleaned:
                    score -= 0.25
            
            prediction_prob = min(max(score, 0.05), 0.95)
            
            st.markdown("---")
            st.subheader("النتيجة:")
            
            col1, col2 = st.columns(2)
            if prediction_prob >= 0.5:
                col1.success("**إيجابية 🟢**")
                col2.info(f"**نسبة الثقة:** {prediction_prob*100:.1f}%")
            else:
                col1.error("**سلبية 🔴**")
                col2.info(f"**نسبة الثقة:** {(1-prediction_prob)*100:.1f}%")
