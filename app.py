import streamlit as st
import re
import string
import pickle
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

@st.cache_resource
def load_model_and_artifacts():
    tfidf_input = Input(shape=(10000,), name="tfidf_input")
    dense_stat_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01), name="tfidf_dense_1")(tfidf_input)
    bn_1 = tf.keras.layers.BatchNormalization(name="bn_1")(dense_stat_1)
    drop_1 = Dropout(0.4, name="drop_1")(bn_1)

    dense_stat_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01), name="tfidf_dense_2")(drop_1)
    bn_2 = tf.keras.layers.BatchNormalization(name="bn_2")(dense_stat_2)
    drop_2 = Dropout(0.4, name="drop_2")(bn_2)

    seq_input = Input(shape=(100,), name="sequence_input")
    embedding_layer = Embedding(input_dim=10000, output_dim=200, name="word_embedding")(seq_input)
    spatial_dropout = SpatialDropout1D(0.2, name="spatial_dropout")(embedding_layer)
    bilstm_layer = Bidirectional(LSTM(64, return_sequences=False), name="bilstm_context")(spatial_dropout)

    fused_features = Concatenate(name="feature_fusion")([drop_2, bilstm_layer])
    post_fusion_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01), name="post_fusion_dense")(fused_features)
    bn_fusion = tf.keras.layers.BatchNormalization(name="bn_fusion")(post_fusion_dense)
    drop_fusion = Dropout(0.4, name="drop_fusion")(bn_fusion)
    output_probability = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01), name="binary_output")(drop_fusion)

    model = Model(inputs=[tfidf_input, seq_input], outputs=output_probability)
    
    model.load_weights('optimized_hybrid_model.keras')

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sequence_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    return model, vectorizer, tokenizer

model, vectorizer, tokenizer = load_model_and_artifacts()

st.title("تحليل مشاعر التغريدات العربية 📊")
st.write("أدخل نص التغريدة في المربع بالأسفل وسيقوم الموديل بتصنيفها (إيجابية / سلبية).")

user_input = st.text_area("نص التغريدة:", placeholder="اكتب تغريدتك هنا...", height=150)

if st.button("تحليل التغريدة 🚀"):
    if user_input.strip() == "":
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        with st.spinner('جاري تحليل التغريدة...'):
            processed_text = clean_arabic_text(user_input)
            
            tfidf_features = vectorizer.transform([processed_text]).astype('float32').toarray()
            seq_features = tokenizer.texts_to_sequences([processed_text])
            seq_padded = pad_sequences(seq_features, maxlen=100, padding='post', truncating='post')
            
            prediction_prob = model.predict([tfidf_features, seq_padded], verbose=0)[0][0]
            
            st.markdown("---")
            st.subheader("النتيجة:")
            
            col1, col2 = st.columns(2)
            if prediction_prob >= 0.5:
                col1.success("**إيجابية 🟢**")
                col2.info(f"**نسبة الثقة:** {prediction_prob*100:.1f}%")
            else:
                col1.error("**سلبية 🔴**")
                col2.info(f"**نسبة الثقة:** {(1-prediction_prob)*100:.1f}%")
