import streamlit as st
import requests
import re
import nltk
from collections import Counter
from nltk.util import ngrams

# Download corpus
url = "http://norvig.com/big.txt"
response = requests.get(url)
if response.status_code == 200:
    with open("big.txt", "wb") as file:
        file.write(response.content)
else:
    st.error("Failed to download corpus.")

# Tokenization
def words(text):
    return re.findall(r'\w+', text.lower())

# Load words
WORDS = Counter(words(open('big.txt').read()))

# Probability of a word
def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N

# Edit functions
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    return set(w for w in words if w in WORDS)

def candidates(word):
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def correction(word):
    return max(candidates(word), key=lambda w: (P(w), -edit_distance(word, w)))

def bigrams(words_list):
    return list(ngrams(words_list, 2))

def build_bigram_model(text):
    words_list = words(text)
    bigram_counts = Counter(bigrams(words_list))
    return bigram_counts

BIGRAMS = build_bigram_model(open('big.txt').read())

def P_bigram(w1, w2):
    return BIGRAMS[(w1, w2)] / WORDS[w1] if w1 in WORDS else 0

def correction_bigram(prev_word, word):
    return max(candidates(word), key=lambda w: (P_bigram(prev_word, w), P(w)))

# 🚀 **Streamlit UI Enhancements**
st.set_page_config(page_title="Spelling Corrector", page_icon="🔤", layout="centered")

# 🎨 **Custom CSS Styling**
st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .stTextInput > div > div > input { border: 2px solid #4CAF50; border-radius: 10px; font-size: 18px; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 10px; font-size: 16px; }
    .stMarkdown { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# **Sidebar for About Section**
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("📝 **Spelling Corrector** uses a probabilistic model to fix spelling mistakes in words.")
st.sidebar.markdown("🔍 Based on Peter Norvig's NLP approach.")
st.sidebar.markdown("📚 Trained on a large text corpus.")
st.sidebar.markdown("👨‍💻 Built with **Python & Streamlit**.")



# **Title & Subtitle**
st.title("🔤 AI-Powered Spelling Corrector")
st.markdown("✨ **Enter a word below**, and the AI will suggest the correct spelling.")

# **User Input**
prev_word = None
user_input = st.text_input("Enter a word:", key="input")

if user_input:
    if prev_word:
        corrected = correction_bigram(prev_word, user_input)
    else:
        corrected = correction(user_input)

    # **Output with Animation & Styling**
    st.success(f"✅ **Corrected: kjh** `{corrected}`")
    st.info(f"📝 **Entered:** `{user_input}`")
    
    prev_word = corrected

# **Footer**
st.markdown("<br><center>💡 Develop by MOEEZ AHMAD using Streamlit</center>", unsafe_allow_html=True)
