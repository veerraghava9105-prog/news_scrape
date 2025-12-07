import streamlit as st
from pymongo import MongoClient
import re

# --- DB ---
client = MongoClient("mongodb://localhost:27017")
db = client["newsdb"]
col = db["articles"]

# --- Page Setup ---
st.set_page_config(page_title="News Browser", layout="wide")

# --- CUTE DARK THEME CSS ---
st.markdown("""
<style>

body {
    background-color: #0e0e0f;
}

.card {
    background: linear-gradient(145deg, #18181a, #121213);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 28px;
    border: 1px solid #2a2a2d;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

.card:hover {
    transform: scale(1.01);
    transition: 0.15s ease-out;
    border-color: #ff4ecb77;
}

.title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ff78e3;
}

.url a {
    font-size: 14px;
    color: #82d1ff;
    text-decoration: none;
}
.url a:hover {
    text-decoration: underline;
}

.meta {
    font-size: 14px;
    color: #b3b3b3;
    margin-top: 6px;
}

.snippet {
    font-size: 15px;
    margin-top: 10px;
    color: #e6e6e6;
}

.count-badge {
    font-size: 16px;
    padding: 6px 14px;
    border-radius: 10px;
    background-color: #ff4ecb22;
    color: #ff9ff6;
    display: inline-block;
    margin-top: -10px;
    margin-bottom: 25px;
    border: 1px solid #ff4ecb55;
}

img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title(" 🐣News DB View")

# --- TOTAL COUNT ---
total = col.count_documents({})
st.markdown(f'<div class="count-badge">Total Articles: {total}</div>', unsafe_allow_html=True)

# --- SEARCH BOX ---
search = st.text_input("Search (title, url, domain, text)", "")

# --- QUERY ---
if search.strip() == "":
    query = {}
else:
    s = search.strip()
    query = {
        "$or": [
            {"title": {"$regex": s, "$options": "i"}},
            {"url": {"$regex": s, "$options": "i"}},
            {"domain": {"$regex": s, "$options": "i"}},
            {"text": {"$regex": s, "$options": "i"}}
        ]
    }

docs = col.find(query).limit(50)

# --- RENDER EACH CARD ---
for d in docs:
    title = d.get("title", "")
    url = d.get("url", "")
    domain = d.get("domain", "")
    text = d.get("text", "")

    # Snippet = first 2 sentences
    snippet = ""
    if text:
        parts = re.split(r'[.!?]', text)
        snippet = ". ".join(parts[:2]).strip() + "..."

    # First image
    imgs = d.get("images", [])
    img_url = imgs[0]["source_url"] if (imgs and imgs[0].get("source_url")) else None

    # CARD UI
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2.6])

    with col1:
        if img_url:
            st.image(img_url, width=260)
        else:
            st.write("No image")

    with col2:
        st.markdown(f'<div class="title">{title}</div>', unsafe_allow_html=True)

        # Working hyperlink 🔗
        st.markdown(f'<div class="url"><a href="{url}" target="_blank">{url}</a></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="meta">🌐 Domain: {domain}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="snippet">{snippet}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
