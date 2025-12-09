import os
import nltk
nltk.download()
nltk.download('punkt_tab')
import streamlit as st
import pdfplumber
import re
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import pandas as pd

st.set_page_config(page_title="PDF Keyword Paragraphs + WordCloud", layout="wide")

st.title("PDF → Paragraph Extractor + Word Cloud")
st.write("Upload a PDF, enter multiple keywords, and see paragraphs that contain them. Word cloud is created from the matched paragraphs.")

# ---------- Sidebar controls ----------
st.sidebar.header("Options")
min_par_len = st.sidebar.number_input("Minimum paragraph length (chars)", value=30, min_value=1)
split_by = "single-newline-join"
st.sidebar.write("Paragraph split method:\n Single-newline-join")
show_page_numbers = st.sidebar.checkbox("Show source page numbers", value=True)
remove_short = st.sidebar.checkbox("Remove very short paragraphs (< min length)", value=True)

# ---------- File uploader ----------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
sample_button = st.sidebar.button("Use sample PDF path (local)")  # if running locally

# ---------- Helper functions ----------
def extract_pages_from_pdf_filelike(filelike):
    """Return list of page texts from a pdf file-like object"""
    pages = []
    with pdfplumber.open(filelike) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages.append(raw)
    return pages

def normalize_paragraphs_for_pages(pages, method="blank-lines"):
    """
    Convert list of page texts into list of (page_no, paragraph_text)
    Method 'blank-lines' splits on 2+ newlines.
    Method 'single-newline-join' will join lines inside a page and split on double-newlines,
    but will also coalesce short line breaks into paragraphs.
    """
    paras = []
    for pno, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue
        text = page_text.replace("\r", "\n")
        if method == "blank-lines":
            # split on one or more blank lines
            raw_pars = re.split(r'\n\s*\n+', text)
            raw_pars = [r.strip() for r in raw_pars if r.strip()]
        else:
            # attempt better coalescing: join lines that likely belong to same paragraph
            # heuristics: if a line ends with '-' (hyphen) or line is short, join; else keep.
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            merged = []
            buffer = lines[0]
            for line in lines[1:]:
                if buffer.endswith('-'):  # hyphenated word split - join without extra space
                    buffer = buffer[:-1] + line.lstrip()
                elif (len(buffer) < 60 and not buffer.endswith('.')) or (len(line) < 60 and not line[0].isupper()):
                    # likely same paragraph line break (heuristic)
                    buffer = buffer + " " + line.lstrip()
                else:
                    merged.append(buffer.strip())
                    buffer = line
            merged.append(buffer.strip())
            # then split merged blocks on explicit blank lines if any
            raw_pars = []
            for block in merged:
                raw_pars.extend([r.strip() for r in re.split(r'\n\s*\n+', block) if r.strip()])
        for rp in raw_pars:
            if remove_short and len(rp) < min_par_len:
                continue
            paras.append((pno, rp))
    return paras

def paragraphs_containing_keywords(paragraphs, keywords):
    """
    paragraphs: list of (page_no, paragraph_text)
    keywords: list of lowercase keywords
    returns list of (page_no, paragraph_text, matched_terms_set)
    """
    results = []
    # Precompile regex that matches any keyword as a word (case-insensitive)
    # Escape keywords to avoid regex meta-characters
    kw_escaped = [re.escape(k) for k in keywords if k.strip()]
    if not kw_escaped:
        return []
    pattern = re.compile(r'\b(?:' + '|'.join(kw_escaped) + r')\b', flags=re.IGNORECASE)
    for pno, para in paragraphs:
        found = pattern.findall(para)
        if found:
            matched = set([f.lower() for f in found])
            results.append((pno, para, matched))
    return results

def make_wordcloud_from_text(text, max_words=200):
    if not text or not text.strip():
        return None
    wc = WordCloud(width=900, height=450, background_color="white", collocations=False, max_words=max_words)
    wc.generate(text)
    return wc

# ---------- Main logic ----------
if uploaded_file is None and not sample_button:
    st.info("Upload a PDF on the left to proceed or click the sidebar sample button if running locally.")
    st.stop()

# If sample button pressed, try a local file path (useful when running locally)
pages = []
if sample_button:
    # (Customize this path for your local test file)
    local_path = "Pernod_Ricard.pdf"
    if Path(local_path).exists():
        pages = extract_pages_from_pdf_filelike(local_path)
        st.sidebar.success(f"Loaded local sample: {local_path}")
    else:
        st.sidebar.error(f"Local sample {local_path} not found. Upload a PDF instead.")
        st.stop()
else:
    # uploaded_file is a SpooledTemporaryFile-like object
    pages = extract_pages_from_pdf_filelike(uploaded_file)

st.sidebar.write(f"Pages detected: {len(pages)}")
st.write(f"**Pages detected:** {len(pages)}")

# Paragraph extraction
paragraph_method = "blank-lines" if split_by == "blank-lines" else "single-newline-join"
paragraphs = normalize_paragraphs_for_pages(pages, method=paragraph_method)
st.write(f"Extracted **{len(paragraphs)}** paragraphs (after filtering).")

# Keywords input: multiple keywords
kw_input = st.text_input("Enter keywords (comma or space separated). Example: market, brand, culture")
kw_button = st.button("Find paragraphs")
if not kw_input:
    st.warning("Enter one or more keywords above to filter paragraphs.")
    st.stop()

# parse keywords
# Accept comma-separated, or space-separated, or newline separated
raw_kw = re.split(r'[,\n;]+', kw_input)
keywords = []
for token in raw_kw:
    token = token.strip().lower()
    if not token:
        continue
    # allow multi-word keywords quoted with double quotes: "global t+1"
    # but for simplicity accept as-is
    keywords.append(token)

if not keywords:
    st.error("No valid keywords parsed. Please enter keywords separated by comma, semicolon, or newline.")
    st.stop()

# find paragraphs with these keywords
matches = paragraphs_containing_keywords(paragraphs, keywords)
st.write(f"Paragraphs matching any of: **{', '.join(keywords)}** → **{len(matches)}** found.")

# Display matched paragraphs in expanders with page numbers and matched terms
if matches:
    st.markdown("### Matched paragraphs")
    for idx, (pno, para, matched_set) in enumerate(matches, start=1):
        header = f"Paragraph #{idx}"
        if show_page_numbers:
            header += f" — Page {pno}"
        header += f" — matched: {', '.join(sorted(matched_set))}"
        with st.expander(header, expanded=(idx <= 3)):
            st.write(para)
            st.download_button(
                label="Download this paragraph",
                data=para,
                file_name=f"paragraph_page{pno}_{idx}.txt",
                mime="text/plain"
            )
else:
    st.info("No paragraphs matched the keywords. Try broader keywords or adjust paragraph-splitting options.")

# Wordcloud from matched paragraphs
if matches:
    combined = " ".join([para for (_, para, _) in matches])
    wc = make_wordcloud_from_text(combined, max_words=200)
    if wc:
        st.markdown("### Word Cloud from matched paragraphs")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Word cloud could not be generated (empty text).")

    # Offer download for all matched paragraphs combined
    combined_bytes = combined.encode("utf-8")
    st.download_button("Download all matched paragraphs (.txt)", data=combined_bytes, file_name="matched_paragraphs.txt", mime="text/plain")
    # Show top 30 word frequencies for quick inspection
    tokens = re.findall(r"\w+", combined.lower())
    freq = Counter(tokens)
    top30 = freq.most_common(30)
    st.markdown("**Top 30 tokens (from matched paragraphs)**")
    st.table(pd.DataFrame(top30, columns=["token", "count"]))
else:
    st.stop()


