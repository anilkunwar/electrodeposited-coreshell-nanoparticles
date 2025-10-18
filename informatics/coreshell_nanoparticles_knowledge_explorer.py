import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import Counter
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
import tempfile

# --------------------------
# Setup paths and logging
# --------------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_universe.db")
PDF_DIR = os.path.join(DB_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(DB_DIR, 'coreshellnanoparticles_query.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="Core-Shell Nanoparticles Query Tool", layout="wide")
st.title("Core-Shell Nanoparticles Query Tool with SciBERT")
st.markdown("""
Query **Ag Cu core-shell nanoparticles prepared by electroless deposition** using SciBERT to prioritize relevant abstracts (>30% relevance). Metadata is stored in SQLite, PDFs in individual files and ZIP downloads.
""")

# --------------------------
# Session State for logs
# --------------------------
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

def display_logs():
    st.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

# --------------------------
# Load SciBERT
# --------------------------
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}")
    st.stop()

# --------------------------
# Key Terms
# --------------------------
KEY_TERMS = [
    "core-shell nanoparticles", "electroless deposition", "thermal stability", "electric resistivity",
    "Ag shell", "Cu core", "flexible electronics", "nanotechnology", "applications",
    "silver shell", "copper core", "core-shell", "nanoparticles", "deposition", "electroless",
    "stability", "resistivity", "electronics", "nano"
]

# --------------------------
# SciBERT scoring
# --------------------------
@st.cache_data
def score_abstract_with_scibert(abstract):
    try:
        inputs = scibert_tokenizer(
            abstract, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        probs = softmax(logits, axis=1)
        relevance_prob = probs[0][1]

        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw.lower() in token.lower() for kw in KEY_TERMS)]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1 and relevance_prob < 0.5:
                relevance_prob = min(relevance_prob + 0.2 * len(keyword_indices), 1.0)

        update_log(f"SciBERT scored abstract: {relevance_prob:.3f} (keywords: {len(keyword_indices)})")
        return relevance_prob
    except Exception as e:
        # Fallback
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = sum(word_counts.get(kw.lower(), 0) for kw in KEY_TERMS) / (total_words + 1e-6)
        relevance_prob = min(score / (len(KEY_TERMS)/10), 1.0)
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# --------------------------
# PDF Extraction
# --------------------------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        update_log(f"PDF extraction failed {pdf_path}: {e}")
        return f"Error: {str(e)}"

# --------------------------
# Initialize DB
# --------------------------
def initialize_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                download_status TEXT,
                matched_terms TEXT,
                relevance_prob REAL,
                pdf_path TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                paper_id TEXT,
                entity_text TEXT,
                entity_label TEXT,
                value REAL,
                unit TEXT,
                context TEXT,
                phase TEXT,
                score REAL,
                co_occurrence BOOLEAN,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
        conn.commit()
        conn.close()
        update_log(f"Initialized DB: {db_file}")
    except Exception as e:
        update_log(f"DB initialization failed: {e}")

# --------------------------
# Universe DB Incremental
# --------------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper["id"],
            paper.get("title", ""),
            paper.get("authors", "Unknown"),
            paper.get("year", 0),
            paper.get("content", "No text extracted")
        ))
        conn.commit()
        conn.close()
        update_log(f"Updated universe DB with paper {paper['id']}")
    except Exception as e:
        update_log(f"Error updating universe DB: {str(e)}")
        raise

# --------------------------
# Save to SQLite (Cached)
# --------------------------
@st.cache_data
def save_to_sqlite(papers_df, params_list, metadata_db_file=METADATA_DB_FILE):
    try:
        initialize_db(metadata_db_file)
        conn = sqlite3.connect(metadata_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        update_log(f"Saved {len(papers_df)} papers and {len(params_list)} parameters")
        return f"Saved to {metadata_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {e}")
        return f"Failed to save to SQLite: {e}"

# --------------------------
# arXiv Query
# --------------------------
@st.cache_data
def query_arxiv(query, categories, max_results, start_year, end_year):
    try:
        query_terms = query.strip().split()
        formatted_terms = [term.strip('"').replace(" ", "+") for term in query_terms]
        api_query = " ".join(formatted_terms)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=api_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract = result.summary.lower()
                title = result.title.lower()
                matched_terms = [word for word in query_terms if word.lower() in abstract or word.lower() in title]
                if not matched_terms:
                    continue
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = abstract
                for term in matched_terms:
                    abstract_highlighted = re.sub(r'\b{}\b'.format(re.escape(term)), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms),
                    "relevance_prob": round(relevance_prob*100,2),
                    "pdf_path": None,
                    "content": None
                })
            if len(papers) >= max_results:
                break
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Found {len(papers)} papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {e}")
        st.error(f"arXiv query failed: {e}")
        return []

# --------------------------
# PDF Download & Extract
# --------------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf_and_extract(pdf_url, paper_id, paper_metadata):
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        text = extract_text_from_pdf(pdf_path)
        if not text.startswith("Error"):
            paper_data = {
                "id": paper_id,
                "title": paper_metadata.get("title",""),
                "authors": paper_metadata.get("authors","Unknown"),
                "year": paper_metadata.get("year",0),
                "content": text
            }
            create_universe_db(paper_data)
            file_size = os.path.getsize(pdf_path)/1024
            update_log(f"Downloaded {paper_id} ({file_size:.2f} KB)")
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, text
        else:
            return f"Failed: {text}", None, text
    except Exception as e:
        update_log(f"PDF download failed {paper_id}: {e}")
        return f"Failed: {str(e)}", None, f"Error: {str(e)}"

# --------------------------
# Safe ZIP Creation
# --------------------------
def create_pdf_zip_safe(pdf_paths):
    try:
        tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pdf_path in pdf_paths:
                if pdf_path and os.path.exists(pdf_path):
                    zipf.write(pdf_path, os.path.basename(pdf_path))
        update_log(f"Created ZIP of {len(pdf_paths)} PDFs")
        return tmp_zip.name
    except Exception as e:
        update_log(f"ZIP creation failed: {e}")
        return None

def download_zip_button(zip_path, label="Download All PDFs as ZIP"):
    if zip_path and os.path.exists(zip_path):
        try:
            with open(zip_path, "rb") as f:
                st.download_button(label=label, data=f, file_name=os.path.basename(zip_path), mime="application/zip")
        except Exception as e:
            st.error(f"Failed ZIP download: {e}")

def download_pdf_buttons(pdf_paths):
    for pdf_path in pdf_paths:
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"Download {os.path.basename(pdf_path)}",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
            except Exception as e:
                update_log(f"Failed PDF button {pdf_path}: {e}")

# --------------------------
# Streamlit UI
# --------------------------
st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Query", value=' OR '.join([f'"{term}"' for term in KEY_TERMS]))
default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
categories = st.sidebar.multiselect("Categories", default_categories, default=default_categories)
max_results = st.sidebar.slider("Max Papers", min_value=1, max_value=500, value=10)
current_year = datetime.now().year
start_year = st.sidebar.number_input("Start Year", min_value=1990, max_value=current_year, value=2010)
end_year = st.sidebar.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
output_formats = st.sidebar.multiselect("Output Formats", ["SQLite (.db)", "CSV", "JSON"], default=["SQLite (.db)"])
search_button = st.sidebar.button("Search arXiv")

if search_button:
    if not query.strip():
        st.error("Enter a valid query")
    elif not categories:
        st.error("Select at least one category")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        if not papers:
            st.warning("No papers found")
        else:
            st.success(f"Found {len(papers)} papers, filtering relevance > 30%")
            relevant_papers = [p for p in papers if p["relevance_prob"]>30.0]
            if not relevant_papers:
                st.warning("No papers with relevance > 30%")
            else:
                st.success(f"Downloading PDFs for {len(relevant_papers)} papers...")
                pdf_paths = []
                progress_bar = st.progress(0)
                for i, paper in enumerate(relevant_papers):
                    if paper["pdf_url"]:
                        status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
                        paper["download_status"] = status
                        paper["pdf_path"] = pdf_path
                        paper["content"] = content
                        if pdf_path:
                            pdf_paths.append(pdf_path)
                    progress_bar.progress((i+1)/len(relevant_papers))
                    time.sleep(0.5)  # avoid rate-limit
                    update_log(f"Processed paper {i+1}/{len(relevant_papers)}")
                
                df = pd.DataFrame(relevant_papers)
                st.subheader("Papers (Relevance > 30%)")
                st.dataframe(df[["id","title","year","categories","abstract_highlighted","matched_terms","relevance_prob","download_status"]], use_container_width=True)

                # Output Formats
                if "SQLite (.db)" in output_formats:
                    status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), [])
                    st.info(status)
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    st.download_button("Download CSV", csv, "coreshellnanoparticles_papers.csv", "text/csv")
                if "JSON" in output_formats:
                    json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                    st.download_button("Download JSON", json_data, "coreshellnanoparticles_papers.json", "application/json")

                # PDF Downloads
                if pdf_paths:
                    st.subheader("Individual PDF Downloads")
                    download_pdf_buttons(pdf_paths)
                    zip_path = create_pdf_zip_safe(pdf_paths)
                    download_zip_button(zip_path)

                display_logs()
