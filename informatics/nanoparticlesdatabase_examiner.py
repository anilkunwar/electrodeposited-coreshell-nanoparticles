import sqlite3
import streamlit as st
import re
import pandas as pd
from collections import Counter
import numpy as np
import os
from datetime import datetime
import logging
from transformers import AutoModel, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wordcloud import WordCloud

# ===== CONFIGURATION =====
# Use temporary directory for cloud environments
if os.path.exists("/tmp"):
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

METADATA_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'database_inspector.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit configuration
st.set_page_config(page_title="Core-Shell Nanoparticles Database Inspector", layout="wide")
st.title("Core-Shell Nanoparticles Database Inspector")
st.markdown("""
This tool inspects SQLite databases (`coreshellnanoparticles_metadata.db` and `coreshellnanoparticles_universe.db`), displays paper metadata, full text content, and extracts/suggests fields of interest (e.g., shell diameter, core diameter, resistivity, thermal stability) using regex and SciBERT-based semantic analysis.
""")

# Session state initialization
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "extracted_fields" not in st.session_state:
    st.session_state.extracted_fields = {}
if "db_file" not in st.session_state:
    st.session_state.db_file = UNIVERSE_DB_FILE if os.path.exists(UNIVERSE_DB_FILE) else None
if "table_name" not in st.session_state:
    st.session_state.table_name = "papers"

def update_log(message):
    """Update log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(log_entry)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# ===== DATABASE CONNECTION =====
def connect_to_db(db_file):
    """Connect to SQLite database"""
    try:
        conn = sqlite3.connect(db_file)
        update_log(f"Connected to database: {db_file}")
        return conn
    except Exception as e:
        update_log(f"Failed to connect to {db_file}: {str(e)}")
        st.error(f"Failed to connect to {db_file}: {str(e)}")
        return None

# ===== DATABASE INSPECTION =====
def inspect_database(db_file):
    """Inspect database schema and tables"""
    conn = connect_to_db(db_file)
    if not conn:
        return None, None, []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        if not tables:
            update_log(f"No tables found in {db_file}")
            st.warning(f"No tables found in {db_file}")
            conn.close()
            return None, None, []
        
        st.subheader(f"Tables in {os.path.basename(db_file)}")
        st.write(tables)
        
        table_name = st.session_state.table_name
        if table_name not in tables:
            update_log(f"Table '{table_name}' not found in {db_file}")
            st.warning(f"Table '{table_name}' not found in {db_file}. Available tables: {', '.join(tables)}")
            conn.close()
            return None, None, tables
        
        # Get schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        schema_df = pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "dflt_value", "pk"])
        available_columns = [col[1] for col in schema]
        update_log(f"Available columns in '{table_name}' table: {', '.join(available_columns)}")
        
        # Fetch papers
        select_columns = ["id", "title", "year"] if "id" in available_columns and "title" in available_columns else available_columns[:3]
        if "content" in available_columns:
            select_columns.append("substr(content, 1, 200) as sample_content")
        if "abstract" in available_columns:
            select_columns.append("substr(abstract, 1, 200) as sample_abstract")
        if "relevance_prob" in available_columns:
            select_columns.append("relevance_prob")
        
        query = f"SELECT {', '.join(select_columns)} FROM {table_name} LIMIT 10"
        try:
            df = pd.read_sql_query(query, conn)
            update_log(f"Fetched {len(df)} sample papers from {table_name} in {db_file}")
        except Exception as e:
            update_log(f"Failed to fetch papers from {table_name} in {db_file}: {str(e)}")
            st.error(f"Failed to fetch papers: {str(e)}")
            df = pd.DataFrame()
        
        conn.close()
        return df, schema_df, tables
    except Exception as e:
        update_log(f"Error inspecting {db_file}: {str(e)}")
        st.error(f"Error inspecting {db_file}: {str(e)}")
        conn.close()
        return None, None, []

# ===== FIELD EXTRACTION AND SUGGESTION =====
FIELDS = {
    "shell diameter": r"(?:shell\s*diameter|diameter\s*of\s*shell|Ag\s*shell\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "core diameter": r"(?:core\s*diameter|diameter\s*of\s*core|Cu\s*core\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "shell thickness": r"(?:shell\s*thickness|thickness\s*of\s*shell|Ag\s*shell\s*thickness)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "resistivity": r"(?:electric\s*resistivity|resistivity)\s*[:=]?\s*([\d\.eE\-]+)\s*(Ω·m|ohm·m|Ω\s*m|ohm\s*m|μΩ·cm|microohm·cm)",
    "thermal stability": r"(?:thermal\s*stability|stability\s*temperature)\s*[:=]?\s*([\d\.]+)\s*(°C|Celsius|K|Kelvin)",
    "deposition time": r"(?:electroless\s*deposition\s*time|deposition\s*time)\s*[:=]?\s*([\d\.]+)\s*(min|minute|hr|hour|s|second)",
    "particle size": r"(?:particle\s*size|nanoparticle\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)"
}

# Load SciBERT model
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
    update_log("Loaded SciBERT model")
except Exception as e:
    st.warning(f"Failed to load SciBERT: {e}. Falling back to regex-only extraction. Install: `pip install transformers torch`")
    scibert_model = None
    scibert_tokenizer = None

@st.cache_data(hash_funcs={str: lambda x: x})
def get_scibert_embedding(text):
    if not scibert_model or not text.strip():
        return None
    try:
        inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        norm = np.linalg.norm(last_hidden_state)
        if norm == 0:
            update_log(f"Zero norm for embedding of '{text}'")
            return None
        return last_hidden_state / norm
    except Exception as e:
        update_log(f"SciBERT embedding failed for '{text}': {str(e)}")
        return None

def extract_fields_from_text(text, similarity_threshold=0.7):
    """Extract predefined fields and use SciBERT for semantic validation"""
    extracted = {}
    for field, pattern in FIELDS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if scibert_model:
                # Validate matches with SciBERT
                field_embedding = get_scibert_embedding(field)
                if field_embedding is not None:
                    valid_matches = []
                    for match in matches:
                        context = f"{match[0]} {match[1]}"
                        context_embedding = get_scibert_embedding(context)
                        if context_embedding is not None:
                            similarity = np.dot(field_embedding, context_embedding) / (np.linalg.norm(field_embedding) * np.linalg.norm(context_embedding))
                            if similarity > similarity_threshold:
                                valid_matches.append(match)
                    extracted[field] = list(set(valid_matches))
                else:
                    extracted[field] = list(set(matches))
            else:
                extracted[field] = list(set(matches))
    return extracted

def suggest_new_fields(text, min_freq=3, similarity_threshold=0.7):
    """Suggest new fields based on frequent phrases near numbers"""
    patterns = re.findall(r"(\w+(?:\s+\w+)?)\s*[:=]\s*([\d\.eE\-]+)\s*(\w+)", text, re.IGNORECASE)
    counter = Counter([match[0].lower() for match in patterns])
    suggested = []
    for field, count in counter.most_common():
        if count >= min_freq and field not in FIELDS:
            if scibert_model:
                field_embedding = get_scibert_embedding(field)
                if field_embedding is not None:
                    # Check similarity to existing fields
                    max_similarity = 0
                    for known_field in FIELDS:
                        known_embedding = get_scibert_embedding(known_field)
                        if known_embedding is not None:
                            similarity = np.dot(field_embedding, known_embedding) / (np.linalg.norm(field_embedding) * np.linalg.norm(known_embedding))
                            max_similarity = max(max_similarity, similarity)
                    if max_similarity < similarity_threshold:
                        suggested.append(field)
            else:
                suggested.append(field)
    return suggested[:10]

@st.cache_data
def plot_field_histogram(extracted_fields, colormap="viridis"):
    """Plot histogram of extracted field values"""
    values = []
    labels = []
    units = []
    for field, matches in extracted_fields.items():
        for value, unit in matches:
            try:
                values.append(float(value))
                labels.append(field)
                units.append(unit)
            except ValueError:
                continue
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    colors = [cm.get_cmap(colormap)(i / len(unique_labels)) for i in range(len(unique_labels))]
    ax.hist([values[i] for i in range(len(values)) if labels[i] in unique_labels], 
            bins=20, label=[f"{label} ({units[i]})" for i, label in enumerate(labels)], 
            color=colors[:len(unique_labels)], alpha=0.7)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Extracted Field Values")
    ax.legend()
    plt.tight_layout()
    return fig

@st.cache_data
def plot_field_wordcloud(suggested_fields, max_font_size=40, colormap="viridis"):
    """Plot word cloud of suggested fields"""
    if not suggested_fields:
        return None
    term_dict = {field: count for field, count in Counter(suggested_fields).items()}
    wordcloud = WordCloud(width=800, height=400, background_color="white", 
                         min_font_size=8, max_font_size=max_font_size, 
                         colormap=colormap, max_words=20).generate_from_frequencies(term_dict)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Suggested Fields")
    plt.tight_layout()
    return fig

# ===== MAIN UI =====
st.header("Select or Upload Database")
db_files = [METADATA_DB_FILE, UNIVERSE_DB_FILE] if os.path.exists(METADATA_DB_FILE) or os.path.exists(UNIVERSE_DB_FILE) else []
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
default_index = db_options.index(os.path.basename(UNIVERSE_DB_FILE)) if os.path.basename(UNIVERSE_DB_FILE) in db_options else 0
db_selection = st.selectbox("Select Database", db_options, index=default_index, key="db_select")

if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
        update_log(f"Uploaded database saved as {temp_db_path}")
else:
    st.session_state.db_file = os.path.join(DB_DIR, db_selection)
    update_log(f"Selected database: {db_selection}")

if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    # Inspect database
    if st.button("Inspect Database", key="inspect_button"):
        with st.spinner(f"Inspecting {os.path.basename(st.session_state.db_file)}..."):
            df, schema_df, available_tables = inspect_database(st.session_state.db_file)
            if available_tables and st.session_state.table_name not in available_tables:
                st.session_state.table_name = st.selectbox("Select Table", available_tables, key="table_select")
            if schema_df is not None:
                st.subheader("Schema of Selected Table")
                st.dataframe(schema_df[["name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)
            if not df.empty:
                st.subheader("Sample Papers (First 10)")
                st.dataframe(df, use_container_width=True)
                st.session_state.papers_df = df
            else:
                st.warning("No papers found in the selected table.")
    
    # Select paper for analysis
    if "papers_df" in st.session_state and not st.session_state.papers_df.empty:
        st.subheader("Analyze Paper")
        selected_id = st.selectbox("Select Paper ID to Inspect", st.session_state.papers_df["id"].tolist(), key="paper_select")
        
        # Fetch full content
        conn = connect_to_db(st.session_state.db_file)
        if conn:
            query = f"SELECT content, title FROM {st.session_state.table_name} WHERE id = ?"
            try:
                content_df = pd.read_sql_query(query, conn, params=(selected_id,))
                if not content_df.empty:
                    content = content_df.iloc[0]["content"]
                    title = content_df.iloc[0]["title"]
                    if content and not content.startswith("Error"):
                        st.subheader(f"Full Text for Paper {selected_id}: {title}")
                        st.text_area("Content Preview (first 2000 chars)", 
                                     content[:2000] + "..." if len(content) > 2000 else content, 
                                     height=300, key=f"content_preview_{selected_id}")
                        
                        # Extract fields
                        if st.button("Extract Fields", key=f"extract_fields_{selected_id}"):
                            extracted = extract_fields_from_text(content)
                            st.session_state.extracted_fields[selected_id] = extracted
                            update_log(f"Extracted fields for paper {selected_id}")
                        
                        # Display extracted fields
                        if selected_id in st.session_state.extracted_fields:
                            st.subheader("Extracted Fields")
                            extracted_df = pd.DataFrame(list(st.session_state.extracted_fields[selected_id].items()), 
                                                      columns=["Field", "Values"])
                            st.dataframe(extracted_df, use_container_width=True)
                            csv_data = extracted_df.to_csv(index=False)
                            st.download_button("Download Extracted Fields CSV", csv_data, 
                                             f"fields_{selected_id}.csv", "text/csv", 
                                             key=f"download_fields_{selected_id}")
                            fig_hist = plot_field_histogram(st.session_state.extracted_fields[selected_id])
                            if fig_hist:
                                st.pyplot(fig_hist)
                        
                        # Suggest new fields
                        if st.button("Suggest New Fields", key=f"suggest_fields_{selected_id}"):
                            suggested = suggest_new_fields(content)
                            if suggested:
                                st.subheader("Suggested New Fields")
                                st.write(", ".join(suggested))
                                update_log(f"Suggested new fields for paper {selected_id}: {', '.join(suggested)}")
                                fig_wc = plot_field_wordcloud(suggested)
                                if fig_wc:
                                    st.pyplot(fig_wc)
                            else:
                                st.info("No new fields suggested.")
                    else:
                        st.warning("No content available for this paper.")
                else:
                    st.warning("Paper not found in the selected table.")
            except Exception as e:
                update_log(f"Failed to fetch content for paper {selected_id}: {str(e)}")
                st.error(f"Failed to fetch content: {str(e)}")
            conn.close()
    
    # Sidebar for parameters
    with st.sidebar:
        st.subheader("Analysis Parameters")
        similarity_threshold = st.slider("Similarity Threshold (SciBERT)", 0.5, 0.9, 0.7, 0.05, key="similarity_threshold")
        min_freq = st.slider("Minimum Frequency for Suggested Fields", 1, 10, 3, key="min_freq")
        colormap = st.selectbox("Colormap for Visualizations", 
                               ["viridis", "plasma", "inferno", "magma", "hot"], key="colormap")
    
    # Display logs
    st.subheader("Logs")
    st.text_area("Recent Logs", "\n".join(st.session_state.log_buffer), height=200, key="logs_display")
else:
    st.warning("Select or upload a valid database file.")

# Footer
st.markdown("---")
st.markdown("*Database Inspector - Extracts and suggests fields like shell diameter, resistivity using regex and SciBERT*")

