```python
import sqlite3
import streamlit as st
import re
import pandas as pd
from collections import Counter
import numpy as np
import os
from datetime import datetime
import logging

# ===== CONFIGURATION =====
# Use temporary directory for cloud environments
if os.path.exists("/tmp"):  # Cloud environments typically have /tmp
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

METADATA_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "coreshellnanoparticles_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'database_inspector.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Core-Shell Nanoparticles Database Inspector", layout="wide")
st.title("Core-Shell Nanoparticles Database Inspector")
st.markdown("""
This tool inspects the databases (`coreshellnanoparticles_metadata.db` and `coreshellnanoparticles_universe.db`), displays paper metadata, full text content, and extracts/suggests fields of interest such as shell diameter, core diameter, resistivity, thermal stability, etc., from the full text using pattern matching and frequency analysis.
""")

# Session state for logs and extracted data
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "extracted_fields" not in st.session_state:
    st.session_state.extracted_fields = {}

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

# ===== FETCH PAPERS =====
def fetch_papers(db_file):
    """Fetch papers from the database"""
    conn = connect_to_db(db_file)
    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM papers", conn)
            update_log(f"Fetched {len(df)} papers from {db_file}")
            conn.close()
            return df
        except Exception as e:
            update_log(f"Failed to fetch papers from {db_file}: {str(e)}")
            st.error(f"Failed to fetch papers from {db_file}: {str(e)}")
        finally:
            conn.close()
    return pd.DataFrame()

# Fetch papers from both databases
metadata_df = fetch_papers(METADATA_DB_FILE)
universe_df = fetch_papers(UNIVERSE_DB_FILE)

# ===== FIELDS OF INTEREST =====
# Predefined fields with patterns (regex to extract values near keywords)
FIELDS = {
    "shell diameter": r"(?:shell\s*diameter|diameter\s*of\s*shell|Ag\s*shell\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "core diameter": r"(?:core\s*diameter|diameter\s*of\s*core|Cu\s*core\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "shell thickness": r"(?:shell\s*thickness|thickness\s*of\s*shell|Ag\s*shell\s*thickness)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)",
    "resistivity": r"(?:electric\s*resistivity|resistivity)\s*[:=]?\s*([\d\.eE\-]+)\s*(Ω·m|ohm·m|Ω\s*m|ohm\s*m|μΩ·cm|microohm·cm)",
    "thermal stability": r"(?:thermal\s*stability|stability\s*temperature)\s*[:=]?\s*([\d\.]+)\s*(°C|Celsius|K|Kelvin)",
    "deposition time": r"(?:electroless\s*deposition\s*time|deposition\s*time)\s*[:=]?\s*([\d\.]+)\s*(min|minute|hr|hour|s|second)",
    "particle size": r"(?:particle\s*size|nanoparticle\s*diameter)\s*[:=]?\s*([\d\.]+)\s*(nm|μm|um|nanometer|micrometer)"
}

# ===== EXTRACT FIELDS FROM TEXT =====
def extract_fields_from_text(text):
    """Extract predefined fields from full text using regex patterns"""
    extracted = {}
    for field, pattern in FIELDS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted[field] = list(set(matches))  # Unique values
    return extracted

def suggest_new_fields(text):
    """Suggest new fields based on frequency of phrases near numbers"""
    patterns = re.findall(r"(\w+(?:\s+\w+)?)\s*[:=]\s*([\d\.eE\-]+)\s*(\w+)", text, re.IGNORECASE)
    counter = Counter([match[0].lower() for match in patterns])
    suggested = [field for field, count in counter.most_common(10) if count > 1 and field not in FIELDS]
    return suggested

# ===== MAIN UI =====
if not metadata_df.empty:
    st.subheader("Papers in Metadata Database")
    st.dataframe(metadata_df[["id", "title", "year", "relevance_prob", "download_status"]])

    # Select paper to inspect
    selected_id = st.selectbox("Select Paper ID to Inspect", metadata_df["id"].tolist(), key="paper_select")

    if selected_id:
        # Get full content (prefer universe db if available)
        content_row = universe_df[universe_df["id"] == selected_id]
        if content_row.empty:
            content_row = metadata_df[metadata_df["id"] == selected_id]
        
        if not content_row.empty:
            content = content_row.iloc[0]["content"]
            if content and not content.startswith("Error"):
                st.subheader(f"Full Text for Paper {selected_id}")
                st.text_area("Content Preview (first 2000 chars)", content[:2000] + "..." if len(content) > 2000 else content, height=300, key=f"content_preview_{selected_id}")
                
                # Extract fields
                if st.button("Extract Fields", key=f"extract_fields_{selected_id}"):
                    extracted = extract_fields_from_text(content)
                    st.session_state.extracted_fields[selected_id] = extracted
                    update_log(f"Extracted fields for paper {selected_id}")
                
                # Display extracted fields
                if selected_id in st.session_state.extracted_fields:
                    st.subheader("Extracted Fields")
                    extracted_df = pd.DataFrame(list(st.session_state.extracted_fields[selected_id].items()), columns=["Field", "Values"])
                    st.dataframe(extracted_df)
                
                # Suggest new fields
                if st.button("Suggest New Fields", key=f"suggest_fields_{selected_id}"):
                    suggested = suggest_new_fields(content)
                    if suggested:
                        st.subheader("Suggested New Fields")
                        st.write(", ".join(suggested))
                        update_log(f"Suggested new fields for paper {selected_id}: {', '.join(suggested)}")
                    else:
                        st.info("No new fields suggested.")
            else:
                st.warning("No content available for this paper.")
        else:
            st.warning("Paper not found in databases.")
else:
    st.warning("No papers found in the metadata database.")

# Display logs
st.subheader("Logs")
st.text_area("Recent Logs", "\n".join(st.session_state.log_buffer), height=200, key="logs_display")

# Add footer
st.markdown("---")
st.markdown("*Database Inspector - Extracts and suggests fields from full text content*")
```
