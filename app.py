"""Home page for the PerTurbo power calculator."""
import streamlit as st
st.set_page_config(page_title="PerTurbo Power Calculator", layout="wide")
st.title("PerTurbo Power Calculator")
st.caption("Analyses 1–2 need a baseline from Data Setup. Screen-wide sensitivity opens directly.")
if st.button("Data Setup", type="primary"):
    st.switch_page("pages/0_Data_Setup.py")
st.header("Choose a question")
cards = [
    ("3 · Screen-wide sensitivity", "How does detection across a whole screen change with cell numbers and sequencing depth?", "Overview first, then inspect an individual perturbation–gene pair.", "pages/analysis3_screen_power.py"),
    ("2 · Target program", "What fraction of a target-gene program could reach the chosen power?", "Use a built-in K562 TF program, a CollecTRI regulon, or your own target list.", "pages/analysis2_target_program.py"),
    ("1 · Nearby genes", "Which nearby genes could be detected if their expression changed?", "Explore Gasperini cis pairs, then inspect a gene and enhancer.", "pages/analysis1_cis_power.py"),
]
for column, (title, question, caption, _) in zip(st.columns(3), cards):
    with column:
        st.subheader(title)
        st.write(question)
for column, (title, _, _, page) in zip(st.columns(3), cards):
    if column.button("Open Analysis " + title[0], width="stretch"):
        st.switch_page(page)
