import streamlit as st
from utils import inject_css, load_model_info, render_sidebar, FEATURE_COLS

# Configure page
st.set_page_config(page_title='House Price Predictor', page_icon='🏠', layout='centered')
inject_css()
model_info = load_model_info()

# Render Dynamic Sidebar
render_sidebar()

# Main Landing Content
st.title("California House Price Predictor")
st.markdown("---")
# Hero Unit
st.markdown("""
Welcome to the House Price Prediction Engine. This application leverages advanced machine learning to evaluate regional parameters and estimate property values, dynamically localized into Indian Rupees (INR).
""")

# Dynamic Metrics Grid
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Training Records', f"{model_info['total_records']:,}")
with col2:
    st.metric('Active Features', len(FEATURE_COLS))
with col3:
    st.metric('Baseline Accuracy', f"{model_info['r2_score'] * 100:.2f}%")

st.markdown("---")

# Workflow & CTA
st.markdown("### Procedural Workflow")
st.markdown("""
1. **Inputs:** Define geographic and demographic parameters.
2. **Evaluation:** Process data through the trained baseline model.
3. **Localization:** Convert predicted USD base to local INR mapping.
""")

st.markdown("<br>", unsafe_allow_html=True)
st.page_link("pages/1_Predictor.py", label="🚀 Start Predicting", use_container_width=True)
st.markdown('<br><hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888888; font-size: 0.8em;">Built by Hit Goyani | Synent Technologies Data Science Internship | Candidate ID: SYN/M2/IP1050</div>', unsafe_allow_html=True)
