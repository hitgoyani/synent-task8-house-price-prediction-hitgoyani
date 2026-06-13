import json
from pathlib import Path
import joblib
import streamlit as st

# Explicit ordered feature schema
FEATURE_COLS = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude',
    'rooms_per_bedrooms'
]

# Fixed exchange rate constant
USD_TO_INR = 85.0

@st.cache_resource
def load_model():
    """Serialize and load the trained model."""
    model_path = Path(__file__).resolve().parent.parent / 'model' / 'model.pkl'
    return joblib.load(model_path)

@st.cache_data
def load_model_info():
    """Load model metadata from JSON with graceful fallback."""
    json_path = Path(__file__).resolve().parent.parent / 'model' / 'model_info.json'
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            return {
                'total_records': info.get('total_records', 20640),
                'r2_score': info.get('R2_score', 0.81),
                'best_model': info.get('best_model', 'Random Forest Regressor')
            }
    except Exception:
        # Graceful fallback
        return {
            'total_records': 20640,
            'r2_score': 0.81,
            'best_model': 'Random Forest Regressor'
        }

def format_inr(amount_usd):
    """
    Convert USD to INR and format using the Indian numbering system.
    Returns a dictionary with structured output formats.
    """
    raw_inr = amount_usd * USD_TO_INR
    
    # Format the raw commatized string according to Indian Numbering System
    s, *d = str(int(raw_inr)).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    raw_inr_string = f"₹{r}"
    
    # Format for display (Lakh/Crore)
    if raw_inr >= 10000000:
        display = f"₹{raw_inr/10000000:.2f} Crore"
    elif raw_inr >= 100000:
        display = f"₹{raw_inr/100000:.2f} Lakh"
    else:
        display = f"₹{raw_inr:,.0f}"
        
    return {
        'display': display,
        'raw_inr_string': raw_inr_string,
        'usd_fmt': f"${amount_usd:,.0f}"
    }

def inject_css():
    """Inject premium CSS styling directly into the Streamlit app."""
    css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Metric Cards Styling */
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            padding: 1rem;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            border-color: #4CAF50;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Highlight specific metric text */
        div[data-testid="stMetricValue"] {
            color: #4CAF50 !important;
        }
        
        /* Hide structural headers and footers */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Hide default Streamlit multi-page sidebar navigation */
        div[data-testid="stSidebarNav"] {display: none !important;}
        
        /* Hide Streamlit top decoration line */
        div[data-testid="stDecoration"] {display: none !important;}
        
        /* Compact info boxes */
        div[data-testid="stAlert"] {
            padding: 0.5rem 1rem !important;
        }
        div[data-testid="stAlert"] p {
            font-size: 0.85rem !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_sidebar():
    """Render the consistent dynamic sidebar across all pages."""
    model_info = load_model_info()
    
    with st.sidebar:
        st.markdown("### Model Engine")
        st.markdown("**Architecture:**")
        st.caption(model_info['best_model'])
        
        st.metric("R² Accuracy", f"{model_info['r2_score'] * 100:.2f}%")
        st.metric("Active Currency", "INR (₹)")
        st.markdown("---")
        
        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/1_Predictor.py", label="🧮 Predictor")
