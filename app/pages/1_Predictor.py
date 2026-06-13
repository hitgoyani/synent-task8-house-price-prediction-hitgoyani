import streamlit as st
import pandas as pd
from utils import load_model, format_inr, FEATURE_COLS, inject_css, render_sidebar

# Apply styling
st.set_page_config(page_title='House Price Calculator', page_icon='🧮', layout='centered')
inject_css()

# Render Dynamic Sidebar
render_sidebar()

st.title('Interactive House Price Calculator')
st.markdown('**Fixed Conversion Rate:** ₹85.00 / $1 USD')
st.markdown('---')

# Load model
model = load_model()

# Header layout
st.subheader('Property Parameters')

col1, col2 = st.columns(2)

with col1:
    st.markdown('**Location Grid**')
    latitude = st.number_input(
        'Latitude',
        min_value=32.0, max_value=42.0,
        value=34.05, step=0.01,
        help="34.05 = LA | 37.77 = SF"
    )
    longitude = st.number_input(
        'Longitude',
        min_value=-125.0, max_value=-114.0,
        value=-118.24, step=0.01,
        help="-118.24 = LA | -122.41 = SF"
    )
    
    st.markdown('**Demographic Context**')
    med_income = st.number_input(
        'Median Household Income (10k $)',
        min_value=0.5, max_value=15.0,
        value=5.0, step=0.1
    )
    population = st.number_input(
        'Block Population',
        min_value=1, max_value=10000,
        value=1200, step=10
    )
    ave_occupancy = st.number_input(
        'Avg Household Occupancy',
        min_value=0.5, max_value=20.0,
        value=3.0, step=0.1
    )

with col2:
    st.markdown('**Property Matrix**')
    house_age = st.slider(
        'Property Age (years)', 1, 52, 20
    )
    ave_rooms = st.number_input(
        'Average Rooms',
        min_value=1.0, max_value=20.0,
        value=6.0, step=0.1
    )
    ave_bedrooms = st.number_input(
        'Average Bedrooms',
        min_value=0.5, max_value=10.0,
        value=1.0, step=0.1
    )

st.markdown('---')

# Prediction workflow with session state protection
if st.button('Calculate House Price', type='primary', use_container_width=True):
    try:
        # Safe division
        rooms_per_bedrooms = ave_rooms / ave_bedrooms if ave_bedrooms > 0 else 0.0

        # Enforce exact column ordering matching FEATURE_COLS
        input_df = pd.DataFrame([[
            med_income, house_age, ave_rooms, ave_bedrooms,
            population, ave_occupancy, latitude, longitude,
            rooms_per_bedrooms
        ]], columns=FEATURE_COLS)

        # Deterministic inference execution
        prediction = model.predict(input_df)[0]
        price_usd = prediction * 100_000
        
        # Localize currency
        formatted_vals = format_inr(price_usd)
        
        st.session_state['pred_display'] = formatted_vals['display']
        st.session_state['pred_raw'] = formatted_vals['raw_inr_string']
        st.session_state['pred_usd'] = formatted_vals['usd_fmt']
        st.session_state['calculated'] = True
        
    except Exception as e:
        st.error(f'Calculation error: {e}')

if st.session_state.get('calculated', False):
    st.success(f"### Estimated Value: {st.session_state['pred_display']}")
    
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Raw INR Base", st.session_state['pred_raw'])
    with mc2:
        st.metric("Original Dataset Value (USD)", st.session_state['pred_usd'])
        
    st.info("ℹ️ **Data Limitation Note**: The California Housing dataset imposes a hard ceiling of $500,001 (approx. ₹4.25 Crore). Predictions nearing this threshold may exhibit artificial clustering.")
    
    if st.button('Calculate Again', use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown('<br><hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #888888; font-size: 0.8em;">Built by Hit Goyani | Synent Technologies Data Science Internship | Candidate ID: SYN/M2/IP1050</div>', unsafe_allow_html=True)
