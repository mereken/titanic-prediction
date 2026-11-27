import base64
import os
import time
from pathlib import Path

import requests
import streamlit as st

# Address of the FastAPI service (can be overridden with an environment variable)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
BASE_DIR = Path(__file__).parent


def html(markup: str) -> str:
    """Collapse HTML into one line: Markdown treats indented or blank lines as code."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon=":material/directions_boat:",
)


# ---------- Background image ----------
@st.cache_data
def load_background(filename: str) -> str:
    path = BASE_DIR / "assets" / filename
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


bg = load_background("background.jpg")
bg_css = f"url(data:image/jpeg;base64,{bg})" if bg else "none"
st.markdown(f"<style>:root {{ --bg-image: {bg_css}; }}</style>", unsafe_allow_html=True)

# ---------- Styling ----------
st.markdown(
        html("""
    <style>
    .stApp {
        background-image:
            linear-gradient(180deg, rgba(8,20,34,.45) 0%, rgba(8,20,34,.85) 65%),
            var(--bg-image);
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
    }
    header[data-testid="stHeader"] { background: transparent; }
    /* hide the small "Running..." indicator in the top-right corner */
    [data-testid="stStatusWidget"] { display: none; }

    .hero {
        background: rgba(8,20,34,.65);
        backdrop-filter: blur(4px);
        border: 1px solid #C9A24B;
        border-radius: 14px;
        padding: 26px 24px;
        text-align: center;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: #C9A24B;
        font-family: Georgia, serif;
        letter-spacing: 4px;
        margin: 8px 0 0 0;
        padding: 0;
    }
    .hero p { color: #CFE0F0; font-style: italic; margin: 8px 0 0 0; }

    .st-key-boarding {
        background: rgba(11,31,51,.78);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(201,162,75,.55);
        border-radius: 14px;
        padding: 18px 20px;
    }

    /* loading indicator: centered and clearly visible */
    div[data-testid="stSpinner"] {
        justify-content: center;
        padding: 18px 0;
        font-size: 1.2rem;
        color: #F0D58A;
    }

    .result {
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin-top: 18px;
        border: 2px solid;
        backdrop-filter: blur(6px);
    }
    .result.ok  { background: rgba(20,80,55,.7);  border-color: #2EA06E; }
    .result.bad { background: rgba(100,30,30,.7); border-color: #C84646; }
    .result .verdict { font-size: 1.05rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }
    .result .big { font-size: 3.2rem; font-weight: 700; font-family: Georgia, serif; }
    .bar { height: 12px; border-radius: 6px; background: rgba(255,255,255,.15); overflow: hidden; margin-top: 12px; }
    .bar > div { height: 100%; background: linear-gradient(90deg, #C9A24B, #F0D58A); }
    .footer { text-align: center; color: #9DB4C9; font-size: .8rem; margin-top: 30px; }
    </style>
    """),
        unsafe_allow_html=True,
    )

SHIP_SVG = """
<svg width="64" height="64" viewBox="0 0 48 48" fill="none" stroke="#C9A24B"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="17" y="12" width="4" height="11"/>
  <rect x="23" y="12" width="4" height="11"/>
  <rect x="29" y="12" width="4" height="11"/>
  <rect x="13" y="23" width="24" height="6"/>
  <path d="M5 29h40l-5 9H10z"/>
  <path d="M3 43q4-3 8 0t8 0 8 0 8 0 8 0"/>
</svg>
"""

LIFEBUOY_SVG = """
<svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="#5FD8A0" stroke-width="3">
  <circle cx="24" cy="24" r="18"/><circle cx="24" cy="24" r="7"/>
  <path d="M11 11l8 8M29 29l8 8M37 11l-8 8M19 29l-8 8"/>
</svg>
"""

WAVES_SVG = """
<svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="#FF8A8A"
     stroke-width="3" stroke-linecap="round">
  <path d="M4 16q5-6 10 0t10 0 10 0 10 0"/>
  <path d="M4 26q5-6 10 0t10 0 10 0 10 0"/>
  <path d="M4 36q5-6 10 0t10 0 10 0 10 0"/>
</svg>
"""

st.markdown(
        html(f"""
    <div class="hero">
        {SHIP_SVG}
        <h1>RMS TITANIC</h1>
        <p>Would you have survived the maiden voyage of April 1912?</p>
    </div>
    """),
        unsafe_allow_html=True,
    )

# ---------- Boarding pass form ----------
CLASSES = {1: "1st class", 2: "2nd class", 3: "3rd class"}
PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
# Typical (median) ticket prices in the dataset, in pounds
TYPICAL_FARE = {1: 60, 2: 14, 3: 8}


def on_class_change():
    """When the class changes, set the price to a typical value for that class."""
    st.session_state["fare"] = TYPICAL_FARE[st.session_state["pclass"]]


if "fare" not in st.session_state:
    st.session_state["fare"] = TYPICAL_FARE[3]

with st.container(border=True, key="boarding"):
    st.markdown("### :material/confirmation_number: Boarding pass")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox(
            ":material/airline_seat_recline_extra: Ticket class",
            list(CLASSES),
            index=2,
            format_func=lambda k: CLASSES[k],
            key="pclass",
            on_change=on_class_change,
        )
        sex = st.selectbox(":material/wc: Sex", ["male", "female"], format_func=str.capitalize)
        age = st.slider(":material/cake: Age", 0, 100, 30)
        embarked = st.selectbox(
            ":material/location_on: Port of embarkation", list(PORTS), format_func=lambda k: PORTS[k]
        )
    with col2:
        sibsp = st.number_input(":material/group: Siblings / spouses aboard", 0, 10, 0)
        parch = st.number_input(":material/family_restroom: Parents / children aboard", 0, 10, 0)
        fare = st.slider(
            ":material/payments: Ticket price (£)",
            min_value=0,
            max_value=300,
            step=1,
            key="fare",
            help="The price paid for the ticket in 1912 pounds sterling. "
                 "It is set automatically to a typical price for the chosen class, "
                 "but you can change it.",
        )

    predict = st.button("Check my fate", type="primary", icon=":material/anchor:")

# ---------- Prediction ----------
if predict:
    payload = {
        "pclass": int(pclass),
        "sex": sex,
        "age": float(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": embarked,
    }
    result, error = None, None
    with st.spinner(":material/hourglass_top: Consulting the ship's records..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as exc:
            error = exc
        time.sleep(0.7)  # short pause so the loading indicator is visible

    if error:
        st.error(
            f"Could not get a prediction from the API ({API_URL}). Is it running?\n\n{error}",
            icon=":material/error:",
        )
    else:
        prob = result["survival_probability"]
        if result["survived"]:
            css, icon, verdict = "ok", LIFEBUOY_SVG, "You made it to a lifeboat"
        else:
            css, icon, verdict = "bad", WAVES_SVG, "You went down with the ship"
        st.markdown(
        html(f"""
            <div class="result {css}">
                {icon}
                <div class="verdict">{verdict}</div>
                <div class="big">{prob:.0%}</div>
                <div>estimated chance of survival</div>
                <div class="bar"><div style="width:{prob * 100:.0f}%"></div></div>
            </div>
            """),
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Random Forest trained on the Titanic dataset · served by FastAPI · tracked with MLflow</div>',
    unsafe_allow_html=True,
)
