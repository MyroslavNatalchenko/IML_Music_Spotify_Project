import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hit Predictor", layout="wide", page_icon="🎵")
st.title("🎧 Music Hit Predictor")
st.markdown("---")

@st.cache_data
def load_initial_data():
    try:
        meta_resp = requests.get(f"{API_URL}/meta/info")
        models_resp = requests.get(f"{API_URL}/models")

        if meta_resp.status_code == 200 and models_resp.status_code == 200:
            return {
                "lists": meta_resp.json(),
                "models": models_resp.json()["models"]
            }
    except requests.exceptions.ConnectionError:
        return None
    return None


data = load_initial_data()

if not data:
    st.error("❌ Backend is offline. Run 'uvicorn main:app --reload' in backend folder.")
    st.stop()

genres_list = data["lists"]["genres"]
available_models = data["models"]

input_col, result_col = st.columns([0.7, 0.3], gap="medium")

with input_col:
    st.subheader("🛠 Track Parameters")
    with st.form("main_form"):
        st.markdown("##### 1. General Info")

        c1, c2, c3 = st.columns(3)
        with c1:
            genre = st.selectbox("Genre", options=genres_list)
        with c2:
            duration_ms = st.number_input("Duration (ms)", value=200000, step=5000)
        with c3:
            st.write("")
            st.write("")
            explicit = st.checkbox("Explicit Content?", value=False)

        st.divider()

        st.markdown("##### 2. Vibe & Mood")
        with st.expander("Adjust Audio Features", expanded=True):
            ac1, ac2 = st.columns(2)
            with ac1:
                danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.65)
                energy = st.slider("⚡ Energy", 0.0, 1.0, 0.70)
                valence = st.slider("🌞 Positivity", 0.0, 1.0, 0.5)
                loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0)
            with ac2:
                acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.1)
                instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0)
                liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.1)
                speechiness = st.slider("🗣 Speechiness", 0.0, 1.0, 0.05)

        st.divider()

        st.markdown("##### 3. Technical Specs")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            tempo = st.number_input("BPM", value=120.0, step=1.0)
        with tc2:
            key = st.selectbox("Key", range(12))
        with tc3:
            mode = st.radio("Mode", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor", horizontal=True)
            time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=1)

        st.write("")

        predict_mode = st.radio("Prediction Mode", ["Compare All Models", "Specific Model"], horizontal=True)

        selected_model = None
        if predict_mode == "Specific Model":
            selected_model = st.selectbox("Choose Model", available_models)

        submit_btn = st.form_submit_button("🚀 Run Prediction", type="primary", use_container_width=True)

with result_col:
    st.subheader("📊 Results")
    placeholder = st.empty()

    if not submit_btn:
        placeholder.info("👈 Configure and click Predict")
    else:
        payload = {
            "track_genre": genre,
            "duration_ms": duration_ms,
            "explicit": explicit,
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "time_signature": time_signature
        }

        with st.spinner("Processing..."):
            try:
                results_to_display = {}

                if predict_mode == "Compare All Models":
                    resp = requests.post(f"{API_URL}/models/all/predict", json=payload)
                    if resp.status_code == 200:
                        results_to_display = resp.json()
                    else:
                        st.error(resp.text)

                else:
                    resp = requests.post(f"{API_URL}/models/{selected_model}/predict", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        results_to_display = {data["model"]: data["popularity_score"]}
                    else:
                        st.error(resp.text)

                placeholder.empty()
                if results_to_display:
                    for m_name, score in results_to_display.items():
                        clean_name = m_name.replace("_model", "").replace("_", " ").title()

                        with st.container(border=True):
                            st.write(f"**{clean_name}**")
                            if score is not None:
                                st.metric("Score", f"{score:.1f}")
                                st.progress(score / 100)
                            else:
                                st.error("Error")

            except Exception as e:
                st.error(f"Connection Error: {e}")