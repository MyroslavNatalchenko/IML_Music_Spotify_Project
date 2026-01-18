import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Hit Predictor", layout="wide", page_icon="🎵")

MODEL_STATS = {
    "Random Forest": {
        "RMSE": 15.1706, "MAE": 10.3634, "R2": 0.5336,
        "Params": {"n_estimators": 300, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "None"}
    },
    "Random Forest (Weighted)": {
        "RMSE": 20.9008, "MAE": 15.7033, "R2": 0.1148,
        "Params": {"n_estimators": 500, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2"}
    },
    "XGBoost": {
        "RMSE": 15.6779, "MAE": 10.4751, "R2": 0.5019,
        "Params": {"n_estimators": 999, "max_depth": 9, "learning_rate": 0.116, "subsample": 0.98}
    },
    "XGBoost (Weighted)": {
        "RMSE": 16.0190, "MAE": 10.9180, "R2": 0.4800,
        "Params": {"n_estimators": 999, "max_depth": 10, "learning_rate": 0.294, "subsample": 0.99}
    },
    "TensorFlow": {
        "RMSE": 19.1617, "MAE": 14.1879, "R2": 0.2560,
        "Params": {"layers": 2, "l1": "64 units", "l2": "96 units (dropout 0.1)", "lr": 0.0005}
    },
    "TensorFlow (Stratified)": {
        "RMSE": 17.0331, "MAE": 12.2528, "R2": 0.3076,
        "Params": {"layers": 1, "l1": "96 units (dropout 0.2)", "lr": 0.0005}
    },
    "TabNet": {
        "RMSE": 18.3855, "MAE": 13.2294, "R2": 0.3150,
        "Params": {"decision_dim": 64, "n_steps": 8, "mask_type": "softmax", "batch_size": 512}
    },
    "TabNet (Stratified)": {
        "RMSE": 16.6083, "MAE": 11.6408, "R2": 0.3417,
        "Params": {"decision_dim": 64, "n_steps": 8, "mask_type": "softmax", "stratified": True}
    }
}

RF_IMPORTANCE_DATA = pd.DataFrame([
    {"Feature": "duration_ms", "Importance": 0.0691},
    {"Feature": "acousticness", "Importance": 0.0688},
    {"Feature": "speechiness", "Importance": 0.0685},
    {"Feature": "loudness", "Importance": 0.0673},
    {"Feature": "valence", "Importance": 0.0672},
    {"Feature": "danceability", "Importance": 0.0666},
    {"Feature": "tempo", "Importance": 0.0665},
    {"Feature": "energy", "Importance": 0.0638},
    {"Feature": "liveness", "Importance": 0.0595},
    {"Feature": "instrumentalness", "Importance": 0.0473},
]).sort_values(by="Importance", ascending=True)

@st.cache_data
def load_initial_data():
    try:
        meta_resp = requests.get(f"{API_URL}/meta/info")
        models_resp = requests.get(f"{API_URL}/models")
        if meta_resp.status_code == 200 and models_resp.status_code == 200:
            return {"lists": meta_resp.json(), "models": models_resp.json()["models"]}
    except:
        return None
    return None

st.title("🎧 Music Hit Predictor")
st.markdown("---")

data = load_initial_data()
if not data:
    st.error("❌ Backend is offline. Run `uvicorn main:app --reload` in backend folder.")
    st.stop()

genres_list = data["lists"]["genres"]
available_models = data["models"]

try:
    dance_index = genres_list.index("dance")
except ValueError:
    dance_index = 0

tab_predict, tab_stats = st.tabs(["🔮 Predictor", "📈 Model Statistics"])

with ((((tab_predict)))):
    with st.container(border=True):
        st.markdown("#### 🎛 Track Configuration")

        c1, c2, c3, c4 = st.columns([2, 1.2, 0.8, 1])
        with c1:
            genre = st.selectbox("Genre", options=genres_list, index=dance_index)
        with c2:
            duration_s = st.number_input("Duration (s)", value=156, step=1)
            duration_ms = duration_s * 1000
        with c3:
            st.write("")
            st.write("")
            explicit = st.checkbox("Explicit", value=False)
        with c4:
            key = st.selectbox("Key", range(12), index=2)

        c1, c2, c3 = st.columns(3)
        with c1:
            tempo = st.number_input("BPM (Tempo)", value=131.12, step=0.01)
        with c2:
            mode = st.radio("Mode", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor", horizontal=True,
                            index=0)
        with c3:
            time_signature = st.selectbox("Time Sig", [3, 4, 5, 6, 7], index=1)

        with st.expander("🎚 Fine-tune Audio Features (Vibe & Mood)", expanded=True):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                danceability = st.slider("💃 Dance", 0.0, 1.0, 0.71)
                energy = st.slider("⚡️ Energy", 0.0, 1.0, 0.47)
            with ac2:
                loudness = st.slider("🔊 Loudness", -60.0, 0.0, -7.38)
                speechiness = st.slider("🗣 Speech", 0.0, 1.0, 0.09)
            with ac3:
                valence = st.slider("🌞 Positivity", 0.0, 1.0, 0.24)
                acousticness = st.slider("🎸 Acoustic", 0.0, 1.0, 0.11)
            with ac4:
                liveness = st.slider("🎤 Live", 0.0, 1.0, 0.27)
                instrumentalness = st.slider("🎹 Instr", 0.0, 1.0, 0.00)

    col_btn, col_res = st.columns([1, 4], gap="large")

    with col_btn:
        st.markdown("##### Action")
        predict_mode = st.radio("Mode", ["Compare All", "Specific"], label_visibility="collapsed")

        selected_model = None
        if predict_mode == "Specific":
            selected_model = st.selectbox("Select Model", available_models, label_visibility="collapsed")

        st.write("")
        run_pred = st.button("🚀 PREDICT", type="primary", width="stretch")

    with col_res:
        st.markdown("##### Results")
        result_area = st.empty()

        if run_pred:
            with result_area.container():
                with st.spinner("Analyzing track DNA..."):
                    payload = {
                        "track_genre": genre, "duration_ms": duration_ms, "explicit": explicit,
                        "danceability": danceability, "energy": energy, "key": key, "loudness": loudness,
                        "mode": mode, "speechiness": speechiness, "acousticness": acousticness,
                        "instrumentalness": instrumentalness, "liveness": liveness, "valence": valence,
                        "tempo": tempo, "time_signature": time_signature
                    }

                    try:
                        results = {}
                        if predict_mode == "Compare All":
                            resp = requests.post(f"{API_URL}/models/all/predict", json=payload)
                            if resp.status_code == 200:
                                results = resp.json()
                            else:
                                st.error(f"Error: {resp.text}")
                        else:
                            resp = requests.post(f"{API_URL}/models/{selected_model}/predict", json=payload)
                            if resp.status_code == 200:
                                d = resp.json()
                                results = {d["model"]: d["popularity_score"]}
                            else:
                                st.error(f"Error: {resp.text}")

                        if results:
                            sorted_res = sorted(results.items(), key=lambda x: x[1] or 0, reverse=True)

                            chunk_size = 4
                            for i in range(0, len(sorted_res), chunk_size):
                                chunk = sorted_res[i:i + chunk_size]
                                cols = st.columns(chunk_size)

                                for j, (m_name, score) in enumerate(chunk):
                                    with cols[j]:
                                        with st.container(border=True):
                                            clean_name = m_name.replace("_model", "").replace("_", " ").title()
                                            clean_name = clean_name.replace("Rf", "Random Forest"
                                                                            ).replace("Tf", "TensorFlow"
                                                                                      ).replace("Xgb", "XGBoost")
                                            if score is not None:
                                                st.metric(f"{clean_name}", f"{score:.1f}")
                                                st.progress(score / 100)
                                            else:
                                                st.error("Failed")
                    except Exception as e:
                        st.error(f"Connection Failed: {e}")
        else:
            result_area.info("👈 Configure parameters on the left/top and click **Predict** to see the magic.")

with tab_stats:
    st.markdown("### 📊 Model Performance Benchmarks")

    metrics_df = pd.DataFrame(MODEL_STATS).T.reset_index().rename(columns={"index": "Model"})
    metrics_melted = metrics_df.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"], var_name="Metric",
                                     value_name="Value")

    fig_metrics = px.bar(
        metrics_melted, x="Model", y="Value", color="Metric", barmode="group",
        title="Metric Comparison (RMSE/MAE: Lower is better, R2: Higher is better)",
        text_auto='.2f', height=500
    )
    st.plotly_chart(fig_metrics, width="stretch")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("### 🌟 Feature Importance (Random Forest)")
        fig_imp = px.bar(
            RF_IMPORTANCE_DATA, x="Importance", y="Feature", orientation='h',
            title="What makes a hit?", color="Importance", color_continuous_scale="Viridis",
            height=400
        )
        st.plotly_chart(fig_imp, width="stretch")

    with col_s2:
        st.markdown("### ⚙️ Architecture Details")
        for model_name, data in MODEL_STATS.items():
            with st.expander(f"🔹 {model_name} Params"):
                st.write(f"**Best R2:** {data['R2']:.4f}")
                st.json(data["Params"])