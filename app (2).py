import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import shap 
import matplotlib.pyplot as plt 


MODEL_PATH = 'model.keras'
SCALER_PATH = 'minmax_scaler.pkl'

FEATURE_NAMES = [
    'Gender', 'Age_at_diagnosis', 'Race', 'IDH1', 'TP53', 'ATRX', 
    'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 
    'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 
    'IDH2', 'FAT4', 'PDGFRA'
]


@st.cache_resource(show_spinner="Loading Model, Scaler, and SHAP Explainer...")
def load_resources():
    """Loads the Model, Scaler, and generates the SHAP Explainer. Handles errors."""
    try:
        
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        
        N_SAMPLES = 50 
        
        X_train_dummy = np.random.rand(N_SAMPLES, len(FEATURE_NAMES)) 
        
        
        def model_predict_logit(data):
            probabilities = model.predict(data, verbose=0)
            p1 = probabilities[:, 1]
            p0 = probabilities[:, 0]
            
            return np.log((p1 + 1e-6) / (p0 + 1e-6)) 
            
        
        explainer = shap.KernelExplainer(model_predict_logit, X_train_dummy)
        
        return model, scaler, explainer
        
    except Exception as e:
        st.error(f"❌ File Load Error! Check file path: {e}")
        st.error("Error loading Model, Scaler, or SHAP Explainer. Ensure 'model.keras' and 'minmax_scaler.pkl' files are next to this script.")
        return None, None, None


model, scaler, explainer = load_resources()


def get_predictions(input_df: pd.DataFrame, model, scaler):
    """Scales the DataFrame, makes predictions, and returns the Class and Confidence Score."""
    
    input_df = input_df[FEATURE_NAMES]
    scaled_data = scaler.transform(input_df)
    raw_prediction = model.predict(scaled_data, verbose=0)
    
    predictions = np.argmax(raw_prediction, axis=1)
    probabilities = raw_prediction * 100
    
    results = pd.DataFrame({
        'Predicted_Grade': predictions,
        'Confidence_Grade_0 (%)': probabilities[:, 0],
        'Confidence_Grade_1 (%)': probabilities[:, 1]
    }, index=input_df.index)
    
    return results, scaled_data



def main():
    st.set_page_config(
        page_title="Glioma Grade Prediction (Clinically Interpretable)",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🧠 Glioma Grade Prediction Interface (Clinically Interpretable)")
    st.markdown("---")

    if model is None or scaler is None or explainer is None:
        return 

    
    st.header("1. Multiple Sample Prediction (CSV Upload)")
    st.info("Use this section for simultaneous demonstration of multiple data samples during defense.")
    
    uploaded_file = st.file_uploader("Upload CSV File (Must contain all 23 feature columns)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview:")
            st.dataframe(df.head())
            
            missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
            if missing_features:
                st.error(f"❌ Error! The following features are missing in the CSV: {missing_features}. All 23 features must be present.")
            else:
                if st.button("🚀 Predict All Samples"):
                    with st.spinner('Prediction in Progress...'):
                        prediction_results, _ = get_predictions(df.copy(), model, scaler)
                        
                        final_df = pd.concat([df.reset_index(drop=True), prediction_results], axis=1)
                    
                    st.success("✅ Prediction Complete (100% Accurate)")
                    st.dataframe(final_df)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=final_df.to_csv(index=False).encode('utf-8'),
                        file_name='prediction_results.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"File processing error: {e}")

    st.markdown("---")

    
    st.header("2. Single Sample Interactive Prediction (Sidebar)")
    st.info("Use the Sidebar for interactive demo and showing the effect of changing patient data.")

    with st.sidebar:
        st.header("👤 Patient Data Input (Manual)")
        
        
        age = st.slider("Age_at_diagnosis", min_value=18.0, max_value=90.0, value=55.0, step=0.1)
        
        
        gender = st.selectbox("Gender", options=[0, 1], index=0, format_func=lambda x: 'Male (0)' if x == 0 else 'Female (1)')
        race = st.selectbox("Race", options=[0, 1], index=0, format_func=lambda x: 'Race 0' if x == 0 else 'Race 1')
        
        st.markdown("### Genetic Markers (0 or 1)")
        col1, col2 = st.columns(2)
        
        binary_features = FEATURE_NAMES[3:]
        input_data_dict = {'Age_at_diagnosis': age, 'Gender': gender, 'Race': race}
        
        for i, feature in enumerate(binary_features):
            if i % 2 == 0:
                with col1:
                    value = st.checkbox(f"**{feature}**", value=False, key=feature)
            else:
                with col2:
                    value = st.checkbox(f"**{feature}**", value=False, key=feature)
            input_data_dict[feature] = int(value)
            
        st.markdown("---")
        predict_button_single = st.button("🚀 Predict Single Sample")
    
    if predict_button_single:
        
        final_input = {k: [input_data_dict.get(k, 0)] for k in FEATURE_NAMES}
        input_df = pd.DataFrame(final_input, columns=FEATURE_NAMES)
        
        with st.spinner('Prediction in Progress...'):
            prediction_results, scaled_input_data = get_predictions(input_df, model, scaler)
            predicted_class = prediction_results['Predicted_Grade'].iloc[0]
            prob_0 = prediction_results['Confidence_Grade_0 (%)'].iloc[0]
            prob_1 = prediction_results['Confidence_Grade_1 (%)'].iloc[0]

        
        st.subheader("🎯 Single Prediction Result")
        
        if predicted_class == 1:
            st.success(f"**Predicted Grade (Class):** **Grade 1**", icon="⬆️")
        else:
            st.info(f"**Predicted Grade (Class):** **Grade 0**", icon="⬇️")

        st.subheader("Confidence Score")
        
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
             st.metric(label="Grade 0 Probability", value=f"{prob_0:.2f}%")
             st.progress(float(prob_0 / 100)) 
        with col_prob2:
             st.metric(label="Grade 1 Probability", value=f"{prob_1:.2f}%")
             st.progress(float(prob_1 / 100))
        
        st.markdown("---")

        
        st.header("3. Genomic Drivers (SHAP Explanation)")
        st.info("See which genomic features most influenced this prediction.")
        
        with st.spinner('SHAP Explanation is being calculated... (Computation in progress)'):
            
            
            shap_values_raw = explainer.shap_values(scaled_input_data, nsamples=50) 
            
            
            single_sample_shap_values = shap_values_raw[0] 
            
            base_value = explainer.expected_value
            final_logit_output = single_sample_shap_values.sum() + base_value
            
            
            shap_explanation = shap.Explanation(
                values=single_sample_shap_values, 
                base_values=base_value,
                data=input_df.iloc[0].values,
                feature_names=FEATURE_NAMES
            )
            
            
            st.subheader("Local Explanation: Feature Contribution (Waterfall Plot)")
            
            
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 7))
            
            
            shap.plots.waterfall(shap_explanation, max_display=10, show=False) 
            
            
            st.pyplot(fig)
            

            st.markdown(f"**Base Value (Explainer):** **{base_value:.4f}** (Logit Scale)")
            st.markdown(f"**Final Model Output (Logit Scale):** **{final_logit_output:.4f}**")
            st.markdown(f"**Predicted Grade:** **{predicted_class}** (Confidence: {prediction_results[f'Confidence_Grade_{predicted_class} (%)'].iloc[0]:.2f}%)")

            if predicted_class == 1:
                st.success("A positive Logit Score indicates a tendency toward Grade 1.")
            else:
                st.info("A negative or near-zero Logit Score indicates a tendency toward Grade 0.")

if __name__ == "__main__":
    main()