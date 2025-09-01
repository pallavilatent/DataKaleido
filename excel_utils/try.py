import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------
# Cached Functions (no widgets)
# ---------------------------
@st.cache_data
def auto_missing_value_imputation(df, method):
    if method == "Mean":
        return df.fillna(df.mean(numeric_only=True))
    elif method == "Median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "Mode":
        return df.fillna(df.mode().iloc[0])
    elif method == "Zero":
        return df.fillna(0)
    elif method == "Drop":
        return df.dropna()
    return df.copy()

@st.cache_data
def apply_scaling(df, method):
    numeric_cols = df.select_dtypes(include="number").columns
    df_scaled = df.copy()
    if method == "StandardScaler":
        df_scaled[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    elif method == "MinMaxScaler":
        df_scaled[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    return df_scaled

@st.cache_data
def apply_encoding(df, method):
    df_encoded = df.copy()
    if method == "OneHot":
        df_encoded = pd.get_dummies(df_encoded)
    elif method == "LabelEncoding":
        from sklearn.preprocessing import LabelEncoder
        for col in df_encoded.select_dtypes(include="object").columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded


# ---------------------------
# Main App
# ---------------------------
def run_eda_app():
    st.title("🔍 EDA Preprocessing App")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if not uploaded_file:
        st.info("Please upload a dataset to begin.")
        return

    df = pd.read_csv(uploaded_file)

    # Session state for storing progress
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = df.copy()

    # Tabs for workflow
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Imputation", "⚖️ Scaling", "🔤 Encoding", "📊 Final Results"])

    # ---------------------------
    # Tab 1: Imputation
    # ---------------------------
    with tab1:
        st.subheader("Handle Missing Values")
        method = st.selectbox("Imputation method:", ["Mean", "Median", "Mode", "Zero", "Drop", "None"])
        if st.button("Apply Imputation"):
            st.session_state.processed_df = auto_missing_value_imputation(st.session_state.processed_df, method)
            st.success(f"✅ Imputation applied using {method}")
            st.dataframe(st.session_state.processed_df.head())

    # ---------------------------
    # Tab 2: Scaling
    # ---------------------------
    with tab2:
        st.subheader("Scale Numeric Features")
        scaling = st.selectbox("Scaling method:", ["StandardScaler", "MinMaxScaler", "None"])
        if st.button("Apply Scaling"):
            if scaling != "None":
                st.session_state.processed_df = apply_scaling(st.session_state.processed_df, scaling)
                st.success(f"✅ Scaling applied using {scaling}")
            else:
                st.info("No scaling applied.")
            st.dataframe(st.session_state.processed_df.head())

    # ---------------------------
    # Tab 3: Encoding
    # ---------------------------
    with tab3:
        st.subheader("Encode Categorical Features")
        encoding = st.selectbox("Encoding method:", ["OneHot", "LabelEncoding", "None"])
        if st.button("Apply Encoding"):
            if encoding != "None":
                st.session_state.processed_df = apply_encoding(st.session_state.processed_df, encoding)
                st.success(f"✅ Encoding applied using {encoding}")
            else:
                st.info("No encoding applied.")
            st.dataframe(st.session_state.processed_df.head())

    # ---------------------------
    # Tab 4: Final Results
    # ---------------------------
    with tab4:
        st.subheader("Final Processed Dataset")
        st.write(f"Shape: {st.session_state.processed_df.shape}")
        st.dataframe(st.session_state.processed_df.head(20))
        st.download_button(
            "⬇️ Download Processed CSV",
            st.session_state.processed_df.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv"
        )


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    run_eda_app()
