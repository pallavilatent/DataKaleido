import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from excel_utils.scales_of_measurement import detect_scales_of_measurement

def detect_outliers_iqr(df, data_type_df, save_dir='output'):
    """
    Streamlit-based outlier detection and capping using IQR method
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    st.header("🔍 Outlier Detection & Capping")
    st.markdown("---")
    
    # Get numeric columns safely, filtering out boolean columns
    numeric_cols = data_type_df.loc[data_type_df['Data type Category'].isin(['Numerical', 'Percentage','Ratio'])]['Column'].tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for outlier detection.")
        return pd.DataFrame(), ["No numeric columns available"], df, data_type_df
    
    # Initialize session state for storing outlier information
    try:
        if 'outlier_summary' not in st.session_state:
            st.session_state.outlier_summary = {}
        if 'outlier_summary_info' not in st.session_state:
            st.session_state.outlier_summary_info = []
        if 'outlier_interpretations' not in st.session_state:
            st.session_state.outlier_interpretations = []
    except Exception as e:
        st.error(f"Error initializing session state: {e}")
        # Fallback to local variables if session state fails
        outlier_summary = {}
        outlier_summary_info = []
        outlier_interpretations = []
    
        # Step 1: Detect Outliers
    st.subheader("📊 Step 1: Outlier Detection")
    
    if st.button("🔍 Detect Outliers", type="primary"):
        try:
            st.session_state.outlier_summary = {}
            st.session_state.outlier_summary_info = []
        except Exception as e:
            st.error(f"Error updating session state: {e}")
            # Continue with local variables if session state fails
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        for idx, col in enumerate(numeric_cols):
            try:
                status_text.text(f"Analyzing column: {col}")
                progress_bar.progress((idx + 1) / len(numeric_cols))
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df.index[outliers_condition].tolist()
                
                if outlier_indices:
                    outlier_values = df.loc[outlier_indices, col].tolist()
                    try:
                        st.session_state.outlier_summary_info.append([col, len(outlier_indices)])
                        st.session_state.outlier_summary[col] = {
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'outlier_count': len(outlier_indices),
                            'outlier_values': outlier_values,
                            'Q1': Q1,
                            'Q3': Q3,
                            'IQR': IQR
                        }
                    except Exception as e:
                        st.error(f"Error updating session state for column {col}: {e}")
                        # Continue processing other columns
                    
            except Exception as e:
                st.error(f"Error processing column '{col}': {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        try:
            if not st.session_state.outlier_summary:
                st.success("✅ No outliers detected in any numeric columns!")
                try:
                    st.session_state.outlier_interpretations.append("No presence of outliers")
                except Exception as e:
                    st.error(f"Error updating session state: {e}")
                return pd.DataFrame(), ["No presence of outliers"], df, data_type_df
            
            st.success(f"🎯 Outliers detected in {len(st.session_state.outlier_summary)} columns!")
        except Exception as e:
            st.error(f"Error checking outlier summary: {e}")
            return pd.DataFrame(), ["Error in outlier detection"], df, data_type_df
    
    # Display outlier summary if available
    try:
        outlier_summary = st.session_state.outlier_summary
    except Exception as e:
        st.error(f"Error accessing session state: {e}")
        outlier_summary = {}
    
    if outlier_summary:
        st.subheader("📋 Outlier Summary")
        
        # Create summary dataframe
        summary_data = []
        for col, info in outlier_summary.items():
            summary_data.append({
                'Column': col,
                'Outlier Count': info['outlier_count'],
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Step 2: Outlier Capping Options
        st.subheader("⚙️ Step 2: Outlier Capping Configuration")
        
        # Use a more robust approach for columns
        cols = st.columns(2)
        
        with cols[0]:
            capping_strategy = st.selectbox(
                "Choose capping strategy:",
                ["Cap all detected columns", "Select specific columns", "Skip capping"]
            )
        
        with cols[1]:
            if capping_strategy == "Select specific columns":
                columns_to_cap = st.multiselect(
                    "Select columns to cap:",
                    options=list(outlier_summary.keys()),
                    default=list(outlier_summary.keys())
                )
            else:
                columns_to_cap = []
        
        # Step 3: Execute Capping
        if st.button("🚀 Apply Outlier Capping", type="primary"):
            if capping_strategy == "Skip capping":
                st.info("Outlier capping skipped as requested.")
                st.session_state.outlier_interpretations.append("Outlier capping was skipped as per user request.")
            else:
                if capping_strategy == "Cap all detected columns":
                    columns_to_cap = list(outlier_summary.keys())
                
                if not columns_to_cap:
                    st.warning("No columns selected for capping.")
                    return pd.DataFrame(), st.session_state.outlier_interpretations, df, data_type_df
                
                # Apply capping
                capped_cols = []
                df_copy = df.copy()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, col in enumerate(columns_to_cap):
                    if col in outlier_summary:
                        status_text.text(f"Capping outliers in column: {col}")
                        progress_bar.progress((idx + 1) / len(columns_to_cap))
                        
                        info = outlier_summary[col]
                        lower_bound = info['lower_bound']
                        upper_bound = info['upper_bound']
                        
                        df_copy[col] = np.where(
                            df_copy[col] < lower_bound, 
                            lower_bound,
                            np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
                        )
                        capped_cols.append(col)
                
                progress_bar.empty()
                status_text.empty()
                
                if capped_cols:
                    st.success(f"✅ Outliers capped successfully in {len(capped_cols)} columns!")
                    st.session_state.outlier_interpretations.append(
                        f"Outliers were detected and capped using lower and upper bounds based on IQR method for the following columns: {', '.join(capped_cols)}"
                    )
                    
                    # Save the modified dataframe
                    df_copy.to_csv(f"{save_dir}/outlier_capping.csv", index=False)
                    st.info(f"📁 Modified DataFrame saved to {save_dir}/outlier_capping.csv")
                    
                    # Update the original dataframe
                    df = df_copy
                    
                    # Re-run scales of measurement
                    try:
                        result = detect_scales_of_measurement(df)
                        if len(result) >= 7:
                            data_type_df, count_num, count_cat, count_text, count_datetime, count_percentage, count_ratio = result[:7]
                        else:
                            st.error(f"Unexpected number of return values from detect_scales_of_measurement: {len(result)}")
                            data_type_df = result[0] if result else None
                    except Exception as e:
                        st.error(f"Error in scales of measurement: {e}")
                        data_type_df = None
                    
    # Return results
    try:
        outlier_summary_info = st.session_state.outlier_summary_info
        outlier_interpretations = st.session_state.outlier_interpretations
    except Exception as e:
        st.error(f"Error accessing session state for return: {e}")
        outlier_summary_info = []
        outlier_interpretations = ["Error accessing session state"]
    
    outlier_summary_df = pd.DataFrame(outlier_summary_info, columns=['Column', 'Number of outliers']) if outlier_summary_info else pd.DataFrame()
    
    return outlier_summary_df, outlier_interpretations, df, data_type_df

