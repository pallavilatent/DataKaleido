import pandas as pd
import os
import streamlit as st
from excel_utils.scales_of_measurement import detect_scales_of_measurement

@st.cache_resource
def build_combined_frequency_dataframe(df, scale_df, save_dir='eda_outputs'):
    """
    Builds a combined frequency DataFrame and a list of summary interpretations
    for categorical columns based on the scale of measurement.
    """
    combined = []
    freq_table_summary = []

    # Get a list of only the categorical columns
    cat_cols = scale_df.loc[scale_df['Data type Category'] == 'Categorical', 'Column'].tolist()

    # --- NEW: Check if any categorical columns are present ---
    if not cat_cols:
        print("No Categorical present in the dataset.")
        message = "No Categorical variables present in the dataset."
        print(message)
        freq_table_summary.append(message)
        empty_df = pd.DataFrame(columns=['Column', 'Value', 'Frequency Count',
                                         'Cumulative Frequency Count', 'Frequency %',
                                         'Cumulative Frequency %'])
        return empty_df , freq_table_summary

        # Check for 'Column' in scale_df and set as index
    if 'Column' in scale_df.columns:
        scale_df = scale_df.set_index("Column")

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # --- Loop over only the categorical columns ---
    for col in cat_cols:
        total_rows = len(df)
        
        # --- Start of Frequency Table Generation ---
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = ['Value', 'Frequency Count']
        
        counts['Cumulative Frequency Count'] = counts['Frequency Count'].cumsum()
        counts['Frequency %'] = (counts['Frequency Count'] / total_rows * 100).round(0)
        counts['Cumulative Frequency %'] = counts['Frequency %'].cumsum().round(0)

        if not counts.empty:
            counts.loc[counts.index[-1], 'Cumulative Frequency %'] = 100

        counts['Frequency %'] = counts['Frequency %'].astype(str) + '%'
        counts['Cumulative Frequency %'] = counts['Cumulative Frequency %'].astype(str) + '%'

        counts.insert(0, 'Column Name', col)
        
        try:
            filename = f"{col}_frequency_table.csv"
            # Use os.path.join to correctly create the file path
            counts.to_csv(os.path.join(save_dir, filename), index=False)
            print(f"Saved frequency table for '{col}' to '{os.path.join(save_dir, filename)}'")
        except Exception as e:
            print(f"Error saving CSV for '{col}': {e}")
            
        combined.append(counts)
        # --- End of Frequency Table Generation ---
        
        # --- Start of Dynamic Summary Generation ---
        value_counts = df[col].value_counts(dropna=False, normalize=True)
        max_categories = 5
        top_n_count = min(len(value_counts), max_categories)
        
        top_n_series = value_counts.head(top_n_count)
        top_n_values = top_n_series.index.tolist()
        
        top_n_percentages = (top_n_series * 100).round(1).tolist()
        
        cumulative_freq_str = '0.0%'
        if top_n_values:
            last_top_value = top_n_values[-1]
            
            if pd.isna(last_top_value):
                cumulative_row = counts[counts['Value'].isnull()]
            else:
                cumulative_row = counts[counts['Value'] == last_top_value]

            if not cumulative_row.empty:
                cumulative_freq_str = cumulative_row['Cumulative Frequency %'].iloc[0]

        top_n_formatted_values = [str(v) if pd.notnull(v) else "NaN" for v in top_n_values]
        top_n_freq_str = [f'{p:.1f}%' for p in top_n_percentages]

        summary_text = (
            f"The top {top_n_count} categories in the <b>{col}</b> are <b>{top_n_formatted_values}</b>, which together account for <b>{cumulative_freq_str}</b> of the data, with individual frequencies of  <b>{top_n_freq_str}</b>, respectively.")
        
        freq_table_summary.append(summary_text)
        # --- End of Dynamic Summary Generation ---
    
    if not combined:
        message = "No frequency tables were generated due to missing or invalid data."
        print(message)
        freq_table_summary.append(message)
        return pd.DataFrame(), freq_table_summary
    
    final_df = pd.concat(combined, ignore_index=True)
    print(final_df)
    return final_df, freq_table_summary