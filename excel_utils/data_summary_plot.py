import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from statistics import geometric_mean, harmonic_mean
import os

from statistics import geometric_mean, StatisticsError

@st.cache_resource
def format_number(value):
    if isinstance(value, (int, float)):
        return "{:,.0f}".format(value)
    return value

@st.cache_resource
def safe_geometric_mean(series):
    """Return geometric mean only if data is positive and non-empty"""
    positive_values = series.dropna()
    positive_values = positive_values[positive_values > 0]

    if not positive_values.empty:
        try:
            return geometric_mean(positive_values)
        except StatisticsError:
            return None
    return None

#count_ratio

@st.cache_resource
def summarize_and_plot(df, metadata_df, save_dir='eda_outputs'):
    #metadata_df, count_num, count_cat, count_text, count_datetime, count_percentage,df,count_ratio = detect_scales_of_measurement(df)
    
    # Check if input data is valid
    if df.empty:
        st.warning("Input DataFrame is empty. Returning empty results.")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], [], [], pd.DataFrame(), [], [])
    
    if metadata_df.empty:
        st.warning("Metadata DataFrame is empty. Returning empty results.")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], [], [], pd.DataFrame(), [], [])
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if not metadata_df.empty:
        print(metadata_df)
    rate_columns = metadata_df.loc[metadata_df['Data type Category'] == 'Percentage','Column'].tolist() if not metadata_df.empty else []
    print("rate_columns:", rate_columns)
    rate_columns_Ratio = metadata_df.loc[metadata_df['Data type Category'] == 'Ratio','Column'].tolist() if not metadata_df.empty else []
    print("rate_columns_Ratio:", rate_columns_Ratio)



    summary_list = []
    column_summary_tables = []
    plot_paths = []
    plot_paths2=[]
    plot_paths3=[]
    plot_paths4=[]
    summary_info=[]
    interpretations=[]



    # Process frequency distribution plots for all columns
    if df.columns.empty:
        st.warning("No columns found in the dataset.")
    else:
        print(f"Processing frequency plots for {len(df.columns)} columns...")
        for col in df.columns:
            if df[col].nunique() <= 50:  # Or another threshold
                try:
                    freq_path = os.path.join(save_dir, f"frequency_dist_{col}.png")
                    plt.figure(figsize=(8, 5))
                    value_counts = df[col].value_counts(dropna=False).head(20)
                    sns.barplot(x=value_counts.values, y=value_counts.index, orient='h')
                    plt.title(f'Frequency Distribution - {col}')
                    plt.xlabel('Count')
                    plt.ylabel('Value')
                    plt.tight_layout()
                    plt.savefig(freq_path)
                    plt.close()
                    plot_paths3.append((col, freq_path))
                except Exception as e:
                    print(f"Error processing frequency plot for column {col}: {str(e)}")
                    continue
    
    summary_metrics_1 = ["Count", "Min", "Max" ]
    summary_metrics_2 = ["Mean", "Median", "Mode","Geometric Mean", "Harmonic Mean", "Std Dev", "Variance"]
    summary_metrics_3 = ["25th Quartile", "50th Quartile", "75th Quartile", "Skewness"]

    summary_data_1 = {}
    summary_data_2 = {}
    summary_data_3 = {}
    
    # Get numeric columns directly from metadata_df, filtering out boolean columns
    if not metadata_df.empty:
        potential_numeric_cols = metadata_df.loc[
            metadata_df['Data type Category'].isin(['Numerical', 'Percentage', 'Ratio'])
        ]['Column'].tolist()
        
        # Filter out boolean columns manually
        numerical_cols = []
        for col in potential_numeric_cols:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and not df[col].dtype == 'bool':
                    numerical_cols.append(col)
    else:
        numerical_cols = []
    
    if not numerical_cols:
        st.warning("No numerical columns found in the dataset.")
        # Return empty DataFrames and empty lists
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], [], [], pd.DataFrame(), [], [])
    
    print(f"Processing {len(numerical_cols)} numerical columns...")
    
    for col in numerical_cols:
        #rate_columns = metadata_df[metadata_df['Data type Category'].isin(['Percentage/Ratio'])]['Column'].tolist()#, 'Rate', 'Ratio (Continuous)', 'Percentage'])
        try:
            print(f"Processing column: {col}")
            series = df[col].dropna()
            #print(series)
            print(f"→ '{col}' in rate_columns? ", col in rate_columns)
            print(f"'{col}' in rate_columns_Ratio?", col in rate_columns_Ratio)
            
            print(" rate_cols:", rate_columns)
            print("rate_columns_Ratio:", rate_columns_Ratio)
            #valid_rate = (col in rate_columns) and (not series.empty) and (series > 0).all()
            is_valid_for_rate_mean = False
            if not series.empty and (series > 0).all():
                is_valid_for_rate_mean = True
            print("→ is_valid_for_rate_mean:", is_valid_for_rate_mean)
            is_valid_for_ratio_mean = False
            if not series.empty and (series > 0).all():
                is_valid_for_ratio_mean = True
            print("→ is_valid_for_ratio_mean:", is_valid_for_ratio_mean)
            
            values_1 = [
                round(int(series.count()),0),
                        round(series.min(), 1),
                        round(series.max(), 1)]
                        #round(series.sum(), 3),
            values_2 = [
                        round(series.mean(), 1),
                        round(series.median(), 1),
                        round(series.mode().iloc[0], 1) if not series.mode().empty else "NA",
                        round(geometric_mean(series), 1) if  col in rate_columns and is_valid_for_rate_mean else "NA",
                        round(harmonic_mean(series), 1) if col in rate_columns_Ratio and is_valid_for_ratio_mean else "NA",
                        round(series.std(), 1),
                        round(series.var(), 1)]
            values_3 = [
                        round(series.quantile(0.25), 1),
                        round(series.quantile(0.50), 1),
                        round(series.quantile(0.75), 1),
                        round(series.skew(), 1)
                        ]
            
            skew_value = round(df[col].skew(), 1)
            mean = format_number(int(df[col].mean()))
            median = format_number(int(df[col].median()))
            std = format_number(int(df[col].std()))
            q75 = format_number(int(df[col].quantile(0.75)))

            if skew_value <-1:
                summary_info.append(f"<b>{col}</b> has an average value of <b>{median}</b> with most of the datapoints typically falling around <b>{std}</b> units away from the average. A skewness of <b>{skew_value}</b> indicates that the variable is <b>Negatively (Left) skewed</b>.")
            elif skew_value > 1:
                summary_info.append(f"<b>{col}</b> has an average value of <b>{median}</b> with most of the datapoints typically falling around <b>{std}</b> units away from the average. A skewness of <b>{skew_value}</b> indicates that the variable is <b>Positively (Right) skewed</b>.")
            else:
                summary_info.append(f"<b>{col}</b> has an average value of <b>{mean}</b> with most of the datapoints typically falling around <b>{std}</b> units away from the average. A skewness of <b>{skew_value}</b> indicates that the variable is <b>Approximately Symmetric</b>")

            formatted_values_1 = [format_number(val) for val in values_1]
            formatted_values_2 = [format_number(val) for val in values_2]
            formatted_values_3 = [val if idx == len(values_3) - 1 else format_number(val) for idx, val in enumerate(values_3)]

            try:
                hist_path = os.path.join(save_dir, f"hist-{col}.png")
                plt.figure(figsize=(3,3))  # Increased from (2,2) to (3,3)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'{col}')
                plt.savefig(hist_path, dpi=100, bbox_inches='tight')  # Added dpi and bbox_inches
                plt.close()
                plot_paths.append((col, hist_path))
            except Exception as e:
                print(f"Error creating histogram for column {col}: {str(e)}")
                continue

            try:
                box_path = os.path.join(save_dir, f"box-{col}.png")
                plt.figure(figsize=(1,1))  # Increased from (2,2) to (3,3)
                sns.boxplot(y=df[col])
                plt.title(f'{col}')
                plt.savefig(box_path, dpi=100, bbox_inches='tight')  # Added dpi and bbox_inches
                plt.close()
                plot_paths2.append((col, box_path))
            except Exception as e:
                print(f"Error creating boxplot for column {col}: {str(e)}")
                continue

        
        except Exception as e:
            #formatted_values = ["Error"] * len(summary_metrics)
            formatted_values_1 = ["Error"] * len(summary_metrics_1)
            formatted_values_2 = ["Error"] * len(summary_metrics_2)
            formatted_values_3 = ["Error"] * len(summary_metrics_3)

        #summary_data[col] = formatted_values
        summary_data_1[col] = formatted_values_1
        summary_data_2[col] = formatted_values_2
        summary_data_3[col] = formatted_values_3
    
    # Initialize skew_info list
    skew_info = []
    
    # Get numeric columns for skewness analysis
    numeric_cols = metadata_df.loc[metadata_df['Data type Category'].isin(['Numerical', 'Percentage','Ratio'])]['Column'].tolist() if not metadata_df.empty else []
    # Filter to only include columns that exist in both DataFrames
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        try:
            all_numeric_series = df[numeric_cols].skew()
            Pos_skew_col = all_numeric_series[(all_numeric_series > 1)].index.tolist()
            Neg_skew_col = all_numeric_series[(all_numeric_series < -1)].index.tolist()
            skew_col = Pos_skew_col + Neg_skew_col
            non_skew_col = [col for col in numeric_cols if col not in skew_col]
            pos_skew_count = len(Pos_skew_col)
            neg_skew_count = len(Neg_skew_col)
            non_skew_count = len(non_skew_col)
            
            skew_info.append(f"Total number of <b>Positive skewed</b> columns: <b>{pos_skew_count}</b>")
            skew_info.append(f"Positive Skewed column names: {Pos_skew_col}")
            skew_info.append(f"Total number of <b>Negative skewed</b> columns: <b>{neg_skew_count}</b>")
            skew_info.append(f"Negative Skewed column names: {Neg_skew_col}")
            skew_info.append(f"Total number of <b>Approximately Symmetric</b> columns: <b>{non_skew_count}</b>")
            skew_info.append(f"Approximately Symmetric column names: {non_skew_col}")
        except Exception as e:
            st.warning(f"Error calculating skewness: {e}")
            skew_info.append("Error calculating skewness for numeric columns.")
    else:
        skew_info.append("No numeric columns to calculate skewness.")
    
    # Convert to DataFrame
    # summary_df = pd.DataFrame(summary_data, index=summary_metrics)
    # summary_df = summary_df.T
    if summary_data_1:
        summary_df_1 = pd.DataFrame(summary_data_1, index=summary_metrics_1)
        summary_df_1 = summary_df_1.T
    else:
        summary_df_1 = pd.DataFrame()
    
    if summary_data_2:
        summary_df_2 = pd.DataFrame(summary_data_2, index=summary_metrics_2)
        summary_df_2 = summary_df_2.T
    else:
        summary_df_2 = pd.DataFrame()
    
    if summary_data_3:
        summary_df_3 = pd.DataFrame(summary_data_3, index=summary_metrics_3)
        summary_df_3 = summary_df_3.T
    else:
        summary_df_3 = pd.DataFrame()
    if not summary_df_2.empty and "Geometric Mean" in summary_df_2.columns and "Harmonic Mean" in summary_df_2.columns:
        print(summary_df_2[["Geometric Mean", "Harmonic Mean"]])
    #print(summary_df[["Geometric Mean", "Harmonic Mean"]])
    #summary_df.reset_index(inplace=True)
    if not summary_df_1.empty:
        summary_df_1.reset_index(inplace=True)
        summary_df_1.rename(columns={'index': 'Variables'}, inplace=True)
    
    if not summary_df_2.empty:
        summary_df_2.reset_index(inplace=True)
        summary_df_2.rename(columns={'index': 'Variables'}, inplace=True)
    
    if not summary_df_3.empty:
        summary_df_3.reset_index(inplace=True)
        summary_df_3.rename(columns={'index': 'Variables'}, inplace=True)

    #interpretations = []

    dist_data = []
    categorical_cols = metadata_df[metadata_df["Data type Category"] == "Categorical"]["Column"].tolist() if not metadata_df.empty else []

# Now iterate through those categorical columns in df
        

   
    
    # Only process categorical columns if they exist
    if categorical_cols:
        print(f"Processing {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else np.nan
            last_val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else np.nan
            unique_vals = df[col].nunique()
            unique_count = unique_vals
            total_count = df[col].count()

            fdist_path = os.path.join(save_dir, f"fdist-{col}.png")

            # Get value counts (including NaN)
            value_counts = df[col].value_counts(dropna=False)

            # Keep top 5, sum the rest as "Others"
            top_5 = value_counts.head(5)
            '''if len(value_counts) > 5:
                others_count = value_counts.iloc[5:].sum()
                top_5["Others"] = others_count'''
            if len(value_counts) > 5:
                others_count = value_counts.iloc[5:].sum()
                # Convert to Series and concat to preserve order
                top_5 = pd.concat([top_5, pd.Series({"Others": others_count})])

            try:
                # Plot
                #fig_height = max(4, len(top_5) * 0.4)  # Adjust height based on categories
                plt.figure(figsize=(3, 2))
                top_5.plot(kind='barh', color="skyblue", edgecolor="black")

                plt.title(f'Frequency Distribution: {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Save and close
                plt.savefig(fdist_path)
                plt.close()

                plot_paths4.append((col, fdist_path))
            except Exception as e:
                print(f"Error creating frequency distribution plot for column {col}: {str(e)}")
                continue

            try:
                dist_data.append({
                    "Column": col,
                    "Unique": unique_vals,
                    "First Value": first_val,
                    "Last Value": last_val,
                })
            except Exception as e:
                print(f"Error adding distribution data for column {col}: {str(e)}")
                continue
    
    if dist_data:
        dist_df = pd.DataFrame(dist_data)
    else:
        dist_df = pd.DataFrame()

    # if 'Geometric Mean' in summary_df.columns and (summary_df['Geometric Mean'] == 'NA').all():
    #     summary_info.append("The Geometric and Harmonic means are NA because the dataset does not contain change% or fraction type variables or may have negative values, for which GM and HM would be appropriate.")
    if not summary_df_2.empty and 'Geometric Mean' in summary_df_2.columns and (summary_df_2['Geometric Mean'] == 'NA').all():
        summary_info.append("The Geometric and Harmonic means are NA because the dataset does not contain change% or fraction type variables or may have negative values, for which GM and HM would be appropriate.")

    #return summary_df, interpretations, plot_paths, plot_paths2, plot_paths3, dist_df, plot_paths4,summary_info
    return summary_df_1,summary_df_2, summary_df_3, interpretations, plot_paths, plot_paths2, plot_paths3, dist_df, plot_paths4,summary_info,skew_info

    #interpretations


@st.cache_resource
def plot_combined_categorical_distribution(df, categorical_cols, save_dir, filename="combined_fdist.png"):
    combined_counts = {}

    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        top_5 = vc.head(1)
        others_sum = vc.iloc[5:].sum()

        # Format category labels with column name prefix
        col_labels = [f"{col}: {val}" for val in top_5.index.astype(str)]
        col_counts = top_5.values.tolist()

        if others_sum > 0:
            col_labels.append(f"{col}: Others")
            col_counts.append(others_sum)

        # Append to combined_counts
        for label, count in zip(col_labels, col_counts):
            combined_counts[label] = count

    # Plot all at once
    plt.figure(figsize=(10, max(6, len(combined_counts) * 0.3)))
    labels = list(combined_counts.keys())
    counts = list(combined_counts.values())

    plt.barh(labels, counts, color="skyblue", edgecolor="black")
    plt.xticks(rotation=90, ha='right')
    plt.ylabel("Count")
    plt.title("Categorical Frequency Distribution (Top 5 per column + Others)")
    plt.tight_layout()

    # Save plot
    combined_path = os.path.join(save_dir, filename)
    plt.savefig(combined_path)
    plt.close()

    return combined_path