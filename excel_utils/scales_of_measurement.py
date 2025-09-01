# This file contains the function that detects the scale of measurement and the data type category for the columns in the data set.

import pandas as pd
import re
import streamlit as st

def is_year_only_column(series):
    #Returns True if the column is likely year-only (int or 4-digit strings).
    try:
        series_clean = series.dropna().astype(str)
        return series_clean.str.fullmatch(r"(19|20)\d{2}").all()
    except:
        return False

def detect_scales_of_measurement(df):
    count_num = 0
    count_cat = 0
    count_datetime = 0
    count_text = 0
    count_percentage = 0
    count_ratio = 0
    count_geo = 0
    scale_info = []
    uniqueness_threshold = 0.8
    length_threshold = 40

    for col in df.columns:
        var_type = "Unknown"
        scale = "Unknown"
        dtype = str(df[col].dtype)  # This is correct - df[col] returns a Series
        nunique = df[col].nunique(dropna=True)
        col_non_null = df[col].dropna()
        col_name_lower = col.lower()

        # --- NEW LOCATION FOR RATIO CHECK ---
        is_ratio_by_name = 'ratio' in col_name_lower
        is_geo_by_name = any(geo_name in col_name_lower for geo_name in ['latitude', 'longitude'])
        is_percent_by_name = 'percent' in col_name_lower
        is_percent_by_content = False
        if not is_percent_by_name:
            is_percent_series = df[col].dropna().astype(str).str.contains('%')
            if not is_percent_series.empty and is_percent_series.sum() / len(is_percent_series) > 0.5:
                is_percent_by_content = True
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for 'ratio' within the numeric loop
            if is_ratio_by_name:
                var_type = "Ratio"
                count_ratio += 1
                scale = "Continuous"
            elif nunique == 2:
                var_type = "Categorical" #Numerical
                count_cat += 1
                scale = "Binary (Boolean)"
            elif is_percent_by_name or is_percent_by_content:
                var_type = "Percentage"
                count_percentage += 1
                scale = "Continuous"
                print(f"Detected column '{col}' as Percentage. Converting...")
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("%", '', regex=False).str.strip(), 
                    errors='coerce'
                )
                dtype = str(df[col].dtype)  # Update dtype after conversion
            elif 'latitude' in col_name_lower or 'longitude' in col_name_lower:
                var_type = "Geography"
                count_geo +=1
                scale = "Geography" # As per user request

            else:
                var_type = "Numerical"
                count_num += 1
                min_val = col_non_null.min()
                if min_val >= 0:
                    scale = "Continuous"
                else:
                    scale = "Interval (Continuous)"
        
        elif pd.api.types.is_bool_dtype(df[col]):
            count_cat += 1
            var_type = "Categorical" 
            scale = "Binary (Boolean)"

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            var_type = "Datetime"
            count_datetime += 1
            unique_months = col_non_null.dt.month.nunique()
            unique_days = col_non_null.dt.day.nunique()
            unique_hours = col_non_null.dt.hour.nunique()
            
            if unique_months == 1 and unique_days == 1 and unique_hours == 1:
                scale = "Datetime (Interval)"
            elif unique_days == 1 and unique_hours == 1:
                scale = "Datetime (Interval)"
            elif unique_hours <= 1:
                scale = "Datetime (Interval)"
            else:
                scale = "Datetime (Continuous)"

        elif is_year_only_column(df[col]):
            var_type = "Datetime"
            count_datetime += 1
            scale = "Datetime (Continuous)"
            df[col] = pd.to_datetime(df[col], format="%Y", errors='coerce')
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            unique_ratio = nunique / len(df)
            avg_length = col_non_null.astype(str).map(len).mean()
            
            # Original check for 'percent' and content, as a separate category
            is_percent_by_name = 'percent' in col_name_lower
            is_percent_by_content = False
            if not is_percent_by_name:
                is_percent_series = df[col].dropna().astype(str).str.contains('%')
                if not is_percent_series.empty and is_percent_series.sum() / len(is_percent_series) > 0.5:
                    is_percent_by_content = True

            if is_percent_by_name or is_percent_by_content:
                var_type = "Percentage"
                count_percentage += 1
                scale = "Continuous"
                print(f"Detected column '{col}' as Percentage. Converting...")
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("%", '', regex=False).str.strip(), 
                    errors='coerce'
                )
                dtype = str(df[col].dtype)  # Update dtype after conversion
            
            # Else, check for 'text'
            elif (unique_ratio > uniqueness_threshold or avg_length > length_threshold):
                var_type = "Text"
                count_text += 1
                scale = "Text (Free Text)"
            
            # Else, it must be a categorical column
            else:
                var_type = "Categorical"
                count_cat += 1
                if nunique == 2:
                    scale = "Binary (Categorical)"
                elif nunique <= 10:
                    if pd.api.types.is_categorical_dtype(df[col]) and df[col].cat.ordered:
                        scale = "Ordinal (Categorical)"
                    else:
                        scale = "Nominal (Categorical - Few Classes)"
                else:
                    scale = "Nominal (Categorical - MultiClass)"

        scale_info.append({
            "Column": col,
            "Data type Category": var_type,
            "Data type": dtype,
            "Scale of Measurement": scale,
            "Unique Values": nunique
        })

    scale_df = pd.DataFrame(scale_info)
    
    return scale_df, count_num, count_cat, count_text, count_datetime, count_percentage, count_ratio, count_geo, df