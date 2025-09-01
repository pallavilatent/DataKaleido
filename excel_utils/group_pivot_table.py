
import numpy as np
import pandas as pd
import os
import streamlit as st
from excel_utils.scales_of_measurement import detect_scales_of_measurement
def groupbyand_pivot(df, data_type_df, save_dir="eda_outputs"):
    """
    Combined function for groupby aggregation and pivot table creation.

    Designed for your EDA pipeline:
    - Interactive.
    - Saves outputs automatically as CSV in `save_dir`.
    """
    all_grouped_dfs = [] 
    grouped_df_i = []
    index_str = []
    col_str = [] 
    val_str = []
    num_col = [] 
    groupby_cols_str = [] 
    agg_funcs_str = []

    proceed = st.selectbox(
        "\nDo you want to proceed with groupby and/or pivot table operations?",
        ["no", "yes"],
        key="group_pivot_proceed"
    )
    if proceed != "yes":
        st.write("Skipping groupby and pivot table operations as per user input.")
        grouped_df_i.append("Skipping groupby and pivot table operations as per user input.")
        return all_grouped_dfs, grouped_df_i, index_str, col_str, val_str, num_col, groupby_cols_str, agg_funcs_str

    #cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = data_type_df.loc[data_type_df['Data type Category'] == 'Categorical', 'Column'].tolist()
    #num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Use safe numeric columns function to filter out boolean columns
    num_cols = data_type_df.loc[data_type_df['Data type Category'].isin(['Numerical', 'Percentage', 'Ratio'])]['Column'].tolist()
    if len(cat_cols) == 0 or len(num_cols) == 0:
        st.write(" Groupby and pivot require both categorical and numerical columns. Skipping operation.")
        grouped_df_i.append("Skipping groupby and pivot table operations due to lack of required columns.")
        return all_grouped_dfs, grouped_df_i, index_str, col_str, val_str, num_col, groupby_cols_str, agg_funcs_str

    ### =============== GROUPBY SECTION ===============
    groupby_proceed = st.selectbox(
        "\nDo you want to perform groupby operations?",
        ["no", "yes"],
        key="groupby_proceed"
    )
    
    if groupby_proceed == "yes":
        st.subheader("📊 GroupBy Operations")
        
        # Single groupby operation with interactive widgets
        if len(cat_cols) > 0 and len(num_cols) > 0:
            # Select grouping column(s)
            group_col = st.selectbox(
                "Select grouping column (Categorical):",
                cat_cols,
                key="group_select_1"
            )
            
            # Select aggregation column
            agg_col = st.selectbox(
                "Select aggregation column (Numerical):",
                num_cols,
                key="agg_select_1"
            )
            
            # Select aggregation function
            agg_func = st.selectbox(
                "Select aggregation function:",
                ["mean", "sum", "count", "min", "max", "median", "std", "var", "nunique"],
                key="agg_func_select_1"
            )
            
            if group_col and agg_col and agg_func:
                try:
                    # Create groupby table
                    grouped_df = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                    grouped_df.columns = [group_col, f'{agg_func.title()} of {agg_col}']
                    
                    st.write(f"**Grouped by {group_col} - {agg_func.title()} of {agg_col}:**")
                    
                    # Save CSV
                    groupby_cols_str = group_col
                    agg_funcs_str = agg_func
                    filename = f"groupby_{agg_col}_by_{groupby_cols_str}_{agg_funcs_str}.csv"
                    filepath = os.path.join(save_dir, filename)
                    grouped_df.to_csv(filepath, index=False)
                    
                    # Create download button for the groupby result
                    csv_data = grouped_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download GroupBy Table ({filename})",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_groupby_{group_col}_{agg_col}_{agg_func}"
                    )
                    st.dataframe(grouped_df, use_container_width=True)

                    all_grouped_dfs.append(grouped_df)
                    grouped_df_i.append(f"GroupBy table for {group_col} with {agg_col} using {agg_func} is created and populated in the csv output file")
                    
                except Exception as e:
                    st.error(f"❌ Error during groupby aggregation: {e}")
        else:
            st.warning("⚠️ Need both categorical and numeric columns for groupby operations.")
            grouped_df_i.append("Skipping groupby operations due to lack of required columns.")
            
            

    else:
        st.write("Skipping groupby operations as per user input.")
        grouped_df_i.append("Skipping pivot table creation as per user input.")
        

    ### =============== PIVOT TABLE SECTION ===============
    pivot_proceed = st.selectbox(
        "\nDo you want to create pivot tables?",
        ["no", "yes"],
        key="pivot_proceed"
    )
    
    if pivot_proceed == "yes":
        st.subheader("📋 Pivot Table Operations")
        
        # Single pivot table operation with interactive widgets
        if len(cat_cols) > 0 and len(num_cols) > 0:
            # Select index column(s) for rows
            index_col = st.selectbox(
                "Select index column for rows (Categorical):",
                cat_cols,
                key="pivot_index_select"
            )
            
            # Select column for columns (optional)
            col_col = st.selectbox(
                "Select column for columns (optional) (Categorical):",
                ["None"] + cat_cols,
                key="pivot_col_select"
            )
            col_col = None if col_col == "None" else col_col
            
            # Select value column
            value_col = st.selectbox(
                "Select value column (Numerical):",
                num_cols,
                key="pivot_value_select"
            )
            
            # Select aggregation function
            agg_func = st.selectbox(
                "Select aggregation function:",
                ["mean", "sum", "count", "min", "max", "median", "std", "var"],
                key="pivot_agg_func_select"
            )
            
            if index_col and value_col and agg_func:
                try:
                    # Data validation and preprocessing
                    #st.write(f"**Data validation for pivot table creation:**")
                    
                    # Check if index column has unique values
                    unique_index_count = df[index_col].nunique()
                    total_rows = len(df)
                    #st.write(f"- Index column '{index_col}' has {unique_index_count} unique values out of {total_rows} total rows")
                    
                    # Check for any non-scalar values in index column
                    if df[index_col].dtype == 'object':
                        # Check for list-like values
                        sample_values = df[index_col].dropna().head(10)
                        #st.write(f"- Sample values in index column: {sample_values.tolist()}")
                        
                        # Convert any list-like strings to proper format
                        # Check if any values contain list-like characters
                        has_list_like = any('[' in str(val) or ']' in str(val) for val in sample_values)
                        if has_list_like:
                            st.warning("⚠️ Index column contains list-like values. Converting to string format.")
                            df_temp = df.copy()
                            df_temp[index_col] = df_temp[index_col].astype(str)
                        else:
                            df_temp = df.copy()
                    else:
                        df_temp = df.copy()
                    
                    # Check if column column has unique values (if specified)
                    if col_col:
                        unique_col_count = df_temp[col_col].nunique()
                        #st.write(f"- Column '{col_col}' has {unique_col_count} unique values")
                        
                        # Check for any non-scalar values in column column
                        if df_temp[col_col].dtype == 'object':
                            sample_col_values = df_temp[col_col].dropna().head(10)
                            #st.write(f"- Sample values in column: {sample_col_values.tolist()}")
                            
                            # Convert any list-like strings to proper format
                            # Check if any values contain list-like characters
                            has_list_like = any('[' in str(val) or ']' in str(val) for val in sample_col_values)
                            if has_list_like:
                                st.warning("⚠️ Column contains list-like values. Converting to string format.")
                                df_temp[col_col] = df_temp[col_col].astype(str)
                    
                    # Create pivot table with error handling
                    st.write("**Creating pivot table...**")
                    
                    if col_col:
                        pivot_df = pd.pivot_table(
                            df_temp,
                            index=index_col,
                            columns=col_col,
                            values=value_col,
                            aggfunc=agg_func,
                            fill_value=0
                        ).reset_index()
                    else:
                        # Simple pivot without column grouping
                        pivot_df = df_temp.groupby(index_col)[value_col].agg(agg_func).reset_index()
                        pivot_df.columns = [index_col, f'{agg_func.title()} of {value_col}']
                    
                    # Clean column names if MultiIndex
                    if isinstance(pivot_df.columns, pd.MultiIndex):
                        pivot_df.columns = ['_'.join(filter(None, map(str, col))).strip('_') for col in pivot_df.columns.values]
                    
                    st.write(f"**Pivot table: {agg_func.title()} of {value_col} by {index_col}**")
                    if col_col:
                        st.write(f"**Columns: {col_col}**")
                    
                    
                    
                    # Save CSV
                    index_str = index_col
                    col_str = col_col if col_col else "None"
                    val_str = f"{value_col}_{agg_func}"
                    filename = f"pivot_{index_str}_col_{col_str}_val_{val_str}.csv"
                    filepath = os.path.join(save_dir, filename)
                    pivot_df.to_csv(filepath, index=False)
                    
                    # Create download button for the pivot table result
                    csv_data = pivot_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Pivot Table ({filename})",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_pivot_{index_col}_{col_col}_{value_col}_{agg_func}"
                    )
                    st.dataframe(pivot_df, use_container_width=True)
                    # Update return variables
                    index_str = index_col
                    col_str = col_col if col_col else "None"
                    val_str = f"{value_col}_{agg_func}"
                    num_col = value_col
                    
                    grouped_df_i.append(f"Pivot table for {index_str} column with {col_str} and {val_str} created and populated as csv output.")
                except Exception as e:
                    st.error(f"❌ Error during pivot table creation: {e}")
                    st.write("**Troubleshooting tips:**")
                    st.write("- Check if the selected columns contain list-like values or special characters")
                    st.write("- Try selecting different columns for index or columns")
                    st.write("- Ensure the data is properly formatted for pivot operations")
                    st.write("- The 'countries' column might contain list-like values that need preprocessing")
        else:
            st.warning("⚠️ Need both categorical and numeric columns for pivot table operations.")
            grouped_df_i.append("Skipping pivot table operations due to lack of required columns.")
    else:
        st.write("Skipping pivot table creation as per user input.")
        grouped_df_i.append("Skipping pivot table creation as per user input.")
    

    return all_grouped_dfs, grouped_df_i, index_str, col_str, val_str, num_col, groupby_cols_str, agg_funcs_str