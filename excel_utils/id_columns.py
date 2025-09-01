import streamlit as st


def remove_id_columns(df, user_additional_id_col=None, skip_id_cols_confirmation=False):
    if df.empty:
        st.warning("The DataFrame is empty. No columns to remove.")
        return df.copy(), []

    columns_to_drop = []
    id_patterns = ['id', 'ID', 'Id']

    st.info(f"Columns in the DataFrame: {list(df.columns)}")

    st.subheader("Checking for ID-like Columns")
    for col in df.columns:
        col_lower = str(col).lower()
        if any(col_lower.startswith(pattern) or col_lower.endswith(pattern) for pattern in id_patterns):
            columns_to_drop.append(col)

    if columns_to_drop:
        st.info(f"Auto-detected ID columns: {columns_to_drop}")
    else:
        st.info("No ID-like columns detected.")
        return df.copy(), []

    # If user wants to add more manually
    if user_additional_id_col:
        columns_to_drop.append(user_additional_id_col)

    # If user wants to skip confirmation, return cleaned DataFrame
    if skip_id_cols_confirmation:
        try:
            df_cleaned = df.drop(columns=columns_to_drop)
            # Ensure we return the correct types
            if not isinstance(columns_to_drop, list):
                st.error(f"Error: columns_to_drop is not a list: {type(columns_to_drop)}")
                columns_to_drop = []
            return df_cleaned, columns_to_drop
        except Exception as e:
            st.error(f"Error removing ID columns: {str(e)}")
            return df.copy(), []

    # If confirmation is required, show interactive UI
    st.subheader("ID Column Removal Confirmation")
    st.write(f"**Detected ID columns:** {', '.join(columns_to_drop)}")
    
    # Show preview of what will be removed
    st.write("**Preview of columns to be removed:**")
    preview_df = df[columns_to_drop].head()
    st.dataframe(preview_df)
    
    # Confirmation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Confirm & Remove ID Columns", type="primary"):
            try:
                df_cleaned = df.drop(columns=columns_to_drop)
                st.success(f"✅ Successfully removed {len(columns_to_drop)} ID columns: {', '.join(columns_to_drop)}")
                return df_cleaned, columns_to_drop
            except Exception as e:
                st.error(f"Error removing ID columns: {str(e)}")
                return df.copy(), []
    
    with col2:
        if st.button("❌ Cancel & Keep ID Columns"):
            st.info("ID column removal cancelled. Keeping all columns.")
            return df.copy(), []
    
    # If no button has been clicked yet, return current state
    # This allows the function to be called again on the next interaction
    # Ensure we always return the correct types
    if not isinstance(columns_to_drop, list):
        st.error(f"Error: columns_to_drop is not a list: {type(columns_to_drop)}")
        columns_to_drop = []
    return df, columns_to_drop


