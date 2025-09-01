# This file contains the code that automatically detects and imputes missing value for columns in the dataset. In case of numerical data, based on skewness the imputation method will be mean or median. In case of categorical data it is mode. For all the other datatypes, the imputaton is skipped or NA. 
# 
import pandas as pd
import numpy as np
#from excel_utils.data_summary_plot import summarize_and_plot
from sklearn.impute import KNNImputer
import streamlit as st

def restore_original_df():
    """Restore the DataFrame to its original state."""
    if 'df_original' in st.session_state and st.session_state['df_original'] is not None:
        st.session_state['df_current'] = st.session_state['df_original'].copy()
        st.session_state['df_modified'] = st.session_state['df_original'].copy()

@st.cache_resource
def apply_imputation_plan(df, imputation_df, interpretations, cols_imputed_mean, cols_imputed_median, cols_imputed_mode):
    """Applies the imputation plan to the DataFrame."""
    for _, row in imputation_df.iterrows():
        col = row['Column']
        method = row['Suggested Imputation']
        
        if method == 'mean':
            impute_val = df[col].mean()
            df[col].fillna(impute_val, inplace=True)
            interpretations.append(f"Imputed '{col}' with mean.")
            cols_imputed_mean.append(col)
        elif method == 'median':
            impute_val = df[col].median()
            df[col].fillna(impute_val, inplace=True)
            interpretations.append(f"Imputed '{col}' with median.")
            cols_imputed_median.append(col)
        elif method == 'mode':
            impute_val = df[col].mode().iloc[0]
            df[col].fillna(impute_val, inplace=True)
            interpretations.append(f"Imputed '{col}' with mode.")
            cols_imputed_mode.append(col)
        elif method == 'NA':
            df[col].fillna("NA", inplace=True)
            interpretations.append(f"Imputed '{col}' with 'NA'.")
        else:
            interpretations.append(f"Skipped imputation for '{col}' as per the plan.")
    
    return df, interpretations, cols_imputed_mean, cols_imputed_median, cols_imputed_mode

def auto_missing_value_imputation(df, scales_of_measurement, threshold_pct=0.05):
    # Initialize DataFrame session states - only once, never reset
    cols_imputed_user = []
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = df.copy()

    df_original=st.session_state.get('df_original')
    total_rows = len(df_original)
    if 'cols_drop_org' not in st.session_state:
        st.session_state['cols_drop_org'] = []
    cols_drop_org= st.session_state.get('cols_drop_org')
    for col in df_original.columns:
        missing_count_org = df_original[col].isnull().sum()
        if missing_count_org > 0:
            missing_pct_org = missing_count_org / total_rows
            if missing_pct_org > 0.6:
                if col not in cols_drop_org:  # Only add if not already tracked
                    cols_drop_org.append(col)

    st.session_state['cols_drop_org'] = cols_drop_org
    st.session_state['df_original'] = df_original

    
    
    if 'df_current' not in st.session_state:
        st.session_state['df_current'] = df.copy()
    
    if 'df_modified' not in st.session_state:
        st.session_state['df_modified'] = df.copy()
    
    if 'scales_original' not in st.session_state:
        st.session_state['scales_original'] = scales_of_measurement.copy()
    
    # Initialize session management states
    if 'missing_imputation_session' not in st.session_state:
        st.session_state['missing_imputation_session'] = {
            'initialized': True,
            'high_missingness_processed': False,
            'user_choice_submitted': False
        }
    
    # Initialize other session states if not exists
    if 'high_missingness_operations' not in st.session_state:
        st.session_state['high_missingness_operations'] = []
    
    if 'imputation_operations' not in st.session_state:
        st.session_state['imputation_operations'] = []
    
    if 'form_inputs' not in st.session_state:
        st.session_state['form_inputs'] = {
            'cols_to_impute': [],
            'imputation_method': 'mean',
            'fixed_value': '',
            'knn_neighbors': 5
        }
    
    if 'user_choice' not in st.session_state:
        st.session_state['user_choice'] = ""
    
    interpretations = []
    
    # Use current DataFrame from session state, or create new one if needed
    if st.session_state['df_current'] is not None:
        df = st.session_state['df_current'].copy()
    else:
        df = df.copy().replace(r'^\s*$', np.nan, regex=True)
        st.session_state['df_current'] = df.copy()
    
    # Update scales if needed
    scales_of_measurement = st.session_state['scales_original'].copy()
    total_rows = len(df)
    missing_summary = []
    
    cols_imputed_mean = []
    cols_imputed_median = []
    cols_imputed_mode = []
    if 'cols_imputed_user' not in st.session_state:
        st.session_state['cols_imputed_user'] = []
    if 'cols_imputed_median_current' not in st.session_state:
        st.session_state['cols_imputed_median_current'] = []
    if 'cols_imputed_mode_current' not in st.session_state:
        st.session_state['cols_imputed_mode_current'] = []
    if 'cols_imputed_mean_current' not in st.session_state:
        st.session_state['cols_imputed_mean_current'] = []
    summary_message = []
 
    # Initialize high missingness operations list if not exists
    if 'high_missingness_operations' not in st.session_state:
        st.session_state['high_missingness_operations'] = []
    
    
    # Get the current list of columns to drop from session state
    cols_to_drop = st.session_state.get('cols_dropped_high_missingness', [])
    
    
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / total_rows
            if missing_pct > 0.6:
                if col not in cols_to_drop:  # Only add if not already tracked
                    cols_to_drop.append(col)
                    st.warning(f" Dropped column '{col}' due to high missing values ({missing_pct:.0%}).")
                    interpretations.append(f"Dropped '{col}' column due to high missing values ({missing_pct:.0%}).")

    if cols_to_drop:
        # Filter cols_to_drop to only include columns that actually exist in the DataFrame
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols_to_drop:
            df.drop(columns=existing_cols_to_drop, inplace=True)
            
            # Also filter scales_of_measurement to only remove columns that exist in it
            if scales_of_measurement is not None and not scales_of_measurement.empty:
                existing_scales_cols_to_drop = [col for col in existing_cols_to_drop if col in scales_of_measurement['Column'].values]
                if existing_scales_cols_to_drop:
                    scales_of_measurement = scales_of_measurement[~scales_of_measurement['Column'].isin(existing_scales_cols_to_drop)].reset_index(drop=True)
            
            # Update session states
            st.session_state['df_current'] = df.copy()
            st.session_state['scales_original'] = scales_of_measurement.copy()
            # Store the dropped columns in session state for persistence across reruns
            st.session_state['cols_dropped_high_missingness'] = existing_cols_to_drop
            
            st.success(f"✅ Successfully dropped {len(existing_cols_to_drop)} columns: {', '.join(existing_cols_to_drop)}")
        else:
            # If no columns exist to drop, clear the tracking
            st.session_state['cols_dropped_high_missingness'] = []
          
    
    imputation_plan = []
    scale_df_indexed = scales_of_measurement.set_index('Column')
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        missing_pct = missing_count / total_rows
        try:
            if col not in scale_df_indexed.index: continue
            col_type = scale_df_indexed.loc[col, 'Data type Category'].lower()
            if col_type in ['numerical', 'percentage', 'ratio']:
                mean = df[col].mean()
                median = df[col].median()
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                
                # Using the user's original heuristic
                if abs(mean - median) / mean < threshold_pct and abs(mean - mode_val) / mean < threshold_pct:
                    method = 'mean'
                else:
                    method = 'median'
            elif col_type in ['categorical']:
                method = 'mode'
            elif col_type == 'text':
                method = 'skipped (text)'
            elif col_type == 'datetime':
                method = 'NA'
            else:
                method = 'unknown type'
                
            imputation_plan.append([col, missing_count, total_rows, f"{missing_pct:.0%}", method])
            
        except KeyError:
            imputation_plan.append([col, missing_count, total_rows, f"{missing_pct:.0%}", 'unknown type'])
    
    # Create imputation plan DataFrame
    imputation_df = pd.DataFrame(imputation_plan, columns=['Column', 'Missing Count', 'Total Rows', 'Missing %', 'Suggested Imputation'])
    if 'imputation_df' not in st.session_state:
        st.session_state['imputation_df'] = imputation_df.copy()
    if imputation_df.empty:
        print("   - No columns with missing values remaining after the initial check.")
        interpretations.append("There are no missing values present in the dataset.")
    else:
        
        # Show detailed table if needed
        with st.expander("📊 Detailed Imputation Plan Table", expanded=True):
            st.dataframe(imputation_df, use_container_width=True)
    
    # --- Step 2: User Confirmation ---
    # Only show user confirmation if choice hasn't been submitted yet
    if not st.session_state['missing_imputation_session'].get('user_choice_submitted', False):
        # Initialize session state for user choice if not exists
        if 'user_choice' not in st.session_state:
            st.session_state['user_choice'] = ""
        
        choice = st.selectbox(
            "\nAre you fine with this imputation plan?",
            ["", "yes", "no"],
            index=["", "yes", "no"].index(st.session_state['user_choice']),
            key="imputation_choice"
        )
        
        # Submit button is always visible
        submit_clicked = st.button("Submit Choice", key="submit_imputation_choice", type="primary", disabled=(choice == ""))
        
        # Only proceed if submit button is clicked and a choice is made
        if submit_clicked and choice:
            st.session_state['user_choice'] = choice
            st.session_state['missing_imputation_session']['user_choice_submitted'] = True
            st.success(f"✅ Choice submitted: {choice}")
            st.rerun()  # Rerun to show the next section
        elif submit_clicked and choice == "":
            st.warning("⚠️ Please select a choice before submitting.")
        elif not submit_clicked:
            if choice == "no":
                st.info("✅ You have selected 'no' for manual imputation. Please click 'Submit Choice' to proceed to the manual imputation section.")
            elif choice == "yes":
                st.info("✅ You have selected 'yes' for automatic imputation. Please click 'Submit Choice' to proceed.")
            else:
                st.info("Please select your choice and click 'Submit Choice' to proceed.")
    else:
        # User choice has already been submitted, get it from session state
        choice = st.session_state['user_choice']
    
    # Check if user choice has been submitted and process accordingly
    if choice == 'yes' and st.session_state['missing_imputation_session'].get('user_choice_submitted', False):
        # --- Apply the suggested imputation methods ---
        df = st.session_state['df_current'].copy()
        
        # Track which columns were imputed with which method
        cols_imputed_mean_current = []
        cols_imputed_median_current = []
        cols_imputed_mode_current = []
        
        for _, row in imputation_df.iterrows():
            col = row['Column']
            method = row['Suggested Imputation']
            
            if method == 'mean':
                impute_val = df[col].mean()
                df[col].fillna(impute_val, inplace=True)
                interpretations.append(f"Imputed '{col}' with mean.")
                cols_imputed_mean_current.append(col)
            elif method == 'median':
                impute_val = df[col].median()
                df[col].fillna(impute_val, inplace=True)
                interpretations.append(f"Imputed '{col}' with median.")
                cols_imputed_median_current.append(col)
            elif method == 'mode':
                impute_val = df[col].mode().iloc[0]
                df[col].fillna(impute_val, inplace=True)
                interpretations.append(f"Imputed '{col}' with mode.")
                cols_imputed_mode_current.append(col)
            elif method == 'NA':
                df[col].fillna("NA", inplace=True)
                interpretations.append(f"Imputed '{col}' with 'NA'.")
            else:
                interpretations.append(f"Skipped imputation for '{col}' as per the plan.")
        
        # Update session states with imputation tracking
        st.session_state['cols_imputed_mean_current'] = cols_imputed_mean_current
        st.session_state['cols_imputed_median_current'] = cols_imputed_median_current
        st.session_state['cols_imputed_mode_current'] = cols_imputed_mode_current
        
        # Update DataFrame session states
        st.session_state['df_current'] = df.copy()
        st.session_state['df_modified'] = df.copy()
        st.write("\n Imputation plan applied successfully.")
        #return df, imputation_df, missing_summary, interpretations, scales_of_measurement, summary_message
        
    elif choice == 'no' and st.session_state['missing_imputation_session'].get('user_choice_submitted', False):
        st.write("\nUser chose to manually handle missing values. Entering interactive mode...")
        df = st.session_state['df_current'].copy()
        # Initialize imputation operations list if not exists
        if 'imputation_operations' not in st.session_state:
            st.session_state['imputation_operations'] = []
        
        # Get columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.success("✅ All missing values have been handled!")
            #return df, imputation_df, missing_summary, interpretations, scales_of_measurement, summary_message
        
        # Create scale_df_indexed for filtering text columns
        if scales_of_measurement is not None and not scales_of_measurement.empty:
            scale_df_indexed = scales_of_measurement.set_index('Column')
            available_cols = [col for col in missing_cols if col in scale_df_indexed.index and 
                              scale_df_indexed.loc[col, 'Data type Category'].lower() != 'text']
        else:
            available_cols = missing_cols  # Fallback if no scales data
        
        if not available_cols:
            st.warning("⚠️ No suitable columns found for imputation (text columns are excluded).")
            #return df, imputation_df, missing_summary, interpretations, scales_of_measurement, summary_message
        
        # Initialize session state for form inputs if not exists
        if 'form_inputs' not in st.session_state:
            st.session_state['form_inputs'] = {
                'cols_to_impute': [],
                'imputation_method': 'mean',
                'fixed_value': '',
                'knn_neighbors': 5
            }
        
        # Form for adding new imputation operations
        with st.form("add_imputation_operation"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cols_to_impute = st.multiselect(
                    "Select columns to impute:",
                    available_cols,
                    default=st.session_state['form_inputs']['cols_to_impute'],
                    key="cols_to_impute_input"
                )
            
            with col2:
                method_options = ["mean", "median", "mode", "min", "max", "zero", "fixed_value", "forward_fill", "backward_fill", "knn"]
                imputation_method = st.selectbox(
                    "Choose imputation method:",
                    method_options,
                    index=method_options.index(st.session_state['form_inputs']['imputation_method']),
                    key="method_select"
                )
            
            with col3:
                # Additional parameters for specific methods
                if imputation_method == "fixed_value":
                    fixed_val = st.text_input(
                        "Fixed value:", 
                        value=st.session_state['form_inputs']['fixed_value'],
                        key="fixed_value_input"
                    )
                elif imputation_method == "knn":
                    n_neighbors = st.number_input(
                        "Number of neighbors:", 
                        min_value=1, 
                        max_value=20, 
                        value=st.session_state['form_inputs']['knn_neighbors'],
                        key="knn_neighbors"
                    )
                else:
                    st.write("")  # Spacer
            
            
            #submitted = st.form_submit_button("Run Imputation")
       
            cols_imputed_user.append(f"{cols_to_impute} ({imputation_method})")
            st.session_state['cols_imputed_user'] = cols_imputed_user
            submitted = st.form_submit_button("Add Imputation Operation")
            if submitted:
                if cols_to_impute:
                    # Update session state with form values
                    st.session_state['form_inputs']['cols_to_impute'] = cols_to_impute
                    st.session_state['form_inputs']['imputation_method'] = imputation_method
                    
                    if imputation_method == "fixed_value":
                        st.session_state['form_inputs']['fixed_value'] = fixed_val
                    elif imputation_method == "knn":
                        st.session_state['form_inputs']['knn_neighbors'] = n_neighbors
                    
                    operation = {
                        'columns': cols_to_impute,
                        'method': imputation_method,
                        'fixed_value': fixed_val if imputation_method == "fixed_value" else None,
                        'knn_neighbors': n_neighbors if imputation_method == "knn" else None
                    }
                    
                    st.session_state['imputation_operations'].append(operation)
                    
                    # Don't clear form inputs - keep them for better user experience
                    st.rerun()
                else:
                    st.warning("Please select at least one column for imputation.")
        
        # Display current imputation operations
        if st.session_state['imputation_operations']:
            st.write("**Imputation operations to be applied:**")
            
            for i, op in enumerate(st.session_state['imputation_operations']):
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    method_display = op['method']
                    if op['method'] == "fixed_value" and op['fixed_value']:
                        method_display += f" ({op['fixed_value']})"
                    elif op['method'] == "knn" and op['knn_neighbors']:
                        method_display += f" ({op['knn_neighbors']} neighbors)"
                    
                    st.write(f"🔧 {method_display} on: {', '.join(op['columns'])}")
                with col2:
                    if st.button("Remove", key=f"remove_imputation_op_{i}"):
                        st.session_state['imputation_operations'].pop(i)
                        st.rerun()
                with col3:
                    st.write("")  # Spacer
        
        # Apply imputation operations button
        if st.session_state['imputation_operations']:
            if st.button("🚀 Apply All Imputation Operations", type="primary"):
                for i, op in enumerate(st.session_state['imputation_operations']):
                    cols_to_impute = op['columns']
                    method = op['method']
                    
                    for col in cols_to_impute:
                        try:
                            if method == "mean":
                                val = df[col].mean()
                                df[col].fillna(val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with mean: {val:.4f}")
                                
                            elif method == "median":
                                val = df[col].median()
                                df[col].fillna(val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with median: {val:.4f}")
                                
                            elif method == "mode":
                                val = df[col].mode().iloc[0]
                                df[col].fillna(val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with mode: {val}")
                                
                            elif method == "min":
                                val = df[col].min()
                                df[col].fillna(val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with min: {val:.4f}")
                                
                            elif method == "max":
                                val = df[col].max()
                                df[col].fillna(val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with max: {val:.4f}")
                                
                            elif method == "zero":
                                df[col].fillna(0, inplace=True)
                                interpretations.append(f"Imputed '{col}' with zero")
                                
                            elif method == "fixed_value":
                                fixed_val = op['fixed_value']
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    fixed_val = pd.to_numeric(fixed_val, errors='coerce')
                                df[col].fillna(fixed_val, inplace=True)
                                interpretations.append(f"Imputed '{col}' with fixed value: {fixed_val}")
                                
                            elif method == "forward_fill":
                                df[col].fillna(method='ffill', inplace=True)
                                interpretations.append(f"Imputed '{col}' with forward fill")
                                
                            elif method == "backward_fill":
                                df[col].fillna(method='bfill', inplace=True)
                                interpretations.append(f"Imputed '{col}' with backward fill")
                                
                            elif method == "knn":
                                if not pd.api.types.is_numeric_dtype(df[col]):
                                    st.warning(f"⚠️ KNN imputation skipped for '{col}' (not numeric)")
                                    continue
                                
                                n_neighbors = op['knn_neighbors']
                                numeric_cols = df.select_dtypes(include=np.number).columns
                                imputer = KNNImputer(n_neighbors=n_neighbors)
                                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                                interpretations.append(f"Imputed '{col}' using KNN with {n_neighbors} neighbors")
                            
                            st.success(f"✅ Imputed '{col}' using {method}")
                            
                        except Exception as e:
                            st.error(f"❌ Error during imputation for '{col}': {e}")

                    # --- FIX: AFTER MANUAL IMPUTATION, APPLY THE PLAN TO REMAINING COLUMNS ---
                missing_cols_after_manual = df.columns[df.isnull().any()].tolist()
                if missing_cols_after_manual:
                    print("\nUser finished manual imputation. Automatically handling remaining missing values...")
                    
                    # Filter the imputation plan to only include remaining columns
                    remaining_imputation_df = imputation_df[imputation_df['Column'].isin(missing_cols_after_manual)].copy()
                    
                    # Apply the automatic plan to the remaining columns
                    df, interpretations, cols_imputed_mean_current, cols_imputed_median_current, cols_imputed_mode_current= apply_imputation_plan(df, remaining_imputation_df, interpretations, cols_imputed_mean, cols_imputed_median, cols_imputed_mode)
                    st.write("Automatic imputation of remaining columns completed.")
                    st.session_state['cols_imputed_mean_current'] = cols_imputed_mean_current
                    st.session_state['cols_imputed_median_current'] = cols_imputed_median_current
                    st.session_state['cols_imputed_mode_current'] = cols_imputed_mode_current

                # Update session states after all imputations
                st.session_state['df_current'] = df.copy()
                st.session_state['df_modified'] = df.copy()
                
                total_ops = len(st.session_state['imputation_operations'])
                st.success(f"✅ Applied {total_ops} imputation operations!")
                
                # Clear operations after applying
                st.session_state['imputation_operations'] = []
                
                st.write("Imputation operations applied successfully.")
        
        # Add navigation and reset buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔙 Back to Choice Selection", key="back_to_choice"):
                st.session_state['missing_imputation_session']['user_choice_submitted'] = False
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Session", key="reset_session"):
                st.session_state['missing_imputation_session']['initialized'] = False
                st.rerun()
        
        # Return current state - function will be called again on next interaction
        st.session_state.modified_df=df.copy()

    # --- Step 3: Final Output ---
    cols_imputed_user = st.session_state.get('cols_imputed_user', [])
    cols_to_drop = st.session_state.get('cols_dropped_high_missingness', [])  # Get from session state
    cols_imputed_median_current = st.session_state.get('cols_imputed_median_current', [])
    cols_imputed_mode_current = st.session_state.get('cols_imputed_mode_current', [])
    cols_imputed_mean_current = st.session_state.get('cols_imputed_mean_current', [])
    cols_drop_org = st.session_state.get('cols_drop_org', [])
    # Fix the summary message to correctly describe dropped columns
    summary_message.append(f"Total columns dropped due to high missingness ({len(cols_drop_org)}): <b>[{', '.join(cols_drop_org)}]</b>")

    summary_message.append(f"Total columns imputed with Mean ({len(cols_imputed_mean_current)}): <b>[{', '.join(cols_imputed_mean_current)}]</b>")
    summary_message.append(f"Total columns imputed with Median ({len(cols_imputed_median_current)}): <b>[{', '.join(cols_imputed_median_current)}]</b>")
    summary_message.append(f"Total columns imputed with Mode ({len(cols_imputed_mode_current)}): <b>[{', '.join(cols_imputed_mode_current)}]</b>")
   
    summary_message.append(f"Total columns imputed based on user input ({len(cols_imputed_user)}): <b>[{', '.join(cols_imputed_user)}]</b>")
    
    # Use the current DataFrame from session state
    df = st.session_state['df_current'].copy()
    imputation_df=st.session_state.get('imputation_df', pd.DataFrame())
    return df, imputation_df, missing_summary, interpretations, scales_of_measurement, summary_message