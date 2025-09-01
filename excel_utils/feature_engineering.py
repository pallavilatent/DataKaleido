import numpy as np
import pandas as pd
from scipy import stats
from excel_utils.scales_of_measurement import detect_scales_of_measurement
import os

# Check if Streamlit is available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def suggest_boxcox_transformation(lmbda):
    """Provides a textual interpretation for a given lambda value."""
    if lmbda < -1:
        return "Inverse Transformation"
    elif -1 <= lmbda < -0.5:
        return "Inverse Square Root Transformation"
    elif -0.5 <= lmbda < 0.5:
        return "Log Transformation"
    elif 0.5 <= lmbda < 1:
        return "Square Root Transformation"
    else:
        return "No Transformation needed"

@st.cache_resource
def proceed_yes(df, numeric_cols, datetime_cols, skew_interpretation, boxcox_interpretation, time_features_i):
    """Performs feature engineering steps in interactive mode."""
    st.write("### 🚀 Feature Engineering")
    st.subheader("📊 Skewness Analysis & Transformations and Datetime Extractions")
    
    if 'transformed_cols' not in st.session_state:
        st.session_state.transformed_cols = []

    filtered_numeric_cols = [
        col for col in numeric_cols 
        if col in df.columns and 
        df[col].dtype in ['int64', 'float64'] and 
        not pd.api.types.is_bool_dtype(df[col]) and 
        col not in st.session_state.transformed_cols
    ]

    for col in filtered_numeric_cols:
        transformed_status = "no_transformation"
        suggested = "N/A"
        try:
            skew_val = df[col].skew()
            if abs(skew_val) > 1:
                st.write(f"Column '{col}' has skewness: {round(skew_val, 2)}.")
                try:
                    df_col_positive = df[col].copy()
                    if (df_col_positive <= 0).any():
                        df_col_positive += abs(df_col_positive.min()) + 1
                    
                    transformed, lmbda = stats.boxcox(df_col_positive.dropna())
                    suggested = suggest_boxcox_transformation(lmbda)

                    if suggested != "No Transformation needed":
                        if suggested == "Square":
                            df[f'{col}_square'] = df[col] ** 2
                        elif suggested == "Square Root":
                            df[f'{col}_sqrt'] = np.sqrt(df[col])
                        elif suggested == "Inverse Square":
                            df[f'{col}_invsq'] = 1 / (df[col] ** 2).replace(0, np.nan)
                        elif suggested == "Inverse Square Root":
                            df[f'{col}_invsqrt'] = 1 / np.sqrt(df[col].replace(0, np.nan))
                        elif suggested == "Natural Log":
                            df[f'{col}_log'] = np.log(df[col].replace(0, np.nan))
                        elif suggested == "Reciprocal":
                            df[f'{col}_reciprocal'] = 1 / df[col].replace(0, np.nan)
                        
                        transformed_status = "transformed"
                        st.session_state.transformed_cols.append(col)
                    else:
                        transformed_status = "no_transformation_needed"
                except Exception as e:
                    transformed_status = "skipped"
            else:
                transformed_status = "no_transformation"

        except Exception as e:
            transformed_status = "error"
            
        if transformed_status == "transformed":
            st.write(f"Box-Cox Lambda for '{col}': {round(lmbda, 2)}.")
            st.write(f"Transformation for '{col}': {suggested}.")
            boxcox_interpretation.append(f"<b>{col}</b> has been successfully transformed using <b>{suggested}</b>.")
            skew_interpretation.append(f"Since <b>{col}</b> has a skewed distribution, based on the Box-Cox Lambda values, consider transforming the variable.")
        elif transformed_status == "no_transformation":
            st.info(f"The'{col}' is Approximately Symmetric.")
            skew_interpretation.append(f"The <b>{col}</b> is Approximately Symmetric.")
            boxcox_interpretation.append(f"No transformation is required for <b>{col}</b>.")
        elif transformed_status == "no_transformation_needed":
            st.write(f"Box-Cox Lambda for '{col}': {round(lmbda, 2)}.")
            st.info(f"The Numerical variable '{col}' is skewed but no transformation was performed as per the Box-Cox suggestion.")
            boxcox_interpretation.append(f"No transformation is required for <b>{col}</b> as per the Box-Cox suggestion.")
        elif transformed_status == "skipped":
            st.warning(f"Skipped transformation for '{col}' due to an error during Box-Cox computation.")
            boxcox_interpretation.append(f"Skipped transformation for '{col}' due to an error.")
        elif transformed_status == "error":
            st.error(f"Could not analyze skewness for '{col}': {e}")
            
    st.success("✅ Skewness analysis completed!")
    
    if datetime_cols:
        for col in datetime_cols:
            print(f"\n Extracting time features from '{col}'...")
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='raise')

                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.day_name()
                df[f'{col}_weekday_num'] = df[col].dt.weekday
                df[f'{col}_is_weekend'] = df[col].dt.weekday >= 5
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_second'] = df[col].dt.second
                
                print(f"Time features extracted and added for '{col}'.")
                time_features_i.append(f"Time features are extracted and added for <b>{col}</b>. Extracted features include year, quarter, month, date, weekdays and time components.")
                st.session_state["time_features_i"] = time_features_i
            except Exception as e:
                print(f"Skipping datetime extraction for '{col}' — not a valid datetime column. Error: {e}")
    else:
        print("No datetime variable columns present in the dataset.")
        time_features_i.append("No datetime variable columns present in the dataset.")
        st.session_state["time_features_i"] = time_features_i
    
    return df, skew_interpretation, boxcox_interpretation, time_features_i

@st.cache_resource
def proceed_scale_norm_fun(df, scale_i, operation_type, cols_to_process):
    if operation_type == 'normalizing':
        for col in cols_to_process:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                try:
                    df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
                except Exception as e:
                    st.warning(f"Could not perform normalization on '{col}': {e}")
            else:
                st.warning(f"Skipping normalization for '{col}' - not a numeric column")
        st.write(f"Normalization performed on: {', '.join(cols_to_process)}")
        scale_i.append(f"Normalization performed on: <b>{', '.join(cols_to_process)}</b>")
        st.session_state["scale_i"] = scale_i
    elif operation_type == 'scaling':
        for col in cols_to_process:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                try:
                    df[f'{col}_scaled'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                except Exception as e:
                    st.warning(f"Could not perform scaling on '{col}': {e}")
            else:
                st.warning(f"Skipping scaling for '{col}' - not a numeric column")
        st.write(f"Scaling performed on: {', '.join(cols_to_process)}")
        scale_i = st.session_state.get("scale_i", [])
        scale_i.append(f"Scaling performed on: <b>{', '.join(cols_to_process)}</b>")
        st.session_state["scale_i"] = scale_i

    return df, scale_i

@st.cache_resource
def proceed_arith_fun(df, numeric_cols, arith_i, cont1, cont2, arith_op):
    if cont1 in numeric_cols and cont2 in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[cont1]) and not pd.api.types.is_bool_dtype(df[cont1]) and \
           pd.api.types.is_numeric_dtype(df[cont2]) and not pd.api.types.is_bool_dtype(df[cont2]):
            try:
                if arith_op == 'add':
                    df[f'{cont1}_plus_{cont2}'] = df[cont1] + df[cont2]
                elif arith_op == 'subtract':
                    df[f'{cont1}_minus_{cont2}'] = df[cont1] - df[cont2]
                elif arith_op == 'multiply':
                    df[f'{cont1}_mul_{cont2}'] = df[cont1] * df[cont2]
                elif arith_op == 'divide':
                    df[f'{cont1}_div_{cont2}'] = df[cont1] / df[cont2].replace(0, np.nan)
                elif arith_op in ['lag-1', 'lag-7', 'lag-14', 'lag-28', 'lag-30']:
                    lag_mapping = {'lag-1': 1, 'lag-7': 7, 'lag-14': 14, 'lag-28': 28, 'lag-30': 30}
                    lag_val = lag_mapping.get(arith_op, 1)
                    df[f'{cont1}_lag_{lag_val}'] = df[cont1].shift(lag_val)
                    df[f'{cont2}_lag_{lag_val}'] = df[cont2].shift(lag_val)
                
                arith_i.append(f"<b>{arith_op}</b> Operation is performed and new feature(s) are created based on <b>{cont1}</b> and <b>{cont2}</b>.")
                st.session_state["arith_i"] = arith_i
            except Exception as e:
                st.warning(f"Could not perform {arith_op} operation on '{cont1}' and '{cont2}': {e}")
                
        else:
            st.warning(f"Skipping {arith_op} operation - one or both columns are not numeric")
            arith_i.append(f"Skipping {arith_op} operation - one or both columns are not numeric")
            st.session_state["arith_i"] = arith_i
            
    return df, arith_i

@st.cache_resource
def proceed_binning(df, numeric_cols, binning_i, cols_to_bin, method, n_bins, bin_labels, custom_bins):
    try:
        for col in cols_to_bin:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                if method == "Equal-width":
                    df[f"{col}_binned_width"] = pd.cut(
                        df[col],
                        bins=n_bins,
                        labels=bin_labels if bin_labels and len(bin_labels) == n_bins else False,
                        include_lowest=True, 
                        right=False, 
                        duplicates='drop'
                    )
                    binning_i.append(f"Performed <b>Equal-width binning</b> for <b>{col}</b> with <b>{n_bins}</b> bins.")
                    st.session_state["binning_i"] = binning_i
                elif method == "Equal-frequency":
                    df[f"{col}_binned_freq"] = pd.qcut(
                        df[col].rank(method="first"),
                        q=n_bins, 
                        labels=bin_labels if bin_labels and len(bin_labels) == n_bins else False,
                        duplicates='drop'
                    )
                    binning_i.append(f"Performed <b>Equal-frequency</b> binning for <b>{col}</b> with <b>{n_bins}</b> bins.")
                    st.session_state["binning_i"] = binning_i
                elif method == "Custom":
                    if custom_bins:
                        custom_bins_list = [float(x.strip()) for x in custom_bins.split(",")]
                        df[f"{col}_binned_custom"] = pd.cut(
                            df[col],
                            bins=custom_bins_list,
                            labels=bin_labels if bin_labels and len(bin_labels) == (len(custom_bins_list) - 1) else False
                        )
                        binning_i = st.session_state.get("binning_i", [])
                        binning_i.append(f"Performed <b>Custom binning</b> for <b>{col}</b> with <b>{len(custom_bins_list)}</b> bins.")
                        st.session_state["binning_i"] = binning_i
            else:
                st.warning(f"Skipping binning for '{col}' - not a numeric column (dtype: {df[col].dtype})")
                binning_i = st.session_state.get("binning_i", [])
                binning_i.append(f"Skipping binning for '{col}' - not a numeric column (dtype: {df[col].dtype})")
                st.session_state["binning_i"] = binning_i
                continue
        
        st.write(f"✅ Binning completed for {len(cols_to_bin)} columns using {method} method!")
    except Exception as e:
        st.error(f"Error during binning: {e}")
    return df, binning_i

@st.cache_resource
def proceed_one_hot_encoding(df, cat_cols, one_hot_encoding_interpretation, ordinal_columns_i, onehot_operations, label_operations):
    # Apply One-Hot encoding operations
    for op in onehot_operations:
        cols_to_encode = op['columns']
        if not cols_to_encode:
            continue
        
        df_encoded = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode, drop_first=False, dtype=int)
        df = pd.concat([df, df_encoded], axis=1)
        
        one_hot_encoding_interpretation.append(f"The following columns are One-Hot encoded based on user input: <b>{', '.join(cols_to_encode)}</b>")
        st.session_state["one_hot_encoding_interpretation"] = one_hot_encoding_interpretation

    # Apply Label encoding operations
    for op in label_operations:
        cols_to_encode = op['columns']
        start_val = 1
        for col in cols_to_encode:
            categories = sorted(df[col].dropna().unique().tolist())
            mapping = {cat: idx + start_val for idx, cat in enumerate(categories)}
            encoded_col_name = col + "_label_encoded"
            df[encoded_col_name] = df[col].map(mapping)
            ordinal_columns_i.append(f"Label Encoding is performed on <b>{col}</b> with value starting from {start_val}.")
            st.session_state["ordinal_columns_i"] = ordinal_columns_i
    
    total_ops = len(onehot_operations) + len(label_operations)
    st.success(f"✅ Applied {total_ops} encoding operations!")
    
    st.session_state.onehot_operations = []
    st.session_state.label_operations = []
    
    return df, one_hot_encoding_interpretation, ordinal_columns_i

# Non-interactive mode function
def proceed_no(df, numeric_cols, cat_cols, datetime_cols):
    """Performs automatic feature engineering steps for non-interactive mode."""
    skew_interpretation = []
    boxcox_interpretation = []
    one_hot_encoding_interpretation = []
    ordinal_columns_i = []
    arith_i = []
    time_features_i = []
    binning_i = []
    scale_i = []

    # Automatic skewness and transformation
    for col in numeric_cols:
        try:
            if df[col].skew() > 1 or df[col].skew() < -1:
                skew_interpretation.append(f"Since <b>{col}</b> has a skewed distribution, a Box-Cox transformation was applied.")
                df_col_positive = df[col].copy()
                if (df_col_positive <= 0).any():
                    df_col_positive += abs(df_col_positive.min()) + 1
                transformed, lmbda = stats.boxcox(df_col_positive.dropna())
                suggested = suggest_boxcox_transformation(lmbda)
                boxcox_interpretation.append(f"<b>{col}</b> has been successfully transformed using <b>{suggested}</b>.")
                if suggested == "Square":
                    df[f'{col}_square'] = df[col] ** 2
                elif suggested == "Square Root":
                    df[f'{col}_sqrt'] = np.sqrt(df[col])
                elif suggested == "Inverse Square":
                    df[f'{col}_invsq'] = 1 / (df[col] ** 2).replace(0, np.nan)
                elif suggested == "Inverse Square Root":
                    df[f'{col}_invsqrt'] = 1 / np.sqrt(df[col].replace(0, np.nan))
                elif suggested == "Natural Log":
                    df[f'{col}_log'] = np.log(df[col].replace(0, np.nan))
                elif suggested == "Reciprocal":
                    df[f'{col}_reciprocal'] = 1 / df[col].replace(0, np.nan)
        except Exception as e:
            skew_interpretation.append(f"Skipped Box-Cox for '{col}' due to error: {e}.")

    # Automatic scaling
    for col in numeric_cols[:3]:
        try:
            df[f'{col}_std'] = (df[col] - df[col].mean()) / df[col].std()
            scale_i.append(f"Standardization applied to <b>{col}</b>")
        except:
            pass
    
    # Automatic arithmetic operations
    if len(numeric_cols) >= 2:
        try:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            arith_i.append(f"Addition operation: {col1} + {col2}")
        except:
            pass
    
    # Automatic binning
    if len(numeric_cols) >= 1:
        try:
            col = numeric_cols[0]
            df[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            binning_i.append(f"Equal-frequency binning applied to {col}")
        except:
            pass
    
    # Automatic categorical encoding
    if len(cat_cols) > 0:
        try:
            for col in cat_cols[:2]:
                df_encoded = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
                df = pd.concat([df, df_encoded], axis=1)
                one_hot_encoding_interpretation.append(f"One-hot encoding applied to {col}")
            
            for col in cat_cols[:2]:
                categories = sorted(df[col].dropna().unique().tolist())
                mapping = {cat: idx + 1 for idx, cat in enumerate(categories)}
                df[f'{col}_label_encoded'] = df[col].map(mapping)
                ordinal_columns_i.append(f"Label encoding applied to {col}")
        except:
            pass

    # Automatic datetime feature extraction
    if datetime_cols:
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                time_features_i.append(f"Time features extracted from <b>{col}</b>.")
            except:
                pass
    
    # Add final interpretation messages
    if not skew_interpretation:
        skew_interpretation.append("No numerical variables required Box-Cox transformation.")
    if not time_features_i:
        time_features_i.append("No datetime variables found.")
    if not one_hot_encoding_interpretation:
        one_hot_encoding_interpretation.append("No categorical variables found for one-hot encoding.")
    if not ordinal_columns_i:
        ordinal_columns_i.append("No categorical variables found for label encoding.")
    if not arith_i:
        arith_i.append("No numerical variables found for arithmetic operations.")
    if not binning_i:
        binning_i.append("No numerical variables found for binning.")
    if not scale_i:
        scale_i.append("No numerical variables found for scaling or normalization.")

    return df, skew_interpretation, boxcox_interpretation, one_hot_encoding_interpretation, ordinal_columns_i, arith_i, time_features_i, binning_i, scale_i, 1

# Main function
def feature_engineering(df, data_type_df, save_dir='output'):
    
    skew_interpretation = []
    boxcox_interpretation = []
    one_hot_encoding_interpretation = []
    ordinal_columns_i = []
    arith_i = []
    time_features_i = []
    binning_i = []
    scale_i = []
    flag = 1
    
    numeric_cols = data_type_df.loc[data_type_df['Data type Category'].isin(['Numerical', 'Percentage','Ratio'])]['Column'].tolist()
    cat_cols = data_type_df.loc[data_type_df['Data type Category'] == 'Categorical', 'Column'].tolist()
    datetime_cols = data_type_df.loc[data_type_df['Data type Category'] == 'Datetime', 'Column'].tolist()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if STREAMLIT_AVAILABLE:
        if 'fe_state' not in st.session_state:
            st.session_state.fe_state = {
                'step': 'start',
                'df': df.copy(),
                'skew_i': [], 'boxcox_i': [], 'time_i': [], 'scale_i': [], 
                'arith_i': [], 'binning_i': [], 'onehot_i': [], 'label_i': []
            }
        
        # Initialize lists to store operations if they don't exist
        if 'scaling_operations' not in st.session_state:
            st.session_state.scaling_operations = []
        if 'arithmetic_operations' not in st.session_state:
            st.session_state.arithmetic_operations = []
        if 'binning_operations' not in st.session_state:
            st.session_state.binning_operations = []
        if 'onehot_operations' not in st.session_state:
            st.session_state.onehot_operations = []
        if 'label_operations' not in st.session_state:
            st.session_state.label_operations = []
        
        state = st.session_state.fe_state
        df = state['df']

        st.write("### 🚀 Feature Engineering")

        # Step 1: Skewness, Transformation, and Datetime Features
        if state['step'] == 'start':
            st.subheader("Step 1: Skewness & Datetime Features")
            if st.button("Start Step 1: Analyze & Extract"):
                df, state['skew_i'], state['boxcox_i'], state['time_i'] = proceed_yes(df, numeric_cols, datetime_cols, [], [], [])
                state['df'] = df.copy()
                state['step'] = 'scale_norm'
                st.success("✅ Step 1 Completed!")
                st.rerun()

        # Step 2: Scaling & Normalizing
        if state['step'] == 'scale_norm':
            st.subheader("Step 2: Scale/Normalize Features")
            
            with st.form("scaling_form"):
                st.write("Add one or more scaling/normalization operations.")
                cols_to_process = st.multiselect("Select columns to scale/normalize:", numeric_cols, key="scale_cols")
                operation_type = st.selectbox("Choose operation:", ["scaling", "normalizing"], key="scale_op_type")
                add_operation = st.form_submit_button("Add Operation")
            
            if add_operation and cols_to_process:
                st.session_state.scaling_operations.append({'columns': cols_to_process, 'operation': operation_type})
                st.info(f"Added {operation_type} operation for: {', '.join(cols_to_process)}")
                st.rerun()

            if st.session_state.scaling_operations:
                st.write("**Pending Scaling Operations:**")
                for i, op in enumerate(st.session_state.scaling_operations):
                    st.write(f"{i+1}. {op['operation']} on: {', '.join(op['columns'])}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Apply All Scaling Operations", type="primary"):
                    for op in st.session_state.scaling_operations:
                        df, state['scale_i'] = proceed_scale_norm_fun(df, state['scale_i'], op['operation'], op['columns'])
                    state['df'] = df.copy()
                    st.session_state.scaling_operations = [] # Clear the queue
                    state['step'] = 'arithmetic'
                    st.success("✅ Step 2 Completed!")
                    st.rerun()
            with col2:
                if st.button("⏩ Skip Step 2"):
                    state['scale_i'].append("Skipped feature scaling and normalization as per user's input.")
                    state['step'] = 'arithmetic'
                    st.success("✅ Step 2 Skipped!")
                    st.rerun()

        # Step 3: Arithmetic Operations
        if state['step'] == 'arithmetic':
            st.subheader("Step 3: Arithmetic Operations")
            with st.form("arithmetic_form"):
                st.write("Add one or more arithmetic operations.")
                cont1 = st.selectbox("First variable:", numeric_cols, key="arith_cont1")
                cont2 = st.selectbox("Second variable:", numeric_cols, key="arith_cont2")
                arith_op = st.selectbox("Operation:", ["add", "subtract", "multiply", "divide", "lag-1", "lag-7", "lag-14", "lag-28", "lag-30"], key="arith_op")
                add_operation = st.form_submit_button("Add Operation")

            if add_operation and cont1 and cont2:
                st.session_state.arithmetic_operations.append({'cont1': cont1, 'cont2': cont2, 'operation': arith_op})
                st.info(f"Added {arith_op} operation on: {cont1} and {cont2}")
                st.rerun()
            
            if st.session_state.arithmetic_operations:
                st.write("**Pending Arithmetic Operations:**")
                for i, op in enumerate(st.session_state.arithmetic_operations):
                    st.write(f"{i+1}. {op['operation']} on: {op['cont1']} and {op['cont2']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Apply All Arithmetic Operations", type="primary"):
                    for op in st.session_state.arithmetic_operations:
                        df, state['arith_i'] = proceed_arith_fun(df, numeric_cols, state['arith_i'], op['cont1'], op['cont2'], op['operation'])
                    state['df'] = df.copy()
                    st.session_state.arithmetic_operations = [] # Clear the queue
                    state['step'] = 'binning'
                    st.success("✅ Step 3 Completed!")
                    st.rerun()
            with col2:
                if st.button("⏩ Skip Step 3"):
                    state['arith_i'].append("Skipped feature creation as per user's input.")
                    state['step'] = 'binning'
                    st.success("✅ Step 3 Skipped!")
                    st.rerun()

        # Step 4: Binning
        if state['step'] == 'binning':
            st.subheader("Step 4: Binning Features")
            with st.form("binning_form"):
                st.write("Add one or more binning operations.")
                cols_to_bin = st.multiselect("Select columns to bin:", numeric_cols, key="bin_cols")
                method = st.selectbox("Method:", ["Equal-width", "Equal-frequency", "Custom"], key="bin_method")
                n_bins = st.number_input("Number of bins:", min_value=2, value=5, key="n_bins")
                bin_labels_input = st.text_input("Enter labels (optional, comma-separated):", key="bin_labels")
                custom_bins = st.text_input("Enter custom bin edges (comma-separated):") if method == "Custom" else ""
                add_operation = st.form_submit_button("Add Operation")
            
            if add_operation and cols_to_bin:
                bin_labels = [label.strip() for label in bin_labels_input.split(',')] if bin_labels_input else False
                st.session_state.binning_operations.append({
                    'columns': cols_to_bin,
                    'method': method,
                    'n_bins': n_bins,
                    'bin_labels': bin_labels,
                    'custom_bins': custom_bins
                })
                st.info(f"Added {method} binning operation for: {', '.join(cols_to_bin)}")
                st.rerun()

            if st.session_state.binning_operations:
                st.write("**Pending Binning Operations:**")
                for i, op in enumerate(st.session_state.binning_operations):
                    st.write(f"{i+1}. {op['method']} binning on: {', '.join(op['columns'])}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Apply All Binning Operations", type="primary"):
                    for op in st.session_state.binning_operations:
                        df, state['binning_i'] = proceed_binning(df, numeric_cols, state['binning_i'], op['columns'], op['method'], op['n_bins'], op['bin_labels'], op['custom_bins'])
                    state['df'] = df.copy()
                    st.session_state.binning_operations = [] # Clear the queue
                    state['step'] = 'encoding'
                    st.success("✅ Step 4 Completed!")
                    st.rerun()
            with col2:
                if st.button("⏩ Skip Step 4"):
                    state['binning_i'].append("Skipped feature binning as per user's input.")
                    state['step'] = 'encoding'
                    st.success("✅ Step 4 Skipped!")
                    st.rerun()

        # Step 5: Encoding
        if state['step'] == 'encoding':
            st.subheader("Step 5: Encoding Features")
            with st.form("encoding_form"):
                st.write("Add one or more encoding operations.")
                onehot_cols = st.multiselect("Select columns for One-Hot Encoding:", cat_cols, key="onehot_cols")
                label_cols = st.multiselect("Select columns for Label Encoding:", cat_cols, key="label_cols")
                add_onehot = st.form_submit_button("Add One-Hot Encoding")
                add_label = st.form_submit_button("Add Label Encoding")
            
            if add_onehot and onehot_cols:
                st.session_state.onehot_operations.append({'columns': onehot_cols})
                st.info(f"Added One-Hot operation for: {', '.join(onehot_cols)}")
                st.rerun()
            if add_label and label_cols:
                st.session_state.label_operations.append({'columns': label_cols})
                st.info(f"Added Label Encoding operation for: {', '.join(label_cols)}")
                st.rerun()
            
            if st.session_state.onehot_operations or st.session_state.label_operations:
                st.write("**Pending Encoding Operations:**")
                for i, op in enumerate(st.session_state.onehot_operations):
                    st.write(f"{i+1}. One-Hot encode: {', '.join(op['columns'])}")
                for i, op in enumerate(st.session_state.label_operations):
                    st.write(f"{len(st.session_state.onehot_operations)+i+1}. Label encode: {', '.join(op['columns'])}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Apply All Encoding Operations", type="primary"):
                    onehot_ops = st.session_state.onehot_operations
                    label_ops = st.session_state.label_operations
                    df, state['onehot_i'], state['label_i'] = proceed_one_hot_encoding(df, cat_cols, state['onehot_i'], state['label_i'], onehot_ops, label_ops)
                    state['df'] = df.copy()
                    st.session_state.onehot_operations = []
                    st.session_state.label_operations = []
                    state['step'] = 'completed'
                    st.success("✅ Step 5 Completed!")
                    st.rerun()
            with col2:
                if st.button("⏩ Skip Step 5"):
                    state['onehot_i'].append("Skipped feature one-hot encoding as per user's input.")
                    state['label_i'].append("Skipped feature label encoding as per user's input.")
                    state['step'] = 'completed'
                    st.success("✅ Step 5 Skipped!")
                    st.rerun()
        
        # Step 6: Completion
        if state['step'] == 'completed':
            st.success("🎉 All Feature Engineering steps completed!")
            if st.button("Reset All"):
                del st.session_state.fe_state
                st.rerun()
                
        return (df, state['skew_i'], state['boxcox_i'], state['onehot_i'], 
                state['label_i'], state['arith_i'], state['time_i'], 
                state['binning_i'], state['scale_i'], flag)
    
    else:
        # Non-interactive mode
        df, skew_interpretation, boxcox_interpretation, one_hot_encoding_interpretation, ordinal_columns_i, arith_i, time_features_i, binning_i, scale_i, flag = proceed_no(df, numeric_cols, cat_cols, datetime_cols)

        data_type_df, _, _, _, _, _, _, _, df = detect_scales_of_measurement(df)
        
        df.to_csv(f"{save_dir}/feature_engineering.csv", index=False)
        
        return (df, skew_interpretation, boxcox_interpretation, one_hot_encoding_interpretation, 
                ordinal_columns_i, arith_i, time_features_i, binning_i, scale_i, flag)