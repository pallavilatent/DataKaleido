import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import sqlite3
import pymysql
import pyodbc
import re
from pathlib import Path
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')


# --- DATABASE UTILS ---
@st.cache_resource
def get_database_tables(engine):
    try:
        with engine.connect() as conn:
            if engine.dialect.name == 'sqlite':
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            else:
                result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
            return tables
    except Exception as e:
        st.write(f"Error fetching tables: {str(e)}")
        return []

@st.cache_resource
def load_data_from_database(engine, table_name, limit=None):
    try:
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.write(f"Error loading data: {str(e)}")
        return None

# Import all the utility functions
from excel_utils.Shape import (
    get_dataframe_shape,
    get_data_types,
    get_head,
    cleaning_data_frame
)
from excel_utils.id_columns import remove_id_columns
from excel_utils.plotter import plot_numeric_columns
from excel_utils.pdf_writer import generate_pdf_report
from excel_utils.scales_of_measurement import detect_scales_of_measurement
from excel_utils.outliers import detect_outliers_iqr
from excel_utils.data_summary_plot import summarize_and_plot
from excel_utils.missing_value_imputation_automatic import auto_missing_value_imputation
from excel_utils.feature_engineering import feature_engineering, suggest_boxcox_transformation
from excel_utils.frequency_tables import build_combined_frequency_dataframe
from excel_utils.modified_correlation_function import compute_auto_correlations_v2
from excel_utils.group_pivot_table import groupbyand_pivot



# Helper functions for Excel date parsing
@st.cache_resource
def is_probable_year_series(series):
    """Check if a series is likely to contain years"""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return series.between(1900, 2100).mean() > 0.8

st.cache_resource
def safe_excel_date_parser(date_series):
    """
    Robust parser for Excel date columns where some values are strings, 
    and others are Excel float serial dates.
    """
    parsed_dates = []
    
    for val in date_series:
        try:
            # If it's already a datetime, keep it
            if isinstance(val, pd.Timestamp):
                parsed_dates.append(val)
            # If it's a float/int (Excel serial date), convert using Excel's origin
            elif isinstance(val, (int, float)) and not pd.isna(val):
                parsed_dates.append(pd.to_datetime('1899-12-30') + pd.to_timedelta(val, unit='D'))
            # If it's a string, parse normally
            elif isinstance(val, str):
                parsed_dates.append(pd.to_datetime(val, errors='coerce'))
            else:
                parsed_dates.append(pd.NaT)
        except (ValueError, OverflowError):
            parsed_dates.append(pd.NaT)
    
    return pd.Series(parsed_dates)

@st.cache_resource
def enhanced_date_detection(series):
    """
    Enhanced date detection that handles mixed MM-DD-YYYY and MM-DD-YY formats
    with better pattern recognition for dates like '03-04-2023', '1-15-23', etc.
    """
    parsed_dates = []
    
    for val in series:
        try:
            # If it's already a datetime, keep it
            if isinstance(val, pd.Timestamp):
                parsed_dates.append(val)
            # If it's a float/int (Excel serial date), convert using Excel's origin
            elif isinstance(val, (int, float)) and not pd.isna(val):
                parsed_dates.append(pd.to_datetime('1899-12-30') + pd.to_timedelta(val, unit='D'))
            # If it's a string, try multiple parsing strategies
            elif isinstance(val, str) and val.strip():
                val_str = val.strip()
                
                # Try different date formats in order of specificity
                parsed = None
                
                # Format 1: MM-DD-YYYY (e.g., "03-04-2023")
                if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', val_str):
                    try:
                        parsed = pd.to_datetime(val_str, format='%m-%d-%Y', errors='coerce')
                    except:
                        pass
                
                # Format 2: MM-DD-YY (e.g., "1-15-23")
                if parsed is None and re.match(r'^\d{1,2}-\d{1,2}-\d{2}$', val_str):
                    try:
                        # Assume 20xx for years 00-29, 19xx for years 30-99
                        parsed = pd.to_datetime(val_str, format='%m-%d-%y', errors='coerce')
                    except:
                        pass
                
                # Format 3: MM/DD/YYYY (e.g., "03/04/2023")
                if parsed is None and re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', val_str):
                    try:
                        parsed = pd.to_datetime(val_str, format='%m/%d/%Y', errors='coerce')
                    except:
                        pass
                
                # Format 4: MM/DD/YY (e.g., "1/15/23")
                if parsed is None and re.match(r'^\d{1,2}/\d{1,2}/\d{2}$', val_str):
                    try:
                        parsed = pd.to_datetime(val_str, format='%m/%d/%y', errors='coerce')
                    except:
                        pass
                
                # Format 5: Try pandas automatic parsing as fallback
                if parsed is None:
                    parsed = pd.to_datetime(val_str, errors='coerce')
                
                parsed_dates.append(parsed)
            else:
                parsed_dates.append(pd.NaT)
        except (ValueError, OverflowError):
            parsed_dates.append(pd.NaT)
    
    return pd.Series(parsed_dates)




# Set page config
st.set_page_config(
    page_title="DataKaleido",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light/sky blue theme with black fonts and styled radio tabs
st.markdown("""
<style>
    .stApp { background-color: #F0F8FF; }
    .main-header { font-size: 3rem; color: #2563EB; text-align: center; margin-bottom: 1rem; font-weight: bold; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .section-header { font-size: 1.5rem; color: #1E3A8A; border-bottom: 2px solid #87CEEB; padding-bottom: 0.5rem; margin-top: 2rem; font-weight: 600; }
    
    /* Hide the default radio circle */
    div[role=radiogroup] > label > div:first-child {
        display: none !important;
    }

    /* Style radio labels like tabs */
    div[role=radiogroup] > label {
        border: 1px solid #87CEEB;
        padding: 0.75rem 1.5rem;
        margin-right: -1px; /* overlap borders */
        cursor: pointer;
        background-color: #F0F8FF;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E3A8A;
        transition: all 0.3s ease;
        min-width: 120px;
        text-align: center;
    }

    /* Hover effect */
    div[role=radiogroup] > label:hover {
        background-color: #E6F3FF;
        border-color: #5F9EA0;
    }

    /* Active tab */
    div[role=radiogroup] > label[data-baseweb="radio"] input:checked + div {
        background-color: #87CEEB;
        border-bottom: 3px solid #1E3A8A;
        font-weight: 700;
        color: #1E3A8A;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Radio container (tabs row) */
    div[role=radiogroup] {
        display: flex;
        border-bottom: 2px solid #87CEEB;
        margin-bottom: 0.5rem;
        background-color: #F0F8FF;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 0.5rem 0 0.5rem;
    }

    /* Regular tab styles (for nested tabs) */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #F0F8FF; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #FFFFFF; border-radius: 6px; color: #000000; font-weight: 700; font-size: 1.2rem; padding: 12px 20px; }
    .stTabs [aria-selected="true"] { background-color: #87CEEB; color: #000000; font-weight: 800; font-size: 1.3rem; }
    
    .stMarkdown, .stText, .stDataFrame, .stTable, .stMetric, .stInfo, .stSuccess, .stWarning, .stError, .stSubheader, .stHeader, .stTitle, .stCaption, .stExpanderContent, .stButton, .stSelectbox, .stTextInput, .stNumberInput, .stFileUploader, .stCheckbox, .stRadio, .stSlider, .stDownloadButton { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGO ---
st.markdown('<h1 class="main-header"> DataKaleido</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #1E3A8A; font-style: italic; margin-bottom: 2rem;">"Accelerate Your Data Discovery through the lens of clarity"</h3>', unsafe_allow_html=True)




# --- TABS ---
tab1, tab2 = st.tabs(["File Upload", "Database Connection"])
df = None
dataset_name = None

with tab1:
    st.header("📁 Upload Your Dataset")
    st.info("**Supported Formats:** CSV, Excel (.xlsx, .xls), JSON")
    st.info("💡 **Excel files:** Multi-sheet support with automatic date detection and conversion")
    
    # Add description before uploading dataset
    
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload your dataset for comprehensive EDA analysis"
    )
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                # Initialize variables for Excel files
                sheet_names = []
                sheet_name = ""
                converted_cols = []
                
                # For Excel files, first check if there are multiple sheets
                if file_extension in ['.xlsx', '.xls']:
                    try:
                        # Read Excel file to get sheet names
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        
                        if len(sheet_names) > 1:
                            # Multiple sheets - let user choose
                            st.info(f"📊 Excel file contains {len(sheet_names)} sheets: {', '.join(sheet_names)}")
                            selected_sheet = st.selectbox(
                                "Select sheet to load:",
                                options=sheet_names,
                                help="Multiple sheets detected. Choose which one to analyze."
                            )
                            sheet_name = selected_sheet
                        else:
                            # Single sheet - use the first one
                            sheet_name = sheet_names[0]
                            st.info(f"📋 Loading sheet: {sheet_name}")
                        
                        # Load Excel file directly with pandas for Streamlit compatibility
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        
                        # Apply the date conversion logic from reader.py manually
                        threshold = 0.5
                        
                        
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                try:
                                    # Enhanced date detection for mixed formats
                                    converted = enhanced_date_detection(df[col])
                                    if converted.notna().sum() > len(df) * threshold:
                                        df[col] = converted
                                        converted_cols.append(col)
                                        #st.info(f"✅ Detected and converted date column: {col}")
                                except Exception:
                                    pass
                            
                            if df[col].dtype == 'int64':
                                # Check if it's a year series
                                if is_probable_year_series(df[col]):
                                    df[col] = pd.to_datetime(df[col].astype(str), format='%Y', errors='coerce')
                                    df[col] = safe_excel_date_parser(df[col])
                                    converted_cols.append(col)
                        
                        
                    except Exception as excel_error:
                        st.error(f"Error reading Excel file: {str(excel_error)}")
                        st.info("💡 Make sure the Excel file is not corrupted and has valid data.")
                        st.stop()
                        
                else:
                    # For non-Excel files, use pandas directly
                    if file_extension == '.csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == '.json':
                        df = pd.read_json(uploaded_file)
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        st.stop()
                
                dataset_name = uploaded_file.name
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                threshold = 0.5
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            # Enhanced date detection for mixed formats
                            converted = enhanced_date_detection(df[col])
                            if converted.notna().sum() > len(df) * threshold:
                                df[col] = converted
                                converted_cols.append(col)
                                #st.info(f"✅ Detected and converted date column: {col}")
                        except Exception:
                            pass
                    
                    if df[col].dtype == 'int64':
                        # Check if it's a year series
                        if is_probable_year_series(df[col]):
                            df[col] = pd.to_datetime(df[col].astype(str), format='%Y', errors='coerce')
                            df[col] = safe_excel_date_parser(df[col])
                            converted_cols.append(col)
                # Show file information
                file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # Convert to MB
                st.info(f"📁 **File Info:** {file_extension.upper()} file ({file_size:.2f} MB)")
                
                # For Excel files, show additional information
                if file_extension in ['.xlsx', '.xls']:
                    st.info(f"📊 **Excel File Details:** {len(df)} rows × {len(df.columns)} columns from sheet '{sheet_name}'")
                    
                    # Show additional Excel-specific information
                    col1= st.columns(1)
                    with col1:
                        st.metric("📋 Sheet Name", sheet_name)
                    
                    
                
                # Print the deep dive message
                st.write(f"**Let's deep dive into the dataset: {uploaded_file.name}**")
                
                # The load_data function already handles date conversion, so we don't need to do it again here
        except Exception as e:
            st.write(f"Error loading file: {str(e)}")

with tab2:
    st.header("🗄️ Connect to Database")
    st.info("**Supported Databases:** MySQL, PostgreSQL, SQL Server, SQLite, AWS RDS, Azure SQL, GCP Cloud SQL, Databricks")
    db_type = st.selectbox(
        "Select Database Type",
        ["MySQL", "PostgreSQL", "SQL Server", "SQLite", "AWS RDS", "Azure SQL", "GCP Cloud SQL", "Databricks"]
    )
    if db_type in ["MySQL", "PostgreSQL", "SQL Server", "AWS RDS", "Azure SQL", "GCP Cloud SQL"]:
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host/Server", "localhost", key="mysql_host_input")
            port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432" if db_type == "PostgreSQL" else "1433", key="mysql_port_input")
        with col2:
            username = st.text_input("Username", key="mysql_username_input")
            password = st.text_input("Password", type="password", key="mysql_password_input")
        database = st.text_input("Database Name", key="mysql_database_input")
        if st.button("🔗 Connect to Database"):
            if host and username and password and database:
                connection_params = {
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'database': database
                }
                engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
                try:
                    tables = get_database_tables(engine)
                    if tables:
                        selected_table = st.selectbox("Select Table", tables)
                        limit = st.number_input("Row Limit (0 for all)", min_value=0, value=1000)
                        if st.button("📊 Load Data"):
                            df = load_data_from_database(engine, selected_table, limit if limit > 0 else None)
                            if df is not None:
                                dataset_name = f"{database}.{selected_table}"
                                st.success(f"✅ Loaded {len(df)} rows from {selected_table}")
                                st.write(f"**Let's deep dive into the dataset: {dataset_name}**")
                            else:
                                st.write("Failed to load data from table")
                    else:
                        st.write("No tables found in the database")
                except Exception as e:
                    st.write(f"Database connection failed: {str(e)}")
            else:
                st.write("Please fill in all connection parameters")
    elif db_type == "SQLite":
        sqlite_file = st.file_uploader("Upload SQLite Database", type=['db', 'sqlite'])
        if sqlite_file:
            try:
                with open("temp_db.sqlite", "wb") as f:
                    f.write(sqlite_file.getbuffer())
                engine = create_engine(f"sqlite:///temp_db.sqlite")
                tables = get_database_tables(engine)
                if tables:
                    selected_table = st.selectbox("Select Table", tables)
                    limit = st.number_input("Row Limit (0 for all)", min_value=0, value=1000)
                    if st.button("📊 Load Data"):
                        df = load_data_from_database(engine, selected_table, limit if limit > 0 else None)
                        if df is not None:
                            dataset_name = f"sqlite.{selected_table}"
                            st.success(f"✅ Loaded {len(df)} rows from {selected_table}")
                            st.write(f"**Let's deep dive into the dataset: {dataset_name}**")
            except Exception as e:
                st.write(f"Error loading SQLite database: {str(e)}")
    elif db_type == "Databricks":
        st.info("Databricks connection requires additional configuration. Please use the Databricks SQL Connector.")
        databricks_host = st.text_input("Databricks Host", key="databricks_host_input")
        databricks_token = st.text_input("Access Token", type="password", key="databricks_token_input")
        databricks_catalog = st.text_input("Catalog", key="databricks_catalog_input")
        databricks_schema = st.text_input("Schema", key="databricks_schema_input")
        if st.button("🔗 Connect to Databricks"):
            st.info("Databricks connection feature requires additional setup")

# --- MAIN ANALYSIS ---
if df is not None:
    try:
        st.session_state["original_df_backup"] = df.copy()
        
        # Add description after uploading dataset
        num_rows, num_columns = get_dataframe_shape(df)
        st.session_state["num_columns"] = num_columns
        df, num_duplicate_rows, percentage_duplicates = cleaning_data_frame(df)
        
        # --- Sample View of the Dataset ---
        st.markdown('<h2 class="section-header">Sample View of the Dataset</h2>', unsafe_allow_html=True)
        
        # Display basic dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("#Total Rows", len(df))
        with col2:
            st.metric("#Total Columns", len(df.columns))
        

        
        # Display first few rows of the dataset
        st.dataframe(df.head(10), use_container_width=True)
        
        # --- ID Columns ---
        st.markdown('<h3 class="section-header">Spot ID Columns</h3>', unsafe_allow_html=True)

        # Use session_state to only show detection once
        if "id_detection_done" not in st.session_state:
            _, detected_id_cols = remove_id_columns(df, skip_id_cols_confirmation=True)
            st.session_state["detected_id_cols"] = detected_id_cols
            st.session_state["id_detection_done"] = True

        # Get all columns except detected ID columns for the dropdown
        all_columns = list(df.columns)
        detected_id_cols = st.session_state.get("detected_id_cols", [])
        available_columns = [col for col in all_columns if col not in detected_id_cols]
        
        # Multi-select dropdown for additional ID columns
        user_additional_id_cols = st.multiselect(
            "Add any additional ID columns to remove (optional):",
            options=available_columns,
            default=[],  # No pre-selected values
            help="Select multiple columns that you want to remove as ID columns",
            key="additional_id_cols_multiselect"
        )
        
        # Show summary of what will be removed
        if detected_id_cols or user_additional_id_cols:
            st.info(f"**Columns to be removed:** {', '.join(detected_id_cols + user_additional_id_cols)}")
        else:
            st.info("No ID columns detected or selected for removal.")
        
        
        if st.button("Remove ID Columns"):
            # Combine detected ID columns with user-selected additional columns
            all_id_cols_to_remove = detected_id_cols + user_additional_id_cols
            
            if all_id_cols_to_remove:
                # Remove all ID columns at once
                df_cleaned = df.drop(columns=all_id_cols_to_remove)
                id_cols = all_id_cols_to_remove
                # Store the cleaned DataFrame in session state for use in analysis
                st.session_state["df_cleaned"] = df_cleaned
                st.success(f"Removed columns: {', '.join(id_cols)}")
                st.session_state["id_columns_removed"] = True
                st.session_state["removed_id_cols"] = id_cols
                st.info(f"✅ Removed ID columns: {', '.join(id_cols)}. All subsequent analysis will use the updated dataset without these columns.")
            else:
                st.warning("No ID columns detected or selected for removal.")
            # Optionally, update session_state to prevent further repeats
            df_cleaned.to_csv("output/EDA_Processed_Data.csv", index=False)
        #reload data frame
       
        
        # Optionally, update session_state to prevent further repeats
        st.session_state["id_detection_done"] = False

        # --- Main Analysis Tabs ---
        if st.session_state.get("id_columns_removed", False):
            st.markdown('<h2 class="section-header">Analysis Sections</h2>', unsafe_allow_html=True)
            
            # Get the cleaned DataFrame from session state
            df_cleaned = st.session_state.get("df_cleaned", df)
            
            # Ensure we have a valid DataFrame to work with
            if df_cleaned is None or df_cleaned.empty:
                st.error("❌ No valid DataFrame available for analysis. Please try removing ID columns again.")
                st.stop()
            
            # Initialize df_missed_cleaned with the cleaned DataFrame
            #df_missed_cleaned = df_cleaned.copy()
            
            # Store in session state for persistence across pages
            #st.session_state["df_missed_cleaned"] = df_missed_cleaned
            
            # Also ensure df_cleaned is stored in session state as fallback
            #st.session_state["df_cleaned"] = df_cleaned
            
            # Re-run scales detection on the updated DataFrame (without ID columns)
            scale_df, count_num, count_cat, count_text, count_datetime, count_percentage, count_ratio, count_geo, df_cleaned = detect_scales_of_measurement(df_cleaned)
            # Ensure scale_df only contains columns that exist in the current DataFrame
            if scale_df is not None and not scale_df.empty:
                scale_df = scale_df[scale_df['Column'].isin(df_cleaned.columns)].copy()
            
            # CRITICAL: Ensure df_missed_cleaned is still valid after scales detection
            if df_cleaned is None or not isinstance(df_cleaned, pd.DataFrame):
                st.error("❌ Critical: df_missed_cleaned became invalid after scales detection. Reinitializing...")
                df_cleaned = df_cleaned.copy()
                st.session_state["df_cleaned"] = df_cleaned
                st.success("✅ DataFrame restored after scales detection.")
            
            # Double-check that we have a working DataFrame
            if df_cleaned is None or df_cleaned.empty:
                st.error("❌ Fatal Error: Cannot proceed without a valid DataFrame.")
                st.stop()
            
                        # Store scale_df and count variables in session state for use across all sections
            st.session_state["scale_df"] = scale_df
            st.session_state["count_num"] = count_num
            st.session_state["count_cat"] = count_cat
            st.session_state["count_text"] = count_text
            st.session_state["count_datetime"] = count_datetime
            st.session_state["count_percentage"] = count_percentage
            st.session_state["count_ratio"] = count_ratio
            st.session_state["count_geo"] = count_geo
            # Initialize outlier variables in session state
            if "outlier_summary_df" not in st.session_state:
                st.session_state["outlier_summary_df"] = pd.DataFrame()
            if "outlier_interpretations" not in st.session_state:
                st.session_state["outlier_interpretations"] = []
            
            
            
            # Styled Radio Navigation - Looks like tabs but with better control
            # Create sticky navigation bar
            st.markdown("""
            <style>
            .sticky-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: white;
                border-bottom: 2px solid #e0e0e0;
                padding: 10px 20px;
                z-index: 1000;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sticky-nav .stRadio > div {
                display: flex;
                justify-content: center;
                gap: 10px;
            }
            .sticky-nav .stRadio > div > label {
                background: #f0f2f6;
                border: 1px solid #d0d7de;
                border-radius: 20px;
                padding: 8px 16px;
                margin: 0;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .sticky-nav .stRadio > div > label:hover {
                background: #e1e5e9;
                transform: translateY(-2px);
            }
            .sticky-nav .stRadio > div > label[data-checked="true"] {
                background: #1f77b4;
                color: white;
                border-color: #1f77b4;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Add top margin to prevent content from being hidden behind sticky nav
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            
            # Sticky navigation container
            with st.container():
                st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
                selected_section = st.radio(
                    "Choose Analysis Section",
                    [
                        "📊 Data Structure",
                        "✅ Data Validation",
                        "📈 Data Distribution",
                        "🔗 Data Relationships",
                        "⚙️ Feature Engineering",
                        "📥 Reports & Downloads"
                    ],
                    key="analysis_navigation",
                    horizontal=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            

            
            # Conditional rendering based on selected section
            if selected_section == "📊 Data Structure":
                st.markdown('<h3 class="section-header">Data Quality Analysis</h3>', unsafe_allow_html=True)
                
                # a. Initial Data Summary (Data Quality Analysis)
                df_original=st.session_state.get("original_df_backup", df)
                # Calculate data quality metrics
                num_rows = len(df_cleaned)
                num_columns = len(df_cleaned.columns)
                num_duplicate_rows = df_cleaned.duplicated().sum()
                percentage_duplicates = (num_duplicate_rows / num_rows) * 100 if num_rows > 0 else 0

                num_rows_org=len(df_original)
                num_columns_org=len(df_original.columns)
                total_missing_count = df_original.isnull().sum().sum()
                missing_percentage = (total_missing_count / (num_rows_org * num_columns_org)) * 100 if (num_rows_org * num_columns_org) > 0 else 0
                
                # Display metrics in columns
                col1, col2,col3,col4,col5,col6 = st.columns(6)
                with col1:
                    st.metric("#Rows", num_rows)
                with col2:
                    st.metric("#Columns", num_columns)
                with col3:
                    st.metric("#Duplicate", num_duplicate_rows)
                with col4:
                    st.metric("#Duplicate %", f"{percentage_duplicates:.0f}%")
                with col5:
                    st.metric("#Missing", total_missing_count)
                with col6:
                    st.metric("#Missing %", f"{missing_percentage:.0f}%")
                
                # b. Know your Datatypes (Scales of Measurement)
                scale_df_copy=st.session_state.get("scale_df", scale_df)
                st.session_state["scale_df_copy"] = scale_df_copy
                st.markdown('<h3 class="section-header">Know your Datatypes</h3>', unsafe_allow_html=True)
                
                try:
                    # Get scale_df from session state
                    scale_df = st.session_state.get("scale_df", scale_df)
                    
                    # Use the scales_df that was already computed above
                    if scale_df is not None and not scale_df.empty:
                        # Display scales information
                        st.dataframe(scale_df, use_container_width=True)
                        
                        # Feature Count Summary
                        
                        # Count different types of features based on scale_df
                        count_num_stream = st.session_state.get('count_num', 0)
                        count_cat_stream = st.session_state.get('count_cat', 0)
                        count_text_stream =    st.session_state.get('count_text', 0)
                        count_datetime_stream = st.session_state.get('count_datetime', 0)
                        count_percentage_stream = st.session_state.get('count_percentage', 0)
                        count_ratio_stream = st.session_state.get('count_ratio', 0)
                        count_geo_stream = st.session_state.get('count_geo', 0)
                        st.session_state['count_num_stream'] = count_num_stream
                        st.session_state['count_cat_stream'] = count_cat_stream
                        st.session_state['count_text_stream'] = count_text_stream
                        st.session_state['count_datetime_stream'] = count_datetime_stream
                        st.session_state['count_percentage_stream'] = count_percentage_stream
                        st.session_state['count_ratio_stream'] = count_ratio_stream
                        st.session_state['count_geo_stream'] = count_geo_stream
                        
                        # Display feature counts in a nice format
                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                        with col1:
                            st.metric("#Categorical Features", count_cat_stream)
                        with col2:
                            st.metric("#Numerical Features", count_num_stream)
                        with col3:
                            st.metric("#Text Features", count_text_stream)
                        with col4:
                            st.metric("#Date-Time Features", count_datetime_stream)
                        with col5:
                            st.metric("#Percentage Features", count_percentage_stream)
                        with col6:
                            st.metric("#Ratio Features", count_ratio_stream)
                        with col7:
                            st.metric("#Geographic Features", count_geo_stream)
                    else:
                        st.info("Could not determine scales of measurement for this dataset.")
                        
                except Exception as e:
                    st.error(f"Error analyzing data types: {str(e)}")
                    st.write("**Basic Data Types:**")
                    st.dataframe(df_cleaned.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}), use_container_width=True)
            
            elif selected_section == "✅ Data Validation":
                # Get df_missed_cleaned from session state or use df_cleaned as fallback
                df_cleaned = st.session_state.get("df_cleaned", df_cleaned)
                
                # Use helper function to ensure DataFrame validity
                
                st.markdown('<h3 class="section-header">Missing Value Report & Imputation</h3>', unsafe_allow_html=True)
                
                # Get scale_df from session state
                scale_df = st.session_state.get("scale_df", scale_df)
                
                # Final validation before function call
                
                
                # Update df_missed_cleaned with imputation results
                df_cleaned, imputation_df, missing_summary, interpretations_miss, scale_df,summary_message = auto_missing_value_imputation(df_cleaned, scale_df)

                st.session_state["scale_df"] = scale_df #update scale_df in session state
                # Store updated DataFrame and missing_summary in session state
                st.session_state["df_cleaned"] = df_cleaned
                st.session_state["missing_summary"] = missing_summary
                st.session_state["imputation_df"] = imputation_df
                st.session_state["interpretations_miss"] = interpretations_miss
                st.session_state["summary_message"] = summary_message
                
                
                # b. Summary Statistics
                st.markdown('<h3 class="section-header">Summary Statistics</h3>', unsafe_allow_html=True)
                
                st.write("**The below table displays the Measures of Central Tendency and Dispersion**")
                
                try:
                    # Get scale_df from session state
                    scale_df = st.session_state.get("scale_df", scale_df)
                    # Use the scale_df that was already computed by detect_scales_of_measurement
                    if scale_df is not None and not scale_df.empty:
                        metadata_df = scale_df.copy()
                    else:
                        # Fallback: create basic metadata DataFrame
                        metadata_df = pd.DataFrame({
                            'Column': df_cleaned.columns,
                            'Data type': df_cleaned.dtypes.values,
                            'Data type Category': ['Unknown'] * len(df_cleaned.columns),
                            'Scale of Measurement': ['Unknown'] * len(df_cleaned.columns),
                            'Unique Values': [df_cleaned[col].nunique() for col in df_cleaned.columns]
                        })
                    
                    # Use summarize_and_plot function with required parameter
                    summary_df_1, summary_df_2, summary_df_3, interpretations, plot_paths, plot_paths2, plot_paths3, dist_df, plot_paths4, summary_info,skew_info = summarize_and_plot(df_cleaned, metadata_df)

                    # Store plot variables and summary data in session state for use in other sections
                    st.session_state["plot_paths"] = plot_paths
                    st.session_state["plot_paths2"] = plot_paths2
                    st.session_state["plot_paths3"] = plot_paths3
                    st.session_state["plot_paths4"] = plot_paths4
                    st.session_state["dist_df"] = dist_df
                    st.session_state["interpretations"] = interpretations
                    st.session_state["summary_df_1"] = summary_df_1
                    st.session_state["summary_df_2"] = summary_df_2
                    st.session_state["summary_df_3"] = summary_df_3
                    st.session_state["summary_info"] = summary_info
                    st.session_state["skew_info"] = skew_info

                    # Display Summary Statistics in organized tabs
                    
                    # Create tabs for different summary statistics
                    stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs(["Basic Stats", "Central Tendency", "Distribution", "Categorical"])
                    
                    with stat_tab1:
                        st.markdown('<h5>Basic Statistics (Count, Min, Max)</h5>', unsafe_allow_html=True)
                        if summary_df_1 is not None and not summary_df_1.empty:
                            st.dataframe(summary_df_1, use_container_width=True)
                            st.info("**Basic Statistics:** Count of non-null values, minimum and maximum values for each numerical column.")
                        else:
                            st.info("No basic statistics available.")
                    
                    with stat_tab2:
                        st.markdown('<h5>Central Tendency & Dispersion</h5>', unsafe_allow_html=True)
                        if summary_df_2 is not None and not summary_df_2.empty:
                            st.dataframe(summary_df_2, use_container_width=True)
                            
                            # Add detailed information about measures
                            st.markdown("""
                            **Understanding the Measures:**
                            
                            * **Mean, Median and Mode** are measures of central tendency that summarises the data in a single value that represent the entire data.
                            
                            * **Geometric mean and harmonic mean** are used to summarize variables that represent the rate of change% and ratios respectively.
                            
                            * The measures of dispersion like **Variance and Standard deviation** describes the spread or variability of the variable, it indicates how much the individual data point is deviated from the central value (mean).
                            """)
                        else:
                            st.info("No central tendency statistics available.")
                    
                    with stat_tab3:
                        st.markdown('<h5>Distribution Statistics (Quantiles & Skewness)</h5>', unsafe_allow_html=True)
                        if summary_df_3 is not None and not summary_df_3.empty:
                            st.dataframe(summary_df_3, use_container_width=True)
                            st.info("**Distribution Stats:** Quartiles (25%, 50%, 75%) and skewness measure for each numerical column.")
                        else:
                            st.info("No distribution statistics available.")
                    
                    with stat_tab4:
                        st.markdown('<h5>Categorical Distribution</h5>', unsafe_allow_html=True)
                        if dist_df is not None and not dist_df.empty:
                            st.dataframe(dist_df, use_container_width=True)
                            st.info("**Categorical Stats:** Unique values count, first and last values for categorical columns.")
                        else:
                            st.info("No categorical distribution data available.")
                    
                    # Display Summary Information
                except Exception as e:
                        st.error(f"Error generating summary statistics: {str(e)}")
                    
                    # Additional statistics summary
                    
                    
                    # Display Generated Plots
                    
            
            elif selected_section == "📈 Data Distribution":
                # Get df_missed_cleaned from session state or use df_cleaned as fallback
                df_cleaned = st.session_state.get("df_cleaned")
                
                # Use helper function to ensure DataFrame validity
            
                
                # Generate plots if not already generated
                if "plot_paths" not in st.session_state or "plot_paths2" not in st.session_state:
                    with st.spinner("Generating plots..."):
                        try:
                            # Get scale_df from session state
                            scale_df = st.session_state.get("scale_df", scale_df)
                            # Use the scale_df that was already computed
                            if scale_df is not None and not scale_df.empty:
                                metadata_df = scale_df.copy()
                            else:
                                # Fallback: create basic metadata DataFrame
                                metadata_df = pd.DataFrame({
                                    'Column': df_cleaned.columns,
                                    'Data type': df_cleaned.dtypes.values,
                                    'Data type Category': ['Unknown'] * len(df_cleaned.columns),
                                    'Scale of Measurement': ['Unknown'] * len(df_cleaned.columns),
                                    'Unique Values': [df_cleaned[col].nunique() for col in df_cleaned.columns]
                                })
                            
                            # Generate plots using summarize_and_plot function
                            summary_df_1, summary_df_2, summary_df_3, interpretations, plot_paths, plot_paths2, plot_paths3, dist_df, plot_paths4, summary_info = summarize_and_plot(df_cleaned, metadata_df)
                            
                            # Store in session state for persistence
                            st.session_state["plot_paths"] = plot_paths
                            st.session_state["plot_paths2"] = plot_paths2
                            st.session_state["plot_paths3"] = plot_paths3
                            st.session_state["plot_paths4"] = plot_paths4
                            st.session_state["dist_df"] = dist_df
                            st.session_state["interpretations"] = interpretations
                            st.session_state["summary_df_1"] = summary_df_1
                            st.session_state["summary_df_2"] = summary_df_2
                            st.session_state["summary_df_3"] = summary_df_3
                            st.session_state["summary_info"] = summary_info
                        except Exception as e:
                            st.error(f"Error generating plots: {str(e)}")
                            # Initialize empty plot variables
                            st.session_state["plot_paths"] = []
                            st.session_state["plot_paths2"] = []
                            st.session_state["plot_paths3"] = []
                            st.session_state["plot_paths4"] = []
                            st.session_state["dist_df"] = pd.DataFrame()
                            st.session_state["interpretations"] = []
                
                # Get plot variables from session state
                plot_paths = st.session_state.get("plot_paths", [])
                plot_paths2 = st.session_state.get("plot_paths2", [])
                plot_paths3 = st.session_state.get("plot_paths3", [])
                plot_paths4 = st.session_state.get("plot_paths4", [])
                dist_df = st.session_state.get("dist_df", pd.DataFrame())
                interpretations = st.session_state.get("interpretations", [])
                
                # Create tabs for different types of plots
                plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs(["Histograms", "Box Plots", "Frequency Distribution", "Group Tables"])
                
                with plot_tab1:
                    st.markdown('<h5>Distribution Histograms</h5>', unsafe_allow_html=True)
                    if plot_paths:
                        dist_cols = [col for col, _ in plot_paths]
                        selected_dist_col = st.selectbox("Select column for histogram:", dist_cols, key="hist_select_1")
                        if selected_dist_col:
                            dist_plot_path = dict(plot_paths).get(selected_dist_col)
                            if dist_plot_path and os.path.exists(dist_plot_path):
                                st.image(dist_plot_path, caption=f'Histogram of {selected_dist_col}', use_container_width=False)
                                
                                # Download button for histogram
                                with open(dist_plot_path, 'rb') as f:
                                    st.download_button(
                                        label=f"📥 Download {selected_dist_col} Histogram",
                                        data=f.read(),
                                        file_name=f"histogram_{selected_dist_col}.png",
                                        mime="image/png",
                                        key=f"hist_download_{selected_dist_col}"
                                    )
                            else:
                                st.warning(f"Plot file not found for {selected_dist_col}")
                    else:
                        st.info("No histogram plots available.")
                
                with plot_tab2:
                    st.markdown('<h5>Box Plots</h5>', unsafe_allow_html=True)
                    if plot_paths2:
                        st.markdown("""
                **Box plot summarizes the distribution of a variable by using its median, quartiles, minimum and maximum values. It is a powerful chart to identify the spread and the presence of outliers in the variable.**
                
                • The Box represents the Inter quartile range. The Q1 is the lower portion of the box and Q3 represents the upper portion. The line inside the box indicates the Median (or Q2).
                
                • If the Median line is closer to the lower end of the box, then the distribution is positively skewed (or right skewed). If the Median line is closer to the upper end of the box, then the distribution is negatively skewed (or left skewed).
                
                • The line at the lower end and upper end of the box represents (called as whiskers) represents the minimum and the maximum value of the distribution.
                
                • The dots below or above the whiskers are identified as Outliers. They are identified based on IQR, If a value is higher than the 1.5IQR above the upper quartile (Q3) or lower than 1.5IQR below the lower quartile (Q1), the value will be considered as outlier.
                """)
                        box_cols = [col for col, _ in plot_paths2]
                        selected_box_col = st.selectbox("Select column for box plot:", box_cols, key="box_select_1")
                        if selected_box_col:
                            box_plot_path = dict(plot_paths2).get(selected_box_col)
                            if box_plot_path and os.path.exists(box_plot_path):
                                st.image(box_plot_path, caption=f'Box Plot of {selected_box_col}', use_container_width=False, width=400)
                                
                                # Download button for box plot
                                with open(box_plot_path, 'rb') as f:
                                    st.download_button(
                                        label=f"📥 Download {selected_box_col} Box Plot",
                                        data=f.read(),
                                        file_name=f"boxplot_{selected_box_col}.png",
                                        mime="image/png",
                                        key=f"box_download_{selected_box_col}"
                                    )
                            else:
                                st.warning(f"Plot file not found for {selected_box_col}")
                    else:
                        st.info("No box plots available.")
                
                # c. Outlier Detection
                    
                    try:
                        # Filter out ID columns that were removed
                        df_cleaned = st.session_state.get("df_cleaned")
                        numeric_cols_outlier = df_cleaned.select_dtypes(include=[np.number]).columns
                        #if 'id_columns_removed' in st.session_state and st.session_state.get('removed_id_cols'):
                        #    removed_cols = st.session_state.get('removed_id_cols', [])
                        #    numeric_cols_outlier = [col for col in numeric_cols_outlier if col not in removed_cols]
                        
                        if len(numeric_cols_outlier) > 0:
                            # Get scale_df from session state
                            scale_df = st.session_state.get("scale_df", scale_df)
                            
                            # Detect outliers using IQR method for the entire dataset
                            outlier_summary_df, outlier_interpretations, df_cleaned, scale_df = detect_outliers_iqr(df_cleaned, scale_df)
                            
                            # Store outlier results in session state
                            st.session_state["outlier_summary_df"] = outlier_summary_df
                            st.session_state["outlier_interpretations"] = outlier_interpretations
                            
                            
                                
                                # Display outlier interpretations if available
                            if outlier_interpretations:
                                st.write("**Outlier Summary:**")
                                for interpretation in outlier_interpretations:
                                    st.write(interpretation)
                            else:
                                #st.info("No outliers detected or error in outlier detection.")
                                pass
                        else:
                            st.info("No numeric columns found for outlier analysis.")
                    except Exception as e:
                        st.error(f"Error in outlier detection: {str(e)}")
                        # Initialize empty outlier variables in case of error
                        st.session_state["outlier_summary_df"] = pd.DataFrame()
                        st.session_state["outlier_interpretations"] = []
                
                    st.session_state["df_cleaned"] = df_cleaned
                    st.session_state["scale_df"] = scale_df
                
                
                # Display Interpretations
                if interpretations:
                    st.markdown('<h4>💡 Data Interpretations</h4>', unsafe_allow_html=True)
                    for interpretation in interpretations:
                        st.write(f"• {interpretation}")
                
                with plot_tab3:
                    st.markdown('<h2 class="section-header">Frequency Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Building frequency tables..."):
                        count_result,freq_table_summary = build_combined_frequency_dataframe(df_cleaned, scale_df)
                        
                        # Store frequency results in session state
                        st.session_state["freq_table_summary"] = freq_table_summary
                        
                        # Unpack the tuple returned by build_combined_frequency_dataframe
                        if isinstance(count_result, tuple):
                            count, freq_summary = count_result
                        else:
                            count = count_result
                            freq_summary = []
                    
                        # Get scale_df from session state
                        scale_df = st.session_state.get("scale_df", scale_df)
                        # Define categorical columns for this section
                        categorical_cols = []
                        if scale_df is not None and not scale_df.empty:
                            categorical_cols = scale_df[scale_df['Data type Category'].str.contains('Categorical')]['Column'].tolist()
                            categorical_cols = [col for col in categorical_cols if col in df_cleaned.columns]
                        
                        if categorical_cols:
                            selected_cat_col = st.selectbox("Select categorical column for frequency analysis:", categorical_cols)
                            if not count.empty and 'Column Name' in count.columns:
                                freq_data = count[count['Column Name'] == selected_cat_col]
                                if not freq_data.empty:
                                    st.dataframe(freq_data, use_container_width=True)
                                    
                                    # Display frequency summary if available
                                    if freq_summary:
                                        print("**Frequency Summary:**")
                                        for summary in freq_summary:
                                            if selected_cat_col in summary:
                                                print(summary)
                                else:
                                    st.write(f"No frequency data found for column '{selected_cat_col}'.")
                            else:
                                st.write("No frequency data available. Please check if categorical columns exist in the dataset.")
                    
                        #if not count.empty and 'Value' in count.columns:
                        #    st.dataframe(count, use_container_width=True)
                        #else:
                        #    st.warning("Frequency table does not have a 'Value' column or is empty.")
                
                
                
                with plot_tab4:
                    st.markdown('<h4>Group Tables</h4>', unsafe_allow_html=True)
                    st.write("**It's a table generated by grouping data based on one or more columns (categories) and applying aggregations**")
                    group_df, grouped_df_i, index_str, col_str, val_str, num_col, groupby_cols_str, agg_funcs_str = groupbyand_pivot(df_cleaned, scale_df, save_dir='eda_outputs')
                    st.session_state["grouped_df_i"] = grouped_df_i
                   
            
            elif selected_section == "🔗 Data Relationships":
                # Get df_missed_cleaned from session state or use df_cleaned as fallback
                df_cleaned = st.session_state.get("df_cleaned")
                
                # Use helper function to ensure DataFrame validity
                #df_missed_cleaned = ensure_valid_dataframe(df_missed_cleaned, df_cleaned, "Data Relationships")
                
                st.markdown('<h3 class="section-header">Data Relationships</h3>', unsafe_allow_html=True)
                
                # a. Correlation Analysis
                st.markdown('<h4>Correlation Analysis</h4>', unsafe_allow_html=True)
                st.markdown("""
                **Correlation is a statistical technique to measure the association between 2 variables. Its gives the strength and direction of association between variables. Here, we have computed the correlation using 5 different methods based on the datatypes.**
                
                1. **Pearson correlation** (for Continuous vs Continuous) variables. Value ranges from -1 to 1
                2. **Spearman's correlation** (for Ordinal vs Ordinal) variables. Value ranges from -1 to 1
                3. **Point bi-serial correlation** (for Binary vs Continuous) variables. Value ranges from -1 to 1
                4. **Cramer's v correlation** (for Categorical vs Categorical) variables. value ranges from 0 to 1
                5. **Correlation ratio** (for Categorical vs Continuous, it measures the proportion of variance in the continuous variable explained by the categorical) variables. Value ranges from 0 to 1.
                """)
                
                
                 # Get scale_df from session state
                scale_df = st.session_state.get("scale_df", scale_df)
                 
                 # Initialize correlation session state if not exists
                if 'correlation_threshold_state' not in st.session_state:
                     st.session_state.correlation_threshold_state = {
                         'current_step': 'threshold_choice',
                         'threshold_set': False,
                         'user_threshold': 0.7,
                         'columns_to_drop': [],
                         'processed_correlations': False
                     }
                 
                 # Ensure all required keys exist (in case session state was partially reset)
                required_keys = ['current_step', 'threshold_set', 'user_threshold', 'columns_to_drop', 'processed_correlations']
                for key in required_keys:
                     if key not in st.session_state.correlation_threshold_state:
                         st.session_state.correlation_threshold_state[key] = {
                             'current_step': 'threshold_choice',
                             'threshold_set': False,
                             'user_threshold': 0.7,
                             'columns_to_drop': [],
                             'processed_correlations': False
                         }[key]
                 
                 # Use the compute_auto_correlations_v2 function
                tidy_df, threshold_df, symmetric_matrix, df_cleaned, interpretation_messages, metadata_df = compute_auto_correlations_v2(df_cleaned, scale_df, 'output')
                st.session_state["df_cleaned"] = df_cleaned
                st.session_state["scale_df"] = scale_df
                # Store correlation results in session state
                st.session_state["tidy_df"] = tidy_df
                st.session_state["threshold_df"] = threshold_df
                st.session_state["symmetric_matrix"] = symmetric_matrix
                st.session_state["interpretation_messages"] = interpretation_messages

            elif selected_section == "⚙️ Feature Engineering":
                # Get df_missed_cleaned from session state or use df_cleaned as fallback
                df_cleaned = st.session_state.get("df_cleaned")
                
                # Use helper function to ensure DataFrame validity
                #df_missed_cleaned = ensure_valid_dataframe(df_missed_cleaned, df_cleaned, "Feature Engineering")
                
                st.markdown('<h3 class="section-header">Feature Engineering</h3>', unsafe_allow_html=True)
                
                # Feature Engineering Steps
                # Initialize feature engineering variables if not already set
               
            
                
                
                
                    # Get scale_df from session state
                scale_df = st.session_state.get("scale_df", scale_df)
                df_cleaned, skew_interpretation, boxcox_interpretation, one_hot_encoding_interpretation, ordinal_columns_i, arith_i, time_features_i, binning_i, scale_i, flag = feature_engineering(df_cleaned, scale_df)
                #df_feature.to_csv(f"eda_outputs/feature_engineering.csv", index=False)
                # Store feature engineering results in session state
                st.session_state["df_cleaned"] = df_cleaned
                #st.session_state["proceed_input"] = proceed_input
                st.session_state["skew"] = skew_interpretation
                st.session_state["box"] = boxcox_interpretation
                st.session_state["onehot"] = one_hot_encoding_interpretation
                st.session_state["ordinal"] = ordinal_columns_i
                st.session_state["arith_i"] = arith_i
                st.session_state["time_features_i"] = time_features_i
                st.session_state["binning"] = binning_i
                st.session_state["scale_i"] = scale_i
                st.session_state["flag"] = flag
                
            
            elif selected_section == "📥 Reports & Downloads":
                st.markdown('<h3 class="section-header">Reports & Downloads</h3>', unsafe_allow_html=True)
                
                # PDF Report Generation
                st.markdown('<h4>📄 PDF Report Generation</h4>', unsafe_allow_html=True)
                st.write("Generate a comprehensive PDF report containing all the analysis results and visualizations.")
                
                if st.button("🔄 Generate PDF Report", type="primary", use_container_width=True):
                    # Check if we have any data to work with
                    if df is None:
                        st.error("❌ No dataset loaded. Please upload a file or connect to a database first.")
                        st.stop()
                    
                    with st.spinner("Generating comprehensive PDF report..."):
                        try:
                            # Get the cleaned DataFrame for calculations with fallback to original df
                            df_cleaned = st.session_state.get("df_cleaned", None)
                            if df_cleaned is None or df_cleaned.empty:
                                df_cleaned = df  # Fallback to original DataFrame
                            
                            # Ensure we have a valid DataFrame
                            if df_cleaned is None or df_cleaned.empty:
                                st.error("❌ No valid DataFrame available for PDF generation. Please ensure data is loaded and ID columns are removed.")
                                st.stop()
                            
                            original_df_backup = st.session_state.get("original_df_backup", None)
                            # Calculate basic metrics safely
                            num_rows = len(original_df_backup)
                            num_columns = len(original_df_backup.columns)
                            num_duplicate_rows = original_df_backup.duplicated().sum()
                            percentage_duplicates = (num_duplicate_rows / num_rows * 100) if num_rows > 0 else 0
                            
                            # Calculate missing value metrics
                            missing_count_total = original_df_backup.isnull().sum().sum()
                            missing_percentage_total = (missing_count_total / (num_rows * num_columns) * 100) if (num_rows * num_columns) > 0 else 0
                            
                            # Get variables from session state with fallback values
                            missing_summary = st.session_state.get("missing_summary", [])
                            if missing_summary is None:
                                missing_summary = []
                            
                            imputation_df = st.session_state.get("imputation_df", pd.DataFrame())
                            if imputation_df is None:
                                imputation_df = pd.DataFrame()
                            
                            interpretations_miss = st.session_state.get("interpretations_miss", [])
                            if interpretations_miss is None:
                                interpretations_miss = []
                            
                            plot_paths = st.session_state.get("plot_paths", [])
                            if plot_paths is None:
                                plot_paths = []
                            
                            plot_paths2 = st.session_state.get("plot_paths2", [])
                            if plot_paths2 is None:
                                plot_paths2 = []
                            
                            plot_paths3 = st.session_state.get("plot_paths3", [])
                            if plot_paths3 is None:
                                plot_paths3 = []
                            
                            plot_paths4 = st.session_state.get("plot_paths4", [])
                            if plot_paths4 is None:
                                plot_paths4 = []
                            
                            dist_df = st.session_state.get("dist_df", pd.DataFrame())
                            if dist_df is None:
                                dist_df = pd.DataFrame()
                            
                            interpretations = st.session_state.get("interpretations", [])
                            if interpretations is None:
                                interpretations = []
                            
                            # Initialize correlation variables if not already set
                            if "interpretation_messages" not in st.session_state:
                                st.session_state["interpretation_messages"] = []
                            if "tidy_df" not in st.session_state:
                                st.session_state["tidy_df"] = pd.DataFrame()
                            if "freq_table_summary" not in st.session_state:
                                st.session_state["freq_table_summary"] = []
                            
                            # Ensure correlation table has the expected structure
                            if st.session_state.get("tidy_df") is not None and not st.session_state.get("tidy_df").empty:
                                # Check if the 'Correlation' column exists, if not create it
                                if 'Correlation' not in st.session_state["tidy_df"].columns:
                                    # Try to find a similar column or create a default one
                                    correlation_cols = [col for col in st.session_state["tidy_df"].columns if 'correlation' in col.lower() or 'value' in col.lower()]
                                    if correlation_cols:
                                        # Rename the first matching column to 'Correlation'
                                        st.session_state["tidy_df"] = st.session_state["tidy_df"].rename(columns={correlation_cols[0]: 'Correlation'})
                                    else:
                                        # Create a default 'Correlation' column with NaN values
                                        st.session_state["tidy_df"]['Correlation'] = np.nan
                            else:
                                # If no correlation data exists, create a minimal default table
                                st.session_state["tidy_df"] = pd.DataFrame({
                                    'Column_1': ['No data'],
                                    'Column_2': ['No data'],
                                    'Method': ['No data'],
                                    'Correlation': [0.0],
                                    'Interpretations': ['No correlation data available']
                                })
                            
                            

                            # Create a comprehensive report data dictionary with proper DataFrame objects
                            # Ensure all values are either strings or DataFrames, avoiding boolean ambiguity
                            report_data = {
                                    'intro_text': f"Let's deep dive into the dataset: <b>{dataset_name or 'Dataset'}</b>",
                                    'data_quality_df': pd.DataFrame([
                                        ["Number of Rows", int(num_rows)],
                                        ["Number of Columns", int(num_columns)],
                                        ["Number of Duplicates", int(num_duplicate_rows)],
                                        ["Duplicates %", f"{int(percentage_duplicates)}%"],
                                        ["Total missing count", int(missing_count_total)],
                                        ["Missing %", f"{int(round(missing_percentage_total))}%"]
                                    ], columns=["Dataset info", "Value"]),
                                    'Removed_id_columns': f"The id columns are removed :<b>{', '.join(st.session_state.get('removed_id_cols', []) if st.session_state.get('removed_id_cols') is not None else [])}</b>",
                                    'summary_scale_df': "The below table summarizes the data type and scales of measurement for variables in the dataset.",
                                    'summary_scale_df_1': f"The dataset contains <b>{round(int(num_rows),0)}</b> observations. ",
                                    'scale_df': st.session_state.get("scale_df", pd.DataFrame()) if st.session_state.get("scale_df") is not None else pd.DataFrame(),
                                    'no_cat': f"Number of categorical features : {st.session_state.get('count_cat_stream', 0) if st.session_state.get('count_cat') is not None else 0}",
                                    'no_num':f"Number of numerical features : {st.session_state.get('count_num_stream', 0) if st.session_state.get('count_num_stream') is not None else 0}",
                                    'no_text':f"Number of text features : {st.session_state.get('count_text_stream', 0) if st.session_state.get('count_text_stream') is not None else 0}",
                                    'no_datetime': f"Number of date-time features : {st.session_state.get('count_datetime_stream', 0) if st.session_state.get('count_datetime_stream') is not None else 0}",
                                    'no_percentage': f"Number of Percentage features : {st.session_state.get('count_percentage_stream', 0) if st.session_state.get('count_percentage_stream') is not None else 0}",
                                    'no_ratio': f"Number of Ratio type features : {st.session_state.get('count_ratio_stream', 0) if st.session_state.get('count_ratio_stream') is not None else 0}",
                                    'count_geo': f"Number of Geographical features : {st.session_state.get('count_geo_stream', 0) if st.session_state.get('count_geo_stream') is not None else 0}",  # Default value since geographical detection not implemented
                                    'interpretations': st.session_state.get("interpretations", []) if st.session_state.get("interpretations") is not None else [],
                                    'summary_message': st.session_state.get("summary_message", "Missing values were handled using appropriate imputation strategies based on data types and missingness patterns."),  # Get from session state with fallback
                                    'missing_df': st.session_state.get("imputation_df", pd.DataFrame()) if st.session_state.get("imputation_df") is not None else pd.DataFrame(),
                                    'summary_df_1': st.session_state.get("summary_df_1", pd.DataFrame()) if st.session_state.get("summary_df_1") is not None else pd.DataFrame(),
                                    'summary_df_2': st.session_state.get("summary_df_2", pd.DataFrame()) if st.session_state.get("summary_df_2") is not None else pd.DataFrame(),
                                    'summary_df_3': st.session_state.get("summary_df_3", pd.DataFrame()) if st.session_state.get("summary_df_3") is not None else pd.DataFrame(),
                                    'summary_info': st.session_state.get("summary_info", []) if st.session_state.get("summary_info") is not None else [],
                                    'plot_paths': st.session_state.get("plot_paths", []) if st.session_state.get("plot_paths") is not None else [],
                                    'corr_interprtations': st.session_state.get("interpretation_messages", []) if st.session_state.get("interpretation_messages") is not None and isinstance(st.session_state.get("interpretation_messages"), list) else [],
                                    'Binning' : " 6. Binning Feature Extraction on user choice",
                                    'group_pivot_text' : "A GroupBy table is a summary table that organizes data into groups based on one or more categories and calculates aggregate values (like sums, averages, or counts) for each group.",
                                    #'feature_proceed': st.session_state.get("proceed_input", "") if st.session_state.get("proceed_input") is not None else "",
                                    'end_feature' : "Feature Engineering is completed with new features added to the csv while preserving original columns",
                                    'Correlation_text_1': "Correlation analysis is a statistical method used to measure and describe the strength and direction of the relationship between two variables.",
                                    'Corr_text_2': "Here, we have computed the Correlation using 5 different methods based on the variable datatypes.",
                                    'Corr_text_3':"1. <b>Pearson Correlation</b> (Continuous vs Continuous variables); Ranges from -1 to 1",
                                    'Corr_text_4':"2. <b>Spearman Correlation</b> (Ordinal vs Ordinal) variables; Ranges from -1 to 1",
                                    'Corr_text_5':"3. <b>Point Bi-serial Correlation</b> (Continuous vs Binary) variables; Ranges from -1 to 1",
                                    'Corr_text_6':"4. <b>Cramer's V method</b> (Categorical vs Categorical) variables; Ranges from 0 to 1",
                                    'Corr_text_7':"5. <b>Correlation ratio</b> (for Categorical vs Continuous, it measures the proportion of variance in the continuous variable explained by the categorical) variables; Ranges from 0 to 1.",
                                    'outlier_summary' : st.session_state.get("outlier_summary_df", pd.DataFrame()) if st.session_state.get("outlier_summary_df") is not None else pd.DataFrame(),
                                    'outlier_interpretations': st.session_state.get("outlier_interpretations", []) if st.session_state.get("outlier_interpretations") is not None else [],
                                    'skew': st.session_state.get("skew", "") if st.session_state.get("skew") is not None else "",
                                    'box': st.session_state.get("box", "") if st.session_state.get("box") is not None else "",
                                    'onehot': st.session_state.get("onehot", "") if st.session_state.get("onehot") is not None else "",
                                    'ordinal': st.session_state.get("ordinal", "") if st.session_state.get("ordinal") is not None else "",
                                    'arith_i': st.session_state.get("arith_i", "") if st.session_state.get("arith_i") is not None else "",
                                    'time_features': st.session_state.get("time_features_i", "") if st.session_state.get("time_features_i") is not None else "",
                                    'binning': st.session_state.get("binning", "") if st.session_state.get("binning") is not None else "",
                                    'missing_summary': missing_summary if missing_summary is not None else [],
                                    'correlation_table': st.session_state.get("threshold_df", pd.DataFrame()) if st.session_state.get("threshold_df") is not None and isinstance(st.session_state.get("threshold_df"), pd.DataFrame) and not st.session_state.get("threshold_df").empty else pd.DataFrame(),
                                    'threshold': 0.7,  # Default correlation threshold
                                    'freq_summary': st.session_state.get("freq_table_summary", []) if st.session_state.get("freq_table_summary") is not None and isinstance(st.session_state.get("freq_table_summary"), list) else [], 
                                    'scale_i': st.session_state.get("scale_i", "") if st.session_state.get("scale_i") is not None else "",
                                    'grouped_df_i': st.session_state.get("grouped_df_i", "") if st.session_state.get("grouped_df_i") is not None else "",
                                    'interpretations_miss': interpretations_miss if interpretations_miss is not None else [],
                                    'skew_info': st.session_state.get("skew_info", []) if st.session_state.get("skew_info") is not None else [],
                                    'missing_summ':"Below are the details of the columns with missing values. The other columns in the dataset have no missing values.\n Columns with more than 60% missing values are dropped. Imputation methods are suggested based on the type and the under-lying distribution of the variables.",
                                    'scale_df_copy':st.session_state.get("scale_df_copy", pd.DataFrame()) if st.session_state.get("scale_df_copy") is not None else pd.DataFrame(),
                                    'flag':st.session_state.get("flag", 1) if st.session_state.get("flag") is not None else 1
                                    
                                    }
                            
                            # Debug: Print report data keys to help identify issues
                            
                            # Validate report data before PDF generation
                            for key, value in report_data.items():
                                if value is None:
                                    st.warning(f"Warning: {key} is None, setting to default value")
                                    if isinstance(value, list):
                                        report_data[key] = []
                                    elif isinstance(value, pd.DataFrame):
                                        report_data[key] = pd.DataFrame()
                                    else:
                                        report_data[key] = ""
                            
                            # Add specific debugging for correlation-related fields
                            
                            
                            try:
                                pdf_result = generate_pdf_report("eda_report.pdf", report_data)
                                st.write("PDF generation completed successfully")
                            except Exception as pdf_error:
                                st.error(f"PDF generation failed with error: {str(pdf_error)}")
                                st.error(f"Error type: {type(pdf_error)}")
                                st.error(f"Error details: {pdf_error}")
                                # Try to identify which field is causing the issue
                                for key, value in report_data.items():
                                    try:
                                        # Test if the value can be converted to string
                                        str(value)
                                    except Exception as field_error:
                                        st.error(f"Field '{key}' with value '{value}' caused error: {field_error}")
                                raise pdf_error
                            
                            # Check if PDF generation was successful
                            if pdf_result is not None and pdf_result:
                                st.success("✅ PDF report generated successfully!")
                                
                                # Create download button for the PDF
                                try:
                                    # Read the generated PDF file
                                    pdf_path = "eda_report.pdf"  # Default path
                                    if os.path.exists(pdf_path):
                                        with open(pdf_path, "rb") as pdf_file:
                                            pdf_bytes = pdf_file.read()
                                        
                                        st.download_button(
                                            label="📥 Download PDF Report",
                                            data=pdf_bytes,
                                            file_name=f"EDA_Report_{dataset_name or 'Dataset'}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("⚠️ PDF file not found. Please check the output directory.")
                                except Exception as e:
                                    st.error(f"❌ Error reading PDF file: {str(e)}")
                            else:
                                a=1 #added to prevent warning
                        except Exception as e:
                            st.error(f"❌ Error during PDF generation: {str(e)}")
                            st.info("💡 Make sure the pdf_writer module is properly configured.")
                
                # CSV Reports Access
                st.markdown('<h4>📊 CSV Reports Access</h4>', unsafe_allow_html=True)
                st.write("Access all generated CSV files and analysis outputs.")
                df_final = pd.concat([df_cleaned, st.session_state.get("original_df_backup", pd.DataFrame())], axis=1).loc[:, lambda df: ~df.columns.duplicated()]
                #df_missed_cleaned.to_csv(f"eda_outputs/EDA_Processed_Data.csv", index=False)
                #Final_data_after_merging for id cols and columns that were dropped. 
                df_final.to_csv(f"output/Final_Data_with_EDA_modifications.csv", index=False) 
                # Check for output directory
                output_dir = "output"
                if os.path.exists(output_dir):
                    #st.success(f"✅ Output directory found: {output_dir}")
                    
                    # List all CSV files in output directory
                    csv_files = []
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith('.csv'):
                                csv_files.append(os.path.join(root, file))
                    
                    if csv_files:
                        st.write(f"**Found {len(csv_files)} CSV files:**")
                        
                        # Create a table of CSV files
                        #csv_info = []
                        #for csv_file in csv_files:
                            #file_size = os.path.getsize(csv_file)
                            #file_size_mb = file_size / (1024 * 1024)
                            #csv_info.append({
                                #"File Name": os.path.basename(csv_file),
                                #"Path": csv_file,
                                #"Size (MB)": f"{file_size_mb:.2f}",
                                #"Actions": "Download"
                            #})
                        
                        # Display CSV files table
                        #csv_df = pd.DataFrame(csv_info)
                        #st.dataframe(csv_df, use_container_width=True)
                        
                        # Download buttons for each CSV
                        #st.write("**Download Individual CSV Files:**")
                        #for csv_file in csv_files:
                            #try:
                            #    with open(csv_file, 'rb') as f:
                            #        csv_data = f.read()
                                
                                #filename = os.path.basename(csv_file)
                                #st.download_button(
                                    #label=f"📥 {filename}",
                                    #data=csv_data,
                                    #file_name=filename,
                                    #mime="text/csv",
                                    #key=f"csv_{filename}_{hash(csv_file)}"
                                #)
                            #except Exception as e:
                                #st.error(f"Error reading {csv_file}: {str(e)}")
                        
                        # Open output directory button
                        if st.button("📁 Open Output Directory", type="secondary", use_container_width=True):
                            try:
                                import subprocess
                                import platform
                                
                                if platform.system() == "Windows":
                                    subprocess.run(["explorer", output_dir])
                                elif platform.system() == "Darwin":  # macOS
                                    subprocess.run(["open", output_dir])
                                else:  # Linux
                                    subprocess.run(["xdg-open", output_dir])
                                
                                st.success("✅ Output directory opened!")
                            except Exception as e:
                                st.warning(f"Could not open directory automatically: {str(e)}")
                                st.info(f"Please manually navigate to: {os.path.abspath(output_dir)}")
                    else:
                        st.info("ℹ️ No CSV files found in the output directory.")
                else:
                    st.warning("⚠️ Output directory not found. CSV reports will be generated here after analysis.")
                
                
        else:
            st.info("💡 Remove ID columns first to access the comprehensive analysis tabs.")
        

        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a file or connect to a database to begin the analysis")

