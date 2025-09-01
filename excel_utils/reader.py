import logging
import os
import pandas as pd
from pandas.errors import ParserError
from dateutil import parser

def is_probable_year_series(series):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return series.between(1900, 2100).mean() > 0.8

def load_data(file_path, delimiter=None, sheet_name=None):
    threshold=0.7
    ext = os.path.splitext(file_path)[-1].lower()
    logging.info("Reading the file")

    if ext in ['.csv']:
        df = pd.read_csv(file_path, delimiter=delimiter if delimiter else ',')
    elif ext in ['.tsv']:
        df = pd.read_csv(file_path, delimiter=delimiter if delimiter else '\t')
    elif ext in ['.txt']: 
        df = pd.read_csv(file_path, delimiter=delimiter if delimiter else r'\s+')
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, sheet_name=sheet_name) # made change here on 14/8
    elif ext in ['.json']:
        df = load_and_flatten_json(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    '''elif ext in ['.sav']:
        if pyreadstat:
            df, meta = pyreadstat.read_sav(file_path)
        else:
            raise ImportError("Please install 'pyreadstat' to read .sav files.")'''
    
    

    # Check for NaT entries
    #print(f"Remaining NaT entries in 'date': {df['released'].isna().sum()}")
    
    converted_cols = []

    #### Old code ########## preserve
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         try:
    #             converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
    #             if converted.notna().sum() > len(df) > threshold:
    #                 df[col] = safe_excel_date_parser(df[col])
    #                 df[col] = converted
    #                 converted_cols.append(col)
    #         except Exception:
    #             pass
                

    # for col in df.columns:
    #     # Check if column is an object type and can be converted to datetime
    #     if df[col].dtype == 'object':
    #         # Attempt a robust conversion for string-based dates
    #         converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            
    #         # Use a threshold to confirm it's a date column
    #         if converted.notna().sum() / len(df) > threshold:
    #             df[col] = converted  # <-- Correctly assign the converted series
    #             converted_cols.append(col)

    # # Simplified check for integer-based years
    # for col in df.columns:
    #     if df[col].dtype == 'int64':
    #         if is_probable_year_series(df[col]):
    #             df[col] = pd.to_datetime(df[col].astype(str), format='%Y', errors='coerce')
    #             converted_cols.append(col)
            # try: 
            #     is_percent_series = df[col].dropna().astype(str).str.contains('%')
            #     if not is_percent_series.empty and is_percent_series.sum() / len(is_percent_series) > 0.5:
            #         print(f"Detected and converted column '{col}' to percentage.")
            #         df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ''), errors = 'coerce')/100
            #         converted_cols_percentage.append(col)
            # except Exception:
            #     pass

       #### Old code ########## preserve 
        # if df[col].dtype == 'int64':
        #     if is_probable_year_series(df[col]):
        #         df[col] = pd.to_datetime(df[col].astype(str), format='%Y', errors='coerce')
        #         converted_cols.append(col)
    
    for col in df.columns:
        # The primary candidates for dates are columns with 'object' dtype.
        if df[col].dtype == 'object':
            # Attempt to convert the entire series to datetime.
            # `errors='coerce'` turns any invalid date strings into `NaT`.
            # `infer_datetime_format=True` helps pandas parse common date formats quickly.
            converted_cols = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            #df[col] = converted_cols
            if converted.notna().sum() / max(len(df), 1) <= threshold:
                try:
                    converted = df[col].apply(
                        lambda x: parser.parse(str(x)) if pd.notna(x) else pd.NaT
                    )
                except Exception:
                    converted = pd.Series([pd.NaT] * len(df))
            #Check if a sufficient portion of the column was successfully converted.
            if converted_cols.notna().sum() / max(len(df), 1) > threshold:
                # If the threshold is met, we re-assign the column with the new datetime series.
                df[col] = converted_cols.astype("datetime64[ns]")
                logging.info(f"Automatically converted object column '{col}' to datetime.")
        
        # We also check for `int64` columns that might represent years.
        elif df[col].dtype == 'int64':
            if is_probable_year_series(df[col]):
                # Convert the year to a datetime object.
                df[col] = pd.to_datetime(df[col].astype(str), format='%Y', errors='coerce')
                logging.info(f"Automatically converted integer column '{col}' to datetime (year).")


    return df


def load_and_flatten_json(file_path):
    import json
    import pandas as pd
    from pandas import json_normalize

    with open(file_path, "r", encoding="utf-8") as f:  #  fix is here
        data = json.load(f)

    if isinstance(data, dict):
        return json_normalize(data)
    elif isinstance(data, list):
        return json_normalize(data)
    else:
        raise ValueError("Unsupported JSON structure")


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
        except (ParserError, ValueError, OverflowError):
            parsed_dates.append(pd.NaT)

    return pd.Series(parsed_dates)

# Apply only to the 'date' column
