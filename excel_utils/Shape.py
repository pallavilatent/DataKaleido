import logging

def get_dataframe_shape(df):
    if df is not None:
        return df.shape
    logging.error("Error reading shape of Excel file: DataFrame is None")
    return None

def get_data_types(df):
    """Returns column data types as a dictionary."""
    if df is not None:
        return df.dtypes.apply(str).to_dict()
    return {}

def get_head(df, n=5):
    """Returns the first n rows of the DataFrame as a list of dictionaries."""
    if df is not None:
        return df.head(n)
    return []


def cleaning_data_frame(df):
    """
    Detects and removes columns whose names start with or end with 'id' (case-insensitive).

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with ID-like columns removed.
    """
    
    print("--- Duplicate Row Handling ---")

    # 1. Identify duplicate rows
    # keep=False marks all occurrences of duplicates as True
    duplicate_rows = df[df.duplicated(keep=False)]

    # 2. Output the count of duplicate rows
    num_duplicate_rows = len(duplicate_rows)
    print(f"Total number of duplicate rows found (including all occurrences): {num_duplicate_rows}")

    # Count of unique duplicate rows (i.e., how many distinct rows appear more than once)
    num_unique_duplicate_rows = len(df[df.duplicated(keep='first')])
    print(f"Number of unique rows that are duplicates (first occurrence kept, subsequent marked): {num_unique_duplicate_rows}")


    # 3. Output the percentage of duplicate rows
    total_rows = len(df)
    if total_rows > 0:
        percentage_duplicates = (num_unique_duplicate_rows / total_rows) * 100
        print(f"Percentage of duplicate rows: {percentage_duplicates:.0f}%")
    else:
        percentage_duplicates = 0
        print("DataFrame is empty, no duplicates to process.")
        return df.copy(), num_duplicate_rows, percentage_duplicates # Return an empty copy if input is empty

    if num_duplicate_rows == 0:
        print("No duplicate rows found. Returning the original DataFrame.")
        return df.copy(), num_duplicate_rows, percentage_duplicates # Return a copy if no duplicates were found

    # 4. Remove duplicate rows from the DataFrame
    # drop_duplicates() by default keeps the first occurrence
    df_no_duplicates = df.drop_duplicates()

    num_rows_after_removal = len(df_no_duplicates)
    print(f"Original number of rows: {total_rows}")
    print(f"Number of rows after removing duplicates: {num_rows_after_removal}")
    print(f"Number of rows removed: {total_rows - num_rows_after_removal}")

    print("--- Duplicate Row Handling Completed ---")
    
    if df_no_duplicates.empty:
        print("The DataFrame is empty. No columns to remove.")
        return df_no_duplicates.copy(), num_duplicate_rows, percentage_duplicates # Return a copy to maintain consistency

   
    return df_no_duplicates, num_duplicate_rows, percentage_duplicates

