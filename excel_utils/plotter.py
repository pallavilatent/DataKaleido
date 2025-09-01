import matplotlib.pyplot as plt
import os

def plot_numeric_columns(df, output_path="plot.png"):
    """Plots all numeric columns and saves to image."""
    numeric_cols = df.select_dtypes(include="number")
    if numeric_cols.empty:
        return None

    ax = numeric_cols.plot(kind='line', figsize=(10, 6), title="Line Plot of Numeric Columns")
    plt.ylabel("Values")
    plt.xlabel("Index")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    return output_path
