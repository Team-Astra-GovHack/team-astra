import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime


def clean_dataframe(df: pd.DataFrame, sheet_name: str, log: list):
    # --- 1. Standardize column names ---
    old_cols = df.columns.tolist()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(" ", "_")
    )
    log.append(f"[{sheet_name}] Standardized column names: {old_cols} -> {df.columns.tolist()}")

    # --- 2. Remove duplicates ---
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    log.append(f"[{sheet_name}] Removed {before - after} duplicate rows")

    # --- 3. Handle missing values ---
    missing_report = df.isnull().sum().to_dict()
    log.append(f"[{sheet_name}] Missing values before cleaning: {missing_report}")

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            )

    df = df.fillna(method="ffill").fillna(method="bfill")

    # --- 4. Standardize text ---
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.title()

    # --- 5. Try automatic type conversion ---
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        except Exception:
            pass
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    log.append(f"[{sheet_name}] Applied type conversions and text cleaning")

    return df


def clean_data(file_path: str, output_dir: str = "cleaned_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log = []
    base_name = os.path.basename(file_path).split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- If CSV ---
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        df = clean_dataframe(df, "Sheet1", log)

        cleaned_file = os.path.join(output_dir, f"{base_name}cleaned{timestamp}.csv")
        df.to_csv(cleaned_file, index=False)
        log.append(f"Saved cleaned CSV: {cleaned_file}")

        cleaned_dfs = {"Sheet1": df}

    # --- If Excel ---
    elif file_path.endswith((".xls", ".xlsx")):
        all_sheets = pd.read_excel(file_path, sheet_name=None)  # dict {sheet_name: df}
        cleaned_dfs = {}
        output_excel = os.path.join(output_dir, f"{base_name}cleaned{timestamp}.xlsx")

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            for sheet_name, df in all_sheets.items():
                cleaned_df = clean_dataframe(df, sheet_name, log)
                cleaned_df.to_excel(writer, sheet_name=sheet_name, index=False)
                cleaned_dfs[sheet_name] = cleaned_df

                # Save each sheet as separate CSV
                sheet_csv_path = os.path.join(
                    output_dir, f"{base_name}{sheet_name}_cleaned{timestamp}.csv"
                )
                cleaned_df.to_csv(sheet_csv_path, index=False)
                log.append(f"[{sheet_name}] Saved cleaned CSV: {sheet_csv_path}")

        log.append(f"Saved cleaned Excel with all sheets -> {output_excel}")

    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    # --- Save cleaning log ---
    log_file = os.path.join(output_dir, f"{base_name}cleaning_log{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write("\n".join(log))

    print(f"✅ Cleaning complete!")
    print(f"📂 Cleaned data saved in: {output_dir}")
    print(f"📝 Cleaning log saved at: {log_file}")

    return cleaned_dfs, log


def main():
    parser = argparse.ArgumentParser(description="General-purpose Data Cleaning Pipeline")
    parser.add_argument("input_file", help="Path to the input CSV or Excel file")
    parser.add_argument(
        "-o", "--output_dir", default="cleaned_data", help="Directory to save cleaned files"
    )
    args = parser.parse_args()

    clean_data(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()