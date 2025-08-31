import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import json

class EmployeeLeaveCleaner:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.cleaning_log = []
        self.cleaning_decisions = {}
        self.limitations = []

    def _log(self, message):
        print(message)
        self.cleaning_log.append(message)

    def _log_decision(self, column, issue, action, details=None):
        key = f"{column}.{issue}"
        self.cleaning_decisions[key] = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def standardize_column_names(self, df):
        old_cols = df.columns.tolist()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
        )
        self._log(f"Standardized column names: {old_cols} -> {df.columns.tolist()}")
        return df

    def ensure_primary_key(self, df):
        if "id" not in df.columns:
            df.insert(0, "id", range(1, len(df) + 1))
            self._log("Added primary key column 'id'")
            self._log_decision("id", "missing_primary_key", "added_auto_increment_id")
        return df

    def correct_data_types(self, df):
        for col in df.columns:
            if col in ["id"]:
                continue
            if "date" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                self._log_decision(col, "type_conversion", "converted_to_datetime")
            elif "days" in col or "taken" in col:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                self._log_decision(col, "type_conversion", "converted_to_numeric")
            else:
                df[col] = df[col].astype(str).str.strip()
        return df

    def clean_categoricals(self, df):
        if "leave_type" in df.columns:
            df["leave_type"] = (
                df["leave_type"]
                .str.strip()
                .str.title()
                .replace({
                    "Maternity Le": "Maternity Leave",
                    "Paternity Lea": "Paternity Leave",
                    "Sick Lea": "Sick Leave",
                    "Earned Lea": "Earned Leave",
                    "Casual Lea": "Casual Leave"
                })
            )
            self._log_decision("leave_type", "standardization", "normalized leave types")
        return df

    def handle_missing_values(self, df):
        missing_before = df.isnull().sum().to_dict()
        self._log(f"Missing values before: {missing_before}")

        for col in df.columns:
            if df[col].dtype.kind in "biufc":  # numeric
                df[col] = df[col].fillna(df[col].median())
            elif "date" in col:
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            else:
                df[col] = df[col].fillna("Unknown")

        missing_after = df.isnull().sum().to_dict()
        self._log(f"Missing values after: {missing_after}")
        return df

    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            self._log(f"Removed {before - after} duplicate rows")
        return df

    def enrich_metadata(self, df, source_file):
        df["data_source"] = os.path.basename(source_file)
        df["last_updated"] = pd.to_datetime(datetime.now().date())
        return df

    def clean_excel(self, file_path, output_dir="cleaned_data"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = os.path.basename(file_path).split(".")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_sheets = pd.read_excel(file_path, sheet_name=None)
        cleaned_dfs = {}
        output_excel = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.xlsx")

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            for sheet_name, df in all_sheets.items():
                self._log(f"\n--- Cleaning sheet: {sheet_name} ---")
                df = self.standardize_column_names(df)
                df = self.ensure_primary_key(df)
                df = self.correct_data_types(df)
                df = self.clean_categoricals(df)
                df = self.handle_missing_values(df)
                df = self.remove_duplicates(df)
                df = self.enrich_metadata(df, file_path)

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                cleaned_dfs[sheet_name] = df

        # Save log
        log_file = os.path.join(output_dir, f"{base_name}_cleaning_log_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write("\n".join(self.cleaning_log))

        # Save cleaning decisions
        with open(os.path.join(output_dir, "cleaning_decisions.json"), "w") as f:
            json.dump(self.cleaning_decisions, f, indent=2)

        self._log(f"\n✅ Cleaning complete! Cleaned file saved at: {output_excel}")
        return cleaned_dfs

# ----------------- Run Example -----------------
if __name__ == "__main__":
    cleaner = EmployeeLeaveCleaner(metadata={"project": "Employee Leave Dataset"})
    cleaner.clean_excel("employee-leave-tracking-data.xlsx")








