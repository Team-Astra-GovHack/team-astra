import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import json

class EmployeeDataCleaner:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.cleaning_log = []
        self.cleaning_decisions = {}

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

    def parse_dates(self, df, date_cols):
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                self._log_decision(col, "type_conversion", "converted_to_datetime")
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

    def clean_employee_master(self, df):
        df = self.standardize_column_names(df)
        df = self.parse_dates(df, ["engdt", "termdt", "dob"])
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        return df

    def clean_performance(self, df):
        df = self.standardize_column_names(df)
        df = self.parse_dates(df, ["perfdate"])
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        return df

    def clean_actions(self, df):
        df = self.standardize_column_names(df)
        df = self.parse_dates(df, ["effectivedt"])
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        return df

    def merge_datasets(self, emp_df, perf_df, act_df):
        # Merge performance and actions into employee master
        merged = emp_df.copy()

        # Performance: one-to-many, keep latest rating
        perf_latest = (
            perf_df.sort_values("perfdate")
            .groupby("empid")
            .tail(1)
            .set_index("empid")
        )
        merged = merged.merge(
            perf_latest[["rating", "perfdate"]],
            on="empid",
            how="left"
        )

        # Actions: keep latest effective date
        act_latest = (
            act_df.sort_values("effectivedt")
            .groupby("empid")
            .tail(1)
            .set_index("empid")
        )
        merged = merged.merge(
            act_latest[["actionid", "effectivedt"]],
            on="empid",
            how="left"
        )

        self._log("Merged employee, performance, and actions datasets")
        return merged

    def save_cleaned(self, df, output_dir="cleaned_data", filename="cleaned_employees.csv"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        self._log(f"✅ Cleaned dataset saved at: {output_path}")
        return output_path


# ----------------- Run Example -----------------
if __name__ == "__main__":
    cleaner = EmployeeDataCleaner(metadata={"project": "Employee Master + Perf + Actions"})

    # Load your raw CSVs (replace with actual file paths)
    emp_df = pd.read_csv("tbl_Employee.csv")
    perf_df = pd.read_csv("tbl_Perf.csv")
    act_df = pd.read_csv("tbl_Action.csv")

    emp_df = cleaner.clean_employee_master(emp_df)
    perf_df = cleaner.clean_performance(perf_df)
    act_df = cleaner.clean_actions(act_df)

    merged_df = cleaner.merge_datasets(emp_df, perf_df, act_df)
    cleaner.save_cleaned(merged_df)