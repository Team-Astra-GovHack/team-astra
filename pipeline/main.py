import argparse
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
import json


class DataCleaningPipeline:
    def __init__(self, data_dictionary: Optional[Dict] = None, 
                 metadata: Optional[Dict] = None):
        """
        Initialize the data cleaning pipeline with optional data dictionary and metadata.
        
        Args:
            data_dictionary: Dictionary containing column descriptions and specifications
            metadata: Additional metadata about the dataset
        """
        self.data_dictionary = data_dictionary or {}
        self.metadata = metadata or {}
        self.cleaning_decisions = {}
        self.limitations = []
        
    def _log_decision(self, sheet_name: str, column: str, issue: str, action: str, details: Any = None):
        """Log cleaning decisions for reproducibility."""
        key = f"{sheet_name}.{column}.{issue}"
        self.cleaning_decisions[key] = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _add_limitation(self, limitation: str):
        """Add a data limitation to be documented."""
        self.limitations.append(limitation)
    
    def _standardize_column_names(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
        """Standardize column names according to data dictionary or snake_case convention."""
        old_cols = df.columns.tolist()
        
        # Use data dictionary if available, otherwise apply standard transformation
        if self.data_dictionary and 'columns' in self.data_dictionary:
            col_mapping = {}
            for col in df.columns:
                clean_name = self.data_dictionary['columns'].get(col, {}).get('standard_name')
                if clean_name:
                    col_mapping[col] = clean_name
                else:
                    # Fallback to automated cleaning
                    clean_name = (
                        col.strip()
                        .lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("%", "pct")
                        .replace("/", "_per_")
                    )
                    # Remove special characters and multiple underscores
                    clean_name = re.sub(r'[^\w_]', '', clean_name)
                    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
                    col_mapping[col] = clean_name
            
            df = df.rename(columns=col_mapping)
        else:
            # Automated column name cleaning
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.replace(r"\s+", "_", regex=True)
            )
        
        log.append(f"[{sheet_name}] Standardized column names: {old_cols} -> {df.columns.tolist()}")
        return df
    
    def _ensure_primary_key(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
        """Ensure each table has a stable, unique primary key."""
        if 'id' not in df.columns:
            # Create a unique ID if none exists
            df.insert(0, 'id', range(1, 1 + len(df)))
            log.append(f"[{sheet_name}] Added primary key column 'id'")
            self._log_decision(sheet_name, "id", "missing_primary_key", "added_auto_increment_id")
        elif df['id'].isnull().any():
            # Fill missing IDs
            missing_count = df['id'].isnull().sum()
            max_id = df['id'].max()
            if pd.isna(max_id):
                max_id = 0
            df.loc[df['id'].isnull(), 'id'] = range(int(max_id) + 1, int(max_id) + 1 + missing_count)
            log.append(f"[{sheet_name}] Filled {missing_count} missing IDs")
            self._log_decision(sheet_name, "id", "missing_ids", "filled_with_sequential_values", missing_count)
        
        return df
    
    def _correct_data_types(
        self, df: pd.DataFrame, sheet_name: str, log: list
    ) -> pd.DataFrame:
        """
        Convert columns to appropriate data types based on content and data dictionary.
        Handles integers, floats, booleans, datetimes (with epoch detection), and text.
        """

        type_conversions = {}

        # Default date format (US style: MM/DD/YYYY)
        default_date_format = "%m/%d/%Y"
        if hasattr(self, "config") and isinstance(self.config, dict):
            default_date_format = self.config.get("default_date_format", default_date_format)

        for col in df.columns:
            # Skip ID column for type conversion
            if col == "id":
                continue

            # -------------------------------
            # 1. Use data dictionary if available
            # -------------------------------
            if self.data_dictionary and "columns" in self.data_dictionary:
                col_info = self.data_dictionary["columns"].get(col, {})
                if col_info.get("data_type"):
                    target_type = col_info["data_type"].lower()
                    try:
                        if target_type in ["int", "integer"]:
                            df[col] = (
                                pd.to_numeric(df[col], errors="coerce").astype("Int64")
                            )
                            type_conversions[col] = "integer"

                        elif target_type in ["float", "double", "decimal"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            type_conversions[col] = "float"

                        elif target_type in ["date", "datetime"]:
                            # Parse with known format first
                            df[col] = pd.to_datetime(
                                df[col], errors="coerce", format=default_date_format
                            )

                            # If still integers, detect epoch unit
                            if pd.api.types.is_integer_dtype(df[col]):
                                max_val = df[col].max()
                                if max_val > 1e12:  # nanoseconds
                                    df[col] = pd.to_datetime(
                                        df[col], unit="ns", errors="coerce"
                                    )
                                elif max_val > 1e9:  # milliseconds
                                    df[col] = pd.to_datetime(
                                        df[col], unit="ms", errors="coerce"
                                    )
                                else:  # seconds
                                    df[col] = pd.to_datetime(
                                        df[col], unit="s", errors="coerce"
                                    )

                            # Normalize to midnight for consistency
                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                df[col] = df[col].dt.normalize()

                            type_conversions[col] = "datetime"

                        elif target_type == "boolean":
                            # Handle various boolean representations
                            df[col] = (
                                df[col]
                                .astype(str)
                                .str.lower()
                                .map(
                                    {
                                        "true": True,
                                        "false": False,
                                        "yes": True,
                                        "no": False,
                                        "1": True,
                                        "0": False,
                                        "t": True,
                                        "f": False,
                                    }
                                )
                            )
                            type_conversions[col] = "boolean"

                        else:
                            df[col] = df[col].astype(str)
                            type_conversions[col] = "string"

                    except Exception as e:
                        log.append(
                            f"[{sheet_name}] Warning: Could not convert {col} to {target_type}: {str(e)}"
                        )
                        self._log_decision(
                            sheet_name,
                            col,
                            "type_conversion_failed",
                            "kept_original_type",
                            str(e),
                        )
                    continue

            # -------------------------------
            # 2. Automated type detection
            # -------------------------------
            col_sample = df[col].dropna()
            if len(col_sample) == 0:
                continue  # Skip empty columns

            # Already numeric
            if df[col].dtype in [np.float64, np.int64]:
                type_conversions[col] = "numeric"
                continue

            # Try numeric conversion
            try:
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                if numeric_vals.notna().mean() > 0.8:  # >80% numeric
                    df[col] = numeric_vals
                    type_conversions[col] = "numeric"
                    continue
            except Exception:
                pass

            # Try datetime conversion
            try:
                date_vals = pd.to_datetime(
                    df[col], errors="coerce", format=default_date_format
                )

                # If still integers, detect epoch unit
                if pd.api.types.is_integer_dtype(date_vals):
                    max_val = date_vals.max()
                    if max_val > 1e12:  # nanoseconds
                        date_vals = pd.to_datetime(date_vals, unit="ns", errors="coerce")
                    elif max_val > 1e9:  # milliseconds
                        date_vals = pd.to_datetime(date_vals, unit="ms", errors="coerce")
                    else:  # seconds
                        date_vals = pd.to_datetime(date_vals, unit="s", errors="coerce")

                if date_vals.notna().mean() > 0.5:  # >50% valid dates
                    df[col] = date_vals.dt.normalize()
                    type_conversions[col] = "datetime"
                    continue
            except Exception:
                pass

            # Categorical detection
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.3:  # Low cardinality
                type_conversions[col] = "categorical"
            else:
                type_conversions[col] = "text"

        log.append(f"[{sheet_name}] Data type conversions: {type_conversions}")
        return df

        def _handle_missing_values(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
            """Handle missing values based on column type and data dictionary guidance."""
            missing_before = df.isnull().sum().to_dict()
            log.append(f"[{sheet_name}] Missing values before handling: {missing_before}")
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue
                    
                # Use data dictionary guidance if available
                if self.data_dictionary and 'columns' in self.data_dictionary:
                    col_info = self.data_dictionary['columns'].get(col, {})
                    if col_info.get('handling_missing') == 'keep_null':
                        self._log_decision(sheet_name, col, "missing_values", "kept_as_null", missing_count)
                        continue
                    elif col_info.get('handling_missing') == 'impute_mean':
                        if df[col].dtype in [np.float64, np.int64]:
                            impute_val = df[col].mean()
                            df[col] = df[col].fillna(impute_val)
                            self._log_decision(sheet_name, col, "missing_values", "imputed_with_mean", 
                                            {"count": missing_count, "value": impute_val})
                            continue
                
                # Automated handling based on data type
                if df[col].dtype in [np.float64, np.int64]:
                    # For numeric columns, use median imputation
                    impute_val = df[col].median()
                    df[col] = df[col].fillna(impute_val)
                    self._log_decision(sheet_name, col, "missing_values", "imputed_with_median", 
                                    {"count": missing_count, "value": impute_val})
                elif df[col].dtype.name in ['category', 'object']:
                    # For categorical columns, use mode or "Unknown"
                    if not df[col].mode().empty:
                        impute_val = df[col].mode()[0]
                        df[col] = df[col].fillna(impute_val)
                        self._log_decision(sheet_name, col, "missing_values", "imputed_with_mode", 
                                        {"count": missing_count, "value": impute_val})
                    else:
                        df[col] = df[col].fillna("Unknown")
                        self._log_decision(sheet_name, col, "missing_values", "filled_with_unknown", 
                                        {"count": missing_count})
                elif 'date' in str(df[col].dtype).lower():
                    # For datetime columns, use forward fill or backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    self._log_decision(sheet_name, col, "missing_values", "filled_with_ffill_bfill", 
                                    {"count": missing_count})
            
            missing_after = df.isnull().sum().to_dict()
            log.append(f"[{sheet_name}] Missing values after handling: {missing_after}")
            return df
        
        def _standardize_categoricals(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
            """Standardize categorical values and create lookup tables where needed."""
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                # Skip if column has too many unique values (likely free text)
                if df[col].nunique() > 100:
                    continue
                    
                # Standardize text: trim, normalize case
                df[col] = df[col].astype(str).str.strip()
                
                # Apply specific standardization based on common patterns
                if col.lower() in ['yes_no', 'yn', 'boolean']:
                    # Standardize yes/no values
                    df[col] = df[col].str.lower().map({
                        'yes': 'Yes', 'y': 'Yes', 'true': 'Yes', 't': 'Yes', '1': 'Yes',
                        'no': 'No', 'n': 'No', 'false': 'No', 'f': 'No', '0': 'No'
                    }).fillna(df[col])
                elif any(term in col.lower() for term in ['status', 'type', 'category']):
                    # Standardize status/type/category values
                    df[col] = df[col].str.title()
                
                # Create lookup table for complex enumerations
                if df[col].nunique() < 20:  # Reasonable number of categories for a lookup table
                    lookup_table = {value: value for value in df[col].unique()}
                    self._log_decision(sheet_name, col, "categorical_values", "created_lookup_table", 
                                    {"values": list(lookup_table.keys())})
            
            log.append(f"[{sheet_name}] Standardized categorical columns: {list(categorical_cols)}")
            return df
        
        def _validate_numerical_ranges(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
            """Validate numerical ranges and flag impossible values."""
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            validation_issues = {}
            for col in numerical_cols:
                issues = []
                
                # Check for negative values where they shouldn't exist
                if any(term in col.lower() for term in ['age', 'price', 'amount', 'cost', 'revenue']):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        issues.append(f"{negative_count} negative values")
                        # Option: set negative values to zero or absolute value
                        # df.loc[df[col] < 0, col] = 0
                
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outlier_count > 0:
                    issues.append(f"{outlier_count} outliers (IQR method)")
                
                # Check for values outside plausible ranges based on column name
                if 'age' in col.lower():
                    impossible_age_count = (df[col] > 120).sum()
                    if impossible_age_count > 0:
                        issues.append(f"{impossible_age_count} values > 120")
                
                if 'pct' in col.lower() or 'percentage' in col.lower():
                    out_of_range_count = ((df[col] < 0) | (df[col] > 100)).sum()
                    if out_of_range_count > 0:
                        issues.append(f"{out_of_range_count} values outside [0, 100] range")
                
                if issues:
                    validation_issues[col] = issues
                    self._log_decision(sheet_name, col, "validation_issues", "flagged_issues", issues)
            
            if validation_issues:
                log.append(f"[{sheet_name}] Numerical validation issues: {validation_issues}")
                self._add_limitation(f"Numerical validation issues in {sheet_name}: {validation_issues}")
            
            return df
        
        def _clean_text_fields(self, df: pd.DataFrame, sheet_name: str, log: list) -> pd.DataFrame:
            """Clean and standardize text fields."""
            text_cols = df.select_dtypes(include=['object']).columns
            
            for col in text_cols:
                # Basic cleaning
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize case based on content type
                if any(term in col.lower() for term in ['name', 'title', 'description']):
                    df[col] = df[col].str.title()
                elif any(term in col.lower() for term in ['code', 'id', 'postcode']):
                    df[col] = df[col].str.upper()
                elif any(term in col.lower() for term in ['email']):
                    df[col] = df[col].str.lower()
                
                # Normalize placeholders
                placeholder_map = {
                    'n/a': 'Unknown', 'na': 'Unknown', 'null': 'Unknown', 
                    'none': 'Unknown', '': 'Unknown', 'nan': 'Unknown'
                }
                for placeholder, replacement in placeholder_map.items():
                    df[col] = df[col].replace(placeholder, replacement, regex=False)
            
            log.append(f"[{sheet_name}] Cleaned text columns: {list(text_cols)}")
            return df
        
        def _enrich_data(self, df: pd.DataFrame, sheet_name: str, log: list, 
                        data_source: str, last_updated: str) -> pd.DataFrame:
            """Add enrichment columns for context and trust."""
            # Add data source column
            if 'data_source' not in df.columns:
                df['data_source'] = data_source
                log.append(f"[{sheet_name}] Added data_source column: {data_source}")
            
            # Add last updated timestamp
            if 'last_updated' not in df.columns:
                df['last_updated'] = pd.to_datetime(last_updated)
                log.append(f"[{sheet_name}] Added last_updated column: {last_updated}")
            
            return df
        
        def clean_dataframe(self, df: pd.DataFrame, sheet_name: str, log: list, 
                        data_source: str, last_updated: str) -> pd.DataFrame:
            """Apply the complete cleaning pipeline to a dataframe."""
            log.append(f"\n--- Cleaning sheet: {sheet_name} ---")
            
            # Record original shape
            original_shape = df.shape
            log.append(f"[{sheet_name}] Original shape: {original_shape}")
            
            # Apply cleaning steps
            df = self._standardize_column_names(df, sheet_name, log)
            df = self._ensure_primary_key(df, sheet_name, log)
            df = self._correct_data_types(df, sheet_name, log)
            df = self._handle_missing_values(df, sheet_name, log)
            df = self._standardize_categoricals(df, sheet_name, log)
            df = self._validate_numerical_ranges(df, sheet_name, log)
            df = self._clean_text_fields(df, sheet_name, log)
            
            # Remove duplicates (after cleaning)
            before_dedup = df.shape[0]
            df = df.drop_duplicates()
            after_dedup = df.shape[0]
            if before_dedup != after_dedup:
                log.append(f"[{sheet_name}] Removed {before_dedup - after_dedup} duplicate rows")
                self._log_decision(sheet_name, "all", "duplicates", "removed_duplicates", 
                                before_dedup - after_dedup)
            
            # Enrich with metadata
            df = self._enrich_data(df, sheet_name, log, data_source, last_updated)
            
            # Record final shape
            final_shape = df.shape
            log.append(f"[{sheet_name}] Final shape: {final_shape}")
            log.append(f"[{sheet_name}] Rows changed: {original_shape[0] - final_shape[0]}")
            log.append(f"[{sheet_name}] Columns changed: {original_shape[1] - final_shape[1]}")
            
            return df
        
        def generate_documentation(self, output_dir: str, dataset_name: str):
                """Generate comprehensive documentation for the cleaned dataset."""
                # Ensure output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Prepare parts outside the f-string
                cleaning_decisions_str = json.dumps(
                    self.cleaning_decisions, indent=2, default=str
                )
                limitations_text = "".join(f"- {limitation}\n" for limitation in self.limitations)
                columns_desc = (
                    json.dumps(self.data_dictionary.get('columns', {}), indent=2, default=str)
                    if self.data_dictionary else 'No data dictionary available'
                )

                # Create README content
                readme_content = f"""# Data Cleaning Documentation for {dataset_name}

        ## Overview
        This document describes the data cleaning process applied to the {dataset_name} dataset.

        ## Data Source Information
        - **Data Source**: {self.metadata.get('data_source', 'Unknown')}
        - **Last Updated**: {self.metadata.get('last_updated', 'Unknown')}
        - **Time Period Covered**: {self.metadata.get('time_period', 'Unknown')}
        - **Geographic Scope**: {self.metadata.get('geographic_scope', 'Unknown')}

        ## Cleaning Decisions
        The following cleaning decisions were made during processing:

        {cleaning_decisions_str}

        ## Known Limitations
        {limitations_text}

        ## Column Descriptions
        {columns_desc}

        ## Processing Information
        Processing Date: {datetime.now().isoformat()}

        Cleaning Pipeline Version: 2.0 (GovHack 2024 Enhanced)
        """

                # Write README.md
                with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
                    f.write(readme_content)

                # Also save cleaning decisions as JSON
                with open(os.path.join(output_dir, "cleaning_decisions.json"), "w", encoding="utf-8") as f:
                    json.dump(self.cleaning_decisions, f, indent=2, default=str)

def clean_data(
    file_path: str,
    output_dir: str = "cleaned_data",
    data_dictionary: Optional[Dict] = None,
    metadata: Optional[Dict] = None
):
    """
    Main function to clean data from CSV or Excel files.

    Args:
        file_path: Path to the input file
        output_dir: Directory to save cleaned files
        data_dictionary: Optional data dictionary for guidance
        metadata: Optional metadata about the dataset
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize cleaning pipeline
    pipeline = DataCleaningPipeline(data_dictionary, metadata)
    log = []
    base_name = os.path.basename(file_path).split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract metadata if not provided
    if metadata is None:
        metadata = {
            "data_source": os.path.basename(file_path),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "time_period": "Unknown",
            "geographic_scope": "Unknown"
        }

    # Read data based on file type
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        cleaned_df = pipeline.clean_dataframe(df, "Sheet1", log,
                                              metadata["data_source"],
                                              metadata["last_updated"])

        cleaned_file = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.csv")
        cleaned_df.to_csv(cleaned_file, index=False)
        log.append(f"Saved cleaned CSV: {cleaned_file}")

        cleaned_dfs = {"Sheet1": cleaned_df}

    elif file_path.endswith((".xls", ".xlsx")):
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        cleaned_dfs = {}
        output_excel = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.xlsx")

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            for sheet_name, df in all_sheets.items():
                cleaned_df = pipeline.clean_dataframe(df, sheet_name, log,
                                                     metadata["data_source"],
                                                     metadata["last_updated"])
                cleaned_df.to_excel(writer, sheet_name=sheet_name, index=False)
                cleaned_dfs[sheet_name] = cleaned_df

                # Save each sheet as separate CSV
                sheet_csv_path = os.path.join(
                    output_dir, f"{base_name}_{sheet_name}_cleaned_{timestamp}.csv"
                )
                cleaned_df.to_csv(sheet_csv_path, index=False)
                log.append(f"[{sheet_name}] Saved cleaned CSV: {sheet_csv_path}")

        log.append(f"Saved cleaned Excel with all sheets -> {output_excel}")

    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    # Generate documentation
    pipeline.generate_documentation(output_dir, base_name)

    # Save cleaning log
    log_file = os.path.join(output_dir, f"{base_name}_cleaning_log_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write("\n".join(log))

    print(f"✅ Cleaning complete!")
    print(f"📂 Cleaned data saved in: {output_dir}")
    print(f"📝 Cleaning log saved at: {log_file}")
    print(f"📄 Documentation generated: README.md and cleaning_decisions.json")

    return cleaned_dfs, log

def main():
    parser = argparse.ArgumentParser(description="Enhanced Data Cleaning Pipeline for GovHack 2024")
    parser.add_argument("input_file", help="Path to the input CSV or Excel file")
    parser.add_argument("-o", "--output_dir", default="cleaned_data",
                        help="Directory to save cleaned files")
    parser.add_argument("-d", "--data_dictionary", default=None,
                        help="Path to JSON file containing data dictionary")
    parser.add_argument("-m", "--metadata", default=None,
                        help="Path to JSON file containing dataset metadata")

    args = parser.parse_args()

    # Load data dictionary and metadata if provided
    data_dict = None
    if args.data_dictionary and os.path.exists(args.data_dictionary):
        with open(args.data_dictionary, 'r') as f:
            data_dict = json.load(f)

    metadata = None
    if args.metadata and os.path.exists(args.metadata):
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)

    clean_data(args.input_file, args.output_dir, data_dict, metadata)

if __name__ == "__main__":
    main()