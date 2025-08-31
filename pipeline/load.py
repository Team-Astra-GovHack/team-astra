import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    MetaData,
    Text,
)
from sqlalchemy.exc import SQLAlchemyError


# -------------------------------
# 1. Map data dictionary types → SQLAlchemy
# -------------------------------
def map_dtype(dtype: str):
    """Map data dictionary type to SQLAlchemy column type."""
    dtype = dtype.lower()
    if dtype in ["int", "integer"]:
        return Integer
    elif dtype in ["float", "double", "decimal"]:
        return Float
    elif dtype in ["string", "text", "varchar"]:
        return Text
    elif dtype in ["boolean", "bool"]:
        return Boolean
    elif dtype in ["date", "datetime", "timestamp"]:
        return DateTime
    else:
        return Text  # fallback


# -------------------------------
# 2. Safe table name generator
# -------------------------------
def safe_table_name(base: str, sheet: str = None, max_length: int = 63) -> str:
    """Generate a safe, readable Postgres table name."""
    # Clean base name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", base.lower())
    name = re.sub(r"_+", "_", name).strip("_")

    # Truncate base to 40 chars for readability
    name = name[:40]

    # Add sheet name if provided
    if sheet:
        sheet_clean = re.sub(r"[^a-zA-Z0-9_]", "_", sheet.lower())[:15]
        name = f"{name}_{sheet_clean}"

    # Ensure max length
    if len(name) > max_length:
        name = name[:max_length]

    return name


# -------------------------------
# 3. Safe column name generator
# -------------------------------
def safe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to be Postgres-friendly (snake_case, short)."""
    new_cols = []
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col).lower())
        clean = re.sub(r"_+", "_", clean).strip("_")
        clean = clean[:63]  # Postgres max identifier length
        if not clean:  # fallback if column name is empty
            clean = "unnamed_col"
        new_cols.append(clean)
    df.columns = new_cols
    return df


# -------------------------------
# 4. Smarter schema inference
# -------------------------------
def infer_sqlalchemy_type(series: pd.Series):
    """Infer SQLAlchemy column type from pandas Series with sample check."""
    dtype = str(series.dtype)

    if "int" in dtype:
        return Integer
    elif "float" in dtype:
        return Float
    elif "bool" in dtype:
        return Boolean
    elif "datetime" in dtype:
        return DateTime
    elif "object" in dtype:
        # Try to detect datetime-like strings
        sample = series.dropna().astype(str).head(20)
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.8:  # if >80% parse as dates
            return DateTime
        else:
            return Text
    else:
        return Text


def create_schema_from_dictionary_or_infer(
    df: pd.DataFrame, data_dictionary: dict, engine, table_name: str, schema: str = "public"
):
    """Create a PostgreSQL table schema from the data dictionary if available, otherwise infer from DataFrame dtypes."""
    metadata = MetaData(schema=schema)
    columns = []

    for col in df.columns:
        if data_dictionary and "columns" in data_dictionary and col in data_dictionary["columns"]:
            col_info = data_dictionary["columns"][col]
            col_type = map_dtype(col_info.get("data_type", "string"))
            is_pk = col_info.get("primary_key", False)
        else:
            col_type = infer_sqlalchemy_type(df[col])
            is_pk = False

        columns.append(Column(col, col_type, primary_key=is_pk))

    # ✅ Ensure at least one primary key, but don’t duplicate "id"
    if not any(c.primary_key for c in columns):
        if "id" not in df.columns:
            columns.insert(0, Column("id", Integer, primary_key=True, autoincrement=True))

    table = Table(table_name, metadata, *columns)
    try:
        metadata.create_all(engine)
    except SQLAlchemyError as e:
        print(f"⚠️ Failed to create table {table_name}: {e}")
    return table


# -------------------------------
# 5. Load one cleaned file into Postgres
# -------------------------------
def load_cleaned_data(
    cleaned_file: str, engine, table_name: str, data_dictionary: dict = None, schema: str = "public"
):
    """Load cleaned CSV/Excel into PostgreSQL with schema from data dictionary or inferred."""
    try:
        if cleaned_file.endswith(".csv"):
            df = pd.read_csv(cleaned_file)
        elif cleaned_file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(cleaned_file)
        else:
            print(f"⚠️ Skipping unsupported file: {cleaned_file}")
            return

        if df.empty:
            print(f"⚠️ Skipping empty file: {cleaned_file}")
            return

        # Clean column names
        df = safe_column_names(df)

        # Create schema
        create_schema_from_dictionary_or_infer(df, data_dictionary, engine, table_name, schema)

        # Insert data
        df.to_sql(table_name, engine, if_exists="append", index=False, schema=schema)
        print(f"✅ Loaded {len(df)} rows into {schema}.{table_name}")

    except Exception as e:
        print(f"❌ Error loading {cleaned_file} into {table_name}: {e}")


# -------------------------------
# 6. Batch load all cleaned files
# -------------------------------
def load_all_cleaned_to_postgres(
    cleaned_dir: str, db_url: str, data_dictionary: dict = None, schema: str = "public"
):
    engine = create_engine(db_url)

    for file in os.listdir(cleaned_dir):
        file_path = os.path.join(cleaned_dir, file)
        base_name = os.path.splitext(file)[0]

        if file.endswith(".csv"):
            table_name = safe_table_name(base_name)
            load_cleaned_data(file_path, engine, table_name, data_dictionary, schema)

        elif file.endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                table_name = safe_table_name(base_name, sheet_name)
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    if df.empty:
                        print(f"⚠️ Skipping empty sheet: {file} -> {sheet_name}")
                        continue
                    df = safe_column_names(df)

                    create_schema_from_dictionary_or_infer(df, data_dictionary, engine, table_name, schema)
                    df.to_sql(table_name, engine, if_exists="append", index=False, schema=schema)
                    print(f"✅ Loaded {len(df)} rows into {schema}.{table_name}")
                except Exception as e:
                    print(f"❌ Error loading sheet {sheet_name} from {file}: {e}")

        else:
            print(f"⚠️ Skipping unsupported file: {file}")


# -------------------------------
# 7. Run script
# -------------------------------
if __name__ == "__main__":
    cleaned_dir = "cleaned_data"
    db_url = "postgresql://neondb_owner:npg_h7utrgEnMxc4@ep-orange-snow-adai6w7v-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

    # Example data dictionary (optional, expand with your real one)
    data_dictionary = {
        "columns": {
            "id": {"data_type": "integer", "primary_key": True},
            "employee_name": {"data_type": "string"},
            "department": {"data_type": "string"},
            "position": {"data_type": "string"},
            "leave_type": {"data_type": "string"},
            "start_date": {"data_type": "datetime"},
            "end_date": {"data_type": "datetime"},
            "days_taken": {"data_type": "integer"},
            "total_leave_entitlement": {"data_type": "integer"},
            "leave_taken_so_far": {"data_type": "integer"},
            "remaining_leaves": {"data_type": "integer"},
            "month": {"data_type": "string"},
            "data_source": {"data_type": "string"},
            "last_updated": {"data_type": "datetime"},
        }
    }

    load_all_cleaned_to_postgres(cleaned_dir, db_url, data_dictionary, schema="public")