#%%
import pandas as pd
import numpy as np
import re

#%%

df = pd.read_excel('Datasets\Employee_Leave_Tracking_Data_by_manishkumar21324(Kaggle)\employee leave tracking data.xlsx')
df.info()
df.describe()

#%%
df['Department'].unique
#%%
# Config 
INPUT_FILE = "'Employee_Leave_Tracking_Data_by_manishkumar21324(Kaggle)\employee leave tracking data.xlsx'"
OUTPUT_FILE = "Employee_Leave_Tracking_Data_by_manishkumar21324(Kaggle)\leaves_cleaned.xlsx"
ISSUES_FILE = "Employee_Leave_Tracking_Data_by_manishkumar21324(Kaggle)\leaves_cleaning_issues.xlsx"

NAME_COL = "Employee Name"
DEPT_COL = "Department"
POS_COL = "Position"
LEAVE_TYPE_COL = "Leave Type"
START_COL = "Start Date"
END_COL = "End Date"
DAYS_TAKEN_COL = "Days Taken"
ENTITLEMENT_COL = "Total Leave Entitlement"
TAKEN_SO_FAR_COL = "Leave Taken So Far"
REMAINING_COL = "Remaining Leaves"
MONTH_COL = "Month"
DEFAULT_ANNUAL_ENTITLEMENT = 20

DEPT_MAP = {"hr": "Human Resources", "human resources": "Human Resources", "sales & marketing": "Sales"}
LEAVE_TYPE_MAP = {"sl": "Sick Leave", "sick": "Sick Leave", "vacation": "Annual Leave", "al": "Annual Leave"}
POSITION_MAP = {"sr. engineer": "Senior Engineer", "senior eng": "Senior Engineer", "engineer": "Engineer"}
#%%
# Helpers
def clean_name(s):
    if pd.isna(s): return s
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.title()

def map_norm(val, m):
    if pd.isna(val): return val
    return m.get(str(val).strip().lower(), str(val).strip())

def to_dt(series):
    return pd.to_datetime(series, errors="coerce")

def normalize_month(val):
    if pd.isna(val): return np.nan
    dt = pd.to_datetime(val, errors="coerce")
    return dt.strftime("%Y-%m") if not pd.isna(dt) else np.nan

# Load
df = pd.read_excel(INPUT_FILE, dtype=object)

issues = []

# 1 Name
if NAME_COL in df:
    df[NAME_COL] = df[NAME_COL].apply(clean_name)
    if df[NAME_COL].isna().any():
        issues.append(("Missing names", df[df[NAME_COL].isna()].index.tolist()))

# 2 Department
if DEPT_COL in df:
    df[DEPT_COL] = df[DEPT_COL].apply(lambda v: map_norm(v, DEPT_MAP))

# 3 Position
if POS_COL in df:
    df[POS_COL] = df[POS_COL].apply(lambda v: map_norm(v, POSITION_MAP).title())

# 4 Leave Type
if LEAVE_TYPE_COL in df:
    df[LEAVE_TYPE_COL] = df[LEAVE_TYPE_COL].apply(lambda v: map_norm(v, LEAVE_TYPE_MAP))

# 5 Dates
df[START_COL] = to_dt(df.get(START_COL))
df[END_COL] = to_dt(df.get(END_COL))
if df[START_COL].isna().any(): issues.append(("Invalid/Missing Start Date", df[df[START_COL].isna()].index.tolist()))
if df[END_COL].isna().any(): issues.append(("Invalid/Missing End Date", df[df[END_COL].isna()].index.tolist()))
bad_order = df[(df[START_COL].notna()) & (df[END_COL].notna()) & (df[END_COL] < df[START_COL])].index.tolist()
if bad_order: issues.append(("End before Start", bad_order))

# 6 Days Taken (compute inclusive)
df["_computed_days"] = np.where(df[START_COL].notna() & df[END_COL].notna(),
                                (df[END_COL] - df[START_COL]).dt.days + 1,
                                np.nan)
df[DAYS_TAKEN_COL] = pd.to_numeric(df.get(DAYS_TAKEN_COL), errors="coerce")
mismatch = df[df[DAYS_TAKEN_COL].notna() & df["_computed_days"].notna() & (df[DAYS_TAKEN_COL] != df["_computed_days"])].index.tolist()
if mismatch: issues.append(("Days mismatch", mismatch))
df[DAYS_TAKEN_COL] = df["_computed_days"].fillna(df[DAYS_TAKEN_COL])

# 7 Entitlement
df[ENTITLEMENT_COL] = pd.to_numeric(df.get(ENTITLEMENT_COL), errors="coerce").fillna(DEFAULT_ANNUAL_ENTITLEMENT)
neg_ent = df[df[ENTITLEMENT_COL] < 0].index.tolist()
if neg_ent: issues.append(("Negative entitlement", neg_ent))
df.loc[df[ENTITLEMENT_COL] < 0, ENTITLEMENT_COL] = DEFAULT_ANNUAL_ENTITLEMENT

# 8 Leave Taken So Far (recalculate per employee name)
df[TAKEN_SO_FAR_COL] = pd.to_numeric(df.get(TAKEN_SO_FAR_COL), errors="coerce")
if NAME_COL in df:
    recalc = df.groupby(NAME_COL)[DAYS_TAKEN_COL].transform("sum")
    diff = df[df[TAKEN_SO_FAR_COL].notna() & (df[TAKEN_SO_FAR_COL] != recalc)].index.tolist()
    if diff: issues.append(("Taken-so-far mismatch", diff))
    df[TAKEN_SO_FAR_COL] = recalc
else:
    df[TAKEN_SO_FAR_COL] = df[DAYS_TAKEN_COL]

# 9 Remaining
df[REMAINING_COL] = df[ENTITLEMENT_COL] - df[TAKEN_SO_FAR_COL]
neg_rem = df[df[REMAINING_COL] < 0].index.tolist()
if neg_rem: issues.append(("Negative remaining", neg_rem))

# 10 Month
df[MONTH_COL] = df.get(MONTH_COL).apply(normalize_month) if MONTH_COL in df else df[START_COL].dt.strftime("%Y-%m")

# Select and save cleaned columns
keep = [c for c in [NAME_COL, DEPT_COL, POS_COL, LEAVE_TYPE_COL, START_COL, END_COL, DAYS_TAKEN_COL,
                    ENTITLEMENT_COL, TAKEN_SO_FAR_COL, REMAINING_COL, MONTH_COL] if c in df.columns]
clean_df = df[keep].copy()
clean_df.to_excel(OUTPUT_FILE, index=False)

pd.DataFrame(issues, columns=["Issue", "IndicesOrSamples"]).to_excel(ISSUES_FILE, index=False)

print(f"Saved cleaned data to {OUTPUT_FILE} and issues to {ISSUES_FILE}")
