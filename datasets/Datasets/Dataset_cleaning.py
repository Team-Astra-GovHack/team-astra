#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
def tbl_action_cleaning():
    global df1
    # Get the absolute path of the current file
    file_path = os.path.abspath(__file__)

    # Get the directory name from the file path
    directory_path = os.path.dirname(file_path)

    print(f"The directory of this file is: {directory_path}")
    # Load the dataset

    # display basic information about the dataset
    df = pd.read_csv(r'HR_Data_Set_(Human_Resources_Data_Set)_by_davidepolizzi_(Kaggle)\tbl_Action.csv')
    df.info()
    df.describe()

    df["ActID"] = pd.to_numeric(df["ActID"], errors="coerce")
    df["ActionID"] = pd.to_numeric(df["ActionID"], errors="coerce")
    df["EmpID"] = pd.to_numeric(df["EmpID"], errors="coerce")
    df["EffectiveDt"] = pd.to_datetime(df["EffectiveDt"], errors="coerce")

    df["ActID"].unique()
    df["ActionID"].unique()
    df["EmpID"].unique()
    df["EffectiveDt"].unique()

    df1 = df.copy()
    # Now we are going to see the number of rows and columns in the dataset
    print(f"Number of rows: {df1.shape[0]}") 
    print(f"Number of columns: {df1.shape[1]}")
    df1.head()

    print(df1.isnull().sum())


    return df1
    
#%%
def tbl_employee_cleaning():
    global df2
    # Get the absolute path of the current file
    file_path = os.path.abspath(__file__)

    # Get the directory name from the file path
    directory_path = os.path.dirname(file_path)

    print(f"The directory of this file is: {directory_path}")
    # Load the dataset
    # display basic information about the dataset
    df = pd.read_csv(r'HR_Data_Set_(Human_Resources_Data_Set)_by_davidepolizzi_(Kaggle)\tbl_Employee.csv')
    df.info()
    df.describe()
    df["EmpName"].astype(str)
    df["EmpID"] = pd.to_numeric(df["EmpID"], errors="coerce")
    df["EngDt"] = pd.to_datetime(df["EngDt"], errors="coerce")
    df["TermDt"] = pd.to_datetime(df["TermDt"], errors="coerce")
    df["DepID"] = pd.to_numeric(df["DepID"], errors="coerce")
    df["GenderID"] = pd.to_numeric(df["GenderID"], errors="coerce")
    df["RaceID"] = pd.to_numeric(df["RaceID"], errors="coerce")
    df["MgrID"] = pd.to_numeric(df["MgrID"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["PayRate"] = pd.to_numeric(df["PayRate"], errors="coerce")
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    #fix the text columns
    df[['LastName', 'FirstName']] = df['EmpName'].str.split(',', expand=True)

    # Optional: Strip any leading/trailing whitespace from the new columns
    df['LastName'] = df['LastName'].str.strip()
    df['FirstName'] = df['FirstName'].str.strip()
    text_cols = ["LastName",'FirstName']
    for col in text_cols:
        df[col] = df[col].astype(str)  # Convert to string
        df[col] = df[col].str.replace(" ", "", regex=False)  # Remove all spaces
    # Remove the 'EmpName' column
    df = df.drop(columns=['EmpName'])
    df.loc[df['EngDt'].isnull() & df['TermDt'].isnull(), ['EngDt', 'TermDt']] = 'NotAvailable'
    # Replace missing values in the 'TermDt' column with 'Employed'
    df['TermDt'] = df['TermDt'].fillna('Employed')
    df['EngDt'] = df['EngDt'].fillna('Starting_Unknown')
    df['PayRate'] = df['PayRate'].fillna('Unknown')
    # df["ActID"].unique()
    # df["ActionID"].unique()
    # df["EmpID"].unique()
    # df["EffectiveDt"].unique()
    print(df.isnull().sum())
    print(df.info())
    
    df2 = df.copy()
    return df2
    
#%%
def tbl_perf_cleaning():
    # Get the absolute path of the current file
    file_path = os.path.abspath(__file__)

    # Get the directory name from the file path
    directory_path = os.path.dirname(file_path)

    print(f"The directory of this file is: {directory_path}")
    # Load the dataset
    # display basic information about the dataset
    df = pd.read_csv(r'HR_Data_Set_(Human_Resources_Data_Set)_by_davidepolizzi_(Kaggle)\tbl_Perf.csv')
    df.info()
    df.describe()
    df["EmpID"] = pd.to_numeric(df["EmpID"], errors="coerce")
    df["PerfID"] = pd.to_numeric(df["PerfID"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["PerfDate"] = pd.to_datetime(df["PerfDate"], errors="coerce")
    print(df.isnull().sum())
    print(df.info())

    df4 = df.copy()
    return df4
    
#%%
def Employee_cleaning():
    # Load the dataset

    # display basic information about the dataset
    df = pd.read_excel(r'Employee_Leave_Tracking_Data_by_manishkumar21324(Kaggle)\employee leave tracking data.xlsx')
    df.info()
    df.describe()

    # Now we are going to see the number of rows and columns in the dataset
    print(f"Number of rows: {df.shape[0]}") 
    print(f"Number of columns: {df.shape[1]}")
    df.head()


    # convert and verify data types

    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
    df["Days Taken"] = pd.to_numeric(df["Days Taken"], errors="coerce")
    df["Total Leave Entitlement"] = pd.to_numeric(df["Total Leave Entitlement"], errors="coerce")
    df["Leave Taken So Far"] = pd.to_numeric(df["Leave Taken So Far"], errors="coerce")
    df["Remaining Leaves"] = pd.to_numeric(df["Remaining Leaves"], errors="coerce")




    #fix the text columns
    text_cols = ["Employee Name", "Department", "Position", "Leave Type", "month"]
    for col in text_cols:
        df[col] = df[col].astype(str)  # Convert to string
        df[col] = df[col].str.replace(" ", "", regex=False)  # Remove all spaces
        


    df['Employee Name'].unique()
    df['Department'].unique()
    df['Position'].unique()
    df['Leave Type'].unique()
    df['month'].unique()



    df3 = df.copy()
    # Now we are going to see the number of rows and columns in the dataset
    print(f"Number of rows: {df3.shape[0]}") 
    print(f"Number of columns: {df3.shape[1]}")
    df3.head()
    print(df3.isnull().sum())

    return df3

#%%
def hr_cleaning():
    # Get the absolute path of the current file
    file_path = os.path.abspath(__file__)

    # Get the directory name from the file path
    directory_path = os.path.dirname(file_path)

    print(f"The directory of this file is: {directory_path}")
    # Load the dataset
    # display basic information about the dataset
    df1 = pd.read_csv(r'HR_Data_Set_(Human_Resources_Data_Set)_by_davidepolizzi_(Kaggle)\HR_DATA.csv')
    df = df1
    df.info()
    df.describe()
    print(df.isnull().sum())

    # Splitting employee name into first and last name
    df[['LastName_Employee', 'FirstName_Employee']] = df['Employee_Name'].str.split(',', expand=True)
    df['LastName_Employee'] = df['LastName_Employee'].str.strip()
    df['FirstName_Employee'] = df['FirstName_Employee'].str.strip()
    df = df.drop(columns=['Employee_Name'])
    df['ManagerName'] = df['ManagerName'].str.replace(' ', '', regex=False)
    df['RecruitmentSource'] = df['RecruitmentSource'].str.replace(' ', '', regex=False)
    df['PerformanceScore'] = df['PerformanceScore'].str.replace(' ', '', regex=False)
    df['Position'] = df['Position'].str.replace(' ', '', regex=False)
    df['CitizenDesc'] = df['CitizenDesc'].str.replace(' ', '', regex=False)
    df['RaceDesc'] = df['RaceDesc'].str.replace(' ', '', regex=False)
    df['TermReason'] = df['TermReason'].str.replace(' ', '', regex=False)
    df['EmploymentStatus'] = df['EmploymentStatus'].str.replace(' ', '', regex=False)
    df['RecruitmentSource'] = df['RecruitmentSource'].str.replace(' ', '', regex=False)
    text_cols = ['Position','State','Sex','MaritalDesc','CitizenDesc','HispanicLatino','RaceDesc','TermReason','EmploymentStatus','Department','ManagerName','RecruitmentSource','PerformanceScore','Original DS','LastName_Employee','FirstName_Employee']
    for col in text_cols:
        df[col] = df[col].astype(str)  # Convert to string
    #convert to numeric
    num_cols = ['EmpID','MarriedID','MaritalStatusID','GenderID','EmpStatusID','DeptID','PerfScoreID','FromDiversityJobFairID','PayRate','Termd','PositionID','Zip','ManagerID','EngagementSurvey','EmpSatisfaction','DaysLateLast30',]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to string    

    #convert to date
    date_cols = ['DOB','DateofHire','DateofTermination','LastPerformanceReview_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # Convert to string    

    #Nonzero null value
    non_zero_nulls = df.isnull().sum()
    non_zero_nulls = non_zero_nulls[non_zero_nulls > 0]

    # Display the columns with non-zero null counts
    print(non_zero_nulls)
    print(df.info())

    df['Zip'] = df['Zip'].fillna('Unknown')
    df['DateofHire'] = df['DateofHire'].fillna('Unknown')
    df['DateofTermination'] = df['DateofTermination'].fillna('Unknown')
    df['ManagerID'] = df['ManagerID'].fillna('Unknown')
    df['LastPerformanceReview_Date'] = df['LastPerformanceReview_Date'].fillna('Unknown')
    df['DaysLateLast30'] = df['DaysLateLast30'].fillna('Unknown')
    print(df.isnull().sum())
    df5 = df.copy()
    
    return df5

#%%
cleaned_tbl_action = tbl_action_cleaning()
#%%
cleaned_tbl_employee = tbl_employee_cleaning()
#%%
cleaned_tbl_perf = tbl_perf_cleaning()
# %%
cleaned_Employee = Employee_cleaning()
#%%
cleaned_hr = hr_cleaning()

# %%
cleaned_tbl_action.to_csv(r'cleaned_datasets/tbl_Action_cleaned.csv', index=False, date_format='%Y-%m-%d %H:%M:%S') 
#%%
cleaned_tbl_employee.to_csv(r'cleaned_datasets/tbl_Employee_cleaned.csv', index=False, date_format='%Y-%m-%d %H:%M:%S') 
cleaned_tbl_perf.to_csv(r'cleaned_datasets\tbl_Perf_cleaned.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
#%%
cleaned_Employee.to_csv(r'cleaned_datasets\employee_leave_tracking_data_cleaned.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
# %%
cleaned_hr.to_csv(r'cleaned_datasets\HR_DATA_cleaned.csv', index=False,date_format='%Y-%m-%d %H:%M:%S')
# %%
