import logging
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd

# NEW: Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging for this script
log = logging.getLogger(__name__)

# IMPORTANT: The run_data_processing function now accepts 'data_root_dir'
# which allows it to correctly locate data files and save preprocessed outputs.


# --- Helper Function for Cleaning ---
def clean_df(df_input, target_col=None, column_map=None, source_name="dataset"):
    """
    General function to clean DataFrame:
    - Strip spaces from column names.
    - Replace inf/-inf with NaN.
    - Rename columns if a map is provided.
    - Drop constant columns (excluding target).
    - Decode object columns (for ARFF files).
    """
    df = df_input.copy()

    # Decode object columns (common for ARFF files loaded via scipy.io.arff)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Check if it's actually bytes and decode
                if isinstance(df[col].iloc[0], bytes):
                    df[col] = df[col].str.decode("utf-8")
            except IndexError:  # Handle empty dataframe case
                pass
            except AttributeError:
                pass  # Not a string column to decode

    # Apply column renaming and strip spaces from new column names
    if column_map:
        df.rename(columns=column_map, inplace=True)
    df.columns = [col.strip() for col in df.columns]

    log.info(f"  {source_name} initial shape: {df.shape}")

    # Handle infinites by replacing with NaN
    infinite_cols = {}
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            # Using .isin() is robust for float comparisons with inf
            inf_count = df[df[col].isin([np.inf, -np.inf])].shape[0]
            if inf_count > 0:
                infinite_cols[col] = inf_count

    if infinite_cols:
        log.info(f"  Infinite values found in {source_name} (replacing with NaN):")
        for col, count in infinite_cols.items():
            log.info(f"    - {col}: {count}")
        for col in infinite_cols.keys():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    else:
        log.info(f"  No infinite values found in {source_name}.")

    # Handle constant columns (excluding the target if specified)
    cols_to_check = [col for col in df.columns if col != target_col]
    constant_cols = [col for col in cols_to_check if df[col].nunique() == 1]
    if constant_cols:
        log.info(f"  Constant columns found in {source_name} (dropping):")
        for col in constant_cols:
            log.info(f"    - {col} (Value: {df[col].iloc[0]})")
        df.drop(columns=constant_cols, inplace=True)
    else:
        log.info(f"  No constant columns found in {source_name}.")

    log.info(f"  {source_name} cleaned shape: {df.shape}")
    return df


def run_data_processing(data_root_dir):
    """
    Executes the entire data consolidation, preprocessing, and balancing pipeline.

    Args:
        data_root_dir (str): Absolute path to the directory where raw data is
                             located and where preprocessed data will be saved.
                             This directory is expected to have 'company-bankruptcy-prediction-taiwan.csv',
                             '1year.arff', '2year.arff', etc., and will contain
                             'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv'
                             and 'models/global_scaler.pkl' after execution.
    """
    log.info("--- Starting Multi-Dataset Consolidation and Preprocessing ---")

    # --- 0. Ensure Directories Exist ---
    # These directories will now be relative to data_root_dir
    MODELS_DIR = os.path.join(data_root_dir, "models")
    PLOTS_DIR = os.path.join(data_root_dir, "plots")

    os.makedirs(data_root_dir, exist_ok=True)  # Ensure data_root_dir itself exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    log.info(
        f"Ensured data directories exist: {data_root_dir}, {MODELS_DIR}, {PLOTS_DIR}"
    )

    # --- Step 1: Load and Process Taiwan Dataset (Original Bankrupt.csv) ---
    log.info("\n--- Processing Taiwan Dataset (Bankrupt.csv) ---")

    # Column renaming map for Taiwan Dataset based on provided descriptions
    taiwan_column_map = {
        "Bankrupt?": "Target_Bankruptcy",
        " ROA(C) before interest and depreciation before interest": "ROA_C_BeforeInterestDepreciation",
        " ROA(A) before interest and % after tax": "ROA_A_BeforeInterestAfterTax",
        " ROA(B) before interest and depreciation after tax": "ROA_B_BeforeInterestDepreciationAfterTax",
        " Operating Gross Margin": "Operating_Gross_Margin",
        " Realized Sales Gross Margin": "Realized_Sales_Gross_Margin",
        " Operating Profit Rate": "Operating_Profit_Rate",
        " Pre-tax net Interest Rate": "PreTax_Net_Interest_Rate",
        " After-tax net Interest Rate": "AfterTax_Net_Interest_Rate",
        " Non-industry income and expenditure/revenue": "NonIndustry_Income_Expenditure_Revenue",
        " Continuous interest rate (after tax)": "Continuous_Interest_Rate_AfterTax",
        " Operating Expense Rate": "Operating_Expense_Rate",
        " Research and development expense rate": "R&D_Expense_Rate",
        " Cash flow rate": "Cash_Flow_Rate",
        " Interest-bearing debt interest rate": "Interest_Bearing_Debt_Interest_Rate",
        " Tax rate (A)": "Tax_Rate_A",
        " Net Value Per Share (B)": "Net_Value_Per_Share_B",
        " Net Value Per Share (A)": "Net_Value_Per_Share_A",
        " Net Value Per Share (C)": "Net_Value_Per_Share_C",
        " Persistent EPS in the Last Four Seasons": "Persistent_EPS_Last_4_Seasons",
        " Cash Flow Per Share": "Cash_Flow_Per_Share",
        " Revenue Per Share (Yuan ¥)": "Revenue_Per_Share",
        " Operating Profit Per Share (Yuan ¥)": "Operating_Profit_Per_Share",
        " Per Share Net profit before tax (Yuan ¥)": "Pretax_Net_Profit_Per_Share",
        " Realized Sales Gross Profit Growth Rate": "Realized_Sales_Gross_Profit_Growth_Rate",
        " Operating Profit Growth Rate": "Operating_Profit_Growth_Rate",
        " After-tax Net Profit Growth Rate": "AfterTax_Net_Profit_Growth_Rate",
        " Regular Net Profit Growth Rate": "Regular_Net_Profit_Growth_Rate",
        " Continuous Net Profit Growth Rate": "Continuous_Net_Profit_Growth_Rate",
        " Total Asset Growth Rate": "Total_Asset_Growth_Rate",
        " Net Value Growth Rate": "Net_Value_Growth_Rate",
        " Total Asset Return Growth Rate Ratio": "Total_Asset_Return_Growth_Rate_Ratio",
        " Cash Reinvestment %": "Cash_Reinvestment_Percentage",
        " Current Ratio": "Current_Ratio",
        " Quick Ratio": "Quick_Ratio_Acid_Test",  # Renamed to match the description (X34: Quick Ratio: Acid Test)
        " Interest Expense Ratio": "Interest_Expense_Ratio",
        " Total debt/Total net worth": "Total_Debt_Total_Net_Worth_Ratio",
        " Debt ratio %": "Debt_Ratio_Percentage",
        " Net worth/Assets": "Net_Worth_Assets_Ratio",
        " Long-term fund suitability ratio (A)": "LongTerm_Fund_Suitability_Ratio_A",
        " Borrowing dependency": "Borrowing_Dependency",
        " Contingent liabilities/Net worth": "Contingent_Liabilities_Net_Worth",
        " Operating profit/Paid-in capital": "Operating_Profit_Paid_in_Capital",
        " Net profit before tax/Paid-in capital": "Net_Profit_BeforeTax_Paid_in_Capital",
        " Inventory and accounts receivable/Net value": "Inventory_Accounts_Receivable_Net_Value",
        " Total Asset Turnover": "Total_Asset_Turnover",
        " Accounts Receivable Turnover": "Accounts_Receivable_Turnover",
        " Average Collection Days": "Average_Collection_Days",
        " Inventory Turnover Rate (times)": "Inventory_Turnover_Rate",
        " Fixed Assets Turnover Frequency": "Fixed_Assets_Turnover_Frequency",
        " Net Worth Turnover Rate (times)": "Net_Worth_Turnover_Rate",
        " Revenue per person": "Revenue_Per_Person",
        " Operating profit per person": "Operating_Profit_Per_Person",
        " Allocation rate per person": "Allocation_Rate_Per_Person",
        " Working Capital to Total Assets": "Working_Capital_to_Total_Assets",
        " Quick Assets/Total Assets": "Quick_Assets_Total_Assets",
        " Current Assets/Total Assets": "Current_Assets_Total_Assets",
        " Cash/Total Assets": "Cash_Total_Assets",
        " Quick Assets/Current Liability": "Quick_Assets_Current_Liability",
        " Cash/Current Liability": "Cash_Current_Liability",
        " Current Liability to Assets": "Current_Liability_to_Assets",
        " Operating Funds to Liability": "Operating_Funds_to_Liability",
        " Inventory/Working Capital": "Inventory_Working_Capital",
        " Inventory/Current Liability": "Inventory_Current_Liability",
        " Current Liabilities/Liability": "Current_Liabilities_Liability",
        " Working Capital/Equity": "Working_Capital_Equity",
        " Current Liabilities/Equity": "Current_Liabilities_Equity",
        " Long-term Liability to Current Assets": "LongTerm_Liability_to_Current_Assets",
        " Retained Earnings to Total Assets": "Retained_Earnings_to_Total_Assets",
        " Total income/Total expense": "Total_Income_Total_Expense",
        " Total expense/Assets": "Total_Expense_Assets",
        " Current Asset Turnover Rate": "Current_Asset_Turnover_Rate",
        " Quick Asset Turnover Rate": "Quick_Asset_Turnover_Rate",
        " Working capitcal Turnover Rate": "Working_Capital_Turnover_Rate",
        " Cash Turnover Rate": "Cash_Turnover_Rate",
        " Cash Flow to Sales": "Cash_Flow_to_Sales",
        " Fixed Assets to Assets": "Fixed_Assets_to_Assets",
        " Current Liability to Liability": "Current_Liability_to_Liability",
        " Current Liability to Equity": "Current_Liability_to_Equity",
        " Equity to Long-term Liability": "Equity_to_LongTerm_Liability",
        " Cash Flow to Total Assets": "Cash_Flow_to_Total_Assets",
        " Cash Flow to Liability": "Cash_Flow_to_Liability",
        " CFO to Assets": "CFO_to_Assets",
        " Cash Flow to Equity": "Cash_Flow_to_Equity",
        " Current Liability to Current Assets": "Current_Liability_to_Current_Assets",
        " Liability-Assets Flag": "Liability_Assets_Flag",
        " Net Income to Total Assets": "Net_Income_to_Total_Assets",
        " Total assets to GNP price": "Total_Assets_to_GNP_Price",
        " No-credit Interval": "No_Credit_Interval",
        " Gross Profit to Sales": "Gross_Profit_to_Sales",
        " Net Income to Stockholder's Equity": "Net_Income_to_Stockholders_Equity",
        " Liability to Equity": "Liability_to_Equity",
        " Degree of Financial Leverage (DFL)": "Degree_of_Financial_Leverage",
        " Interest Coverage Ratio (Interest expense to EBIT)": "Interest_Coverage_Ratio",
        " Net Income Flag": "Net_Income_Flag",
        " Equity to Liability": "Equity_to_Liability",
    }

    df_taiwan = pd.DataFrame()  # Initialize as empty DataFrame
    taiwan_file_path = os.path.join(
        data_root_dir, "company-bankruptcy-prediction-dataset-taiwan.csv"
    )
    try:
        df_taiwan = pd.read_csv(taiwan_file_path)
        df_taiwan = clean_df(
            df_taiwan,
            target_col="Bankrupt?",
            column_map=taiwan_column_map,
            source_name="Taiwan dataset",
        )
        # Ensure target name is final after clean_df has applied the map
        df_taiwan.rename(
            columns={"Bankrupt?": "Target_Bankruptcy"}, inplace=True, errors="ignore"
        )
        log.info(f"Taiwan dataset (Bankrupt.csv) processed. Shape: {df_taiwan.shape}")
    except FileNotFoundError:
        log.warning(f"Error: '{taiwan_file_path}' not found. Skipping Taiwan dataset.")
    except Exception as e:
        log.error(
            f"An unexpected error occurred during processing {taiwan_file_path}: {e}. Skipping Taiwan dataset."
        )

    # --- Step 2: Load and Process Polish Datasets (Multiple ARFF files) ---
    log.info("\n--- Processing Polish Datasets (Multiple ARFF files) ---")

    polish_arff_files = [
        "1year.arff",
        "2year.arff",
        "3year.arff",
        "4year.arff",
        "5year.arff",
    ]
    all_polish_dfs = []

    # Polish Dataset Column Descriptions (from user input)
    # Map 'AttrX' (typical ARFF naming) to descriptive names.
    polish_attr_to_desc_map = {
        "Attr1": "Net_Profit_Total_Assets",
        "Attr2": "Total_Liabilities_Total_Assets",
        "Attr3": "Working_Capital_Total_Assets",
        "Attr4": "Current_Assets_Short_Term_Liabilities",
        "Attr5": "Cash_STSec_Receivables_STL_OperatingExpenses_Depreciation_365",
        "Attr6": "Retained_Earnings_Total_Assets",
        "Attr7": "EBIT_Total_Assets",
        "Attr8": "Book_Value_Equity_Total_Liabilities",
        "Attr9": "Sales_Total_Assets",
        "Attr10": "Equity_Total_Assets",
        "Attr11": "GrossProfit_ExtraordinaryItems_FinancialExpenses_TotalAssets",
        "Attr12": "Gross_Profit_Short_Term_Liabilities",
        "Attr13": "GrossProfit_Depreciation_Sales",
        "Attr14": "GrossProfit_Interest_TotalAssets",
        "Attr15": "TotalLiabilities_365_GrossProfit_Depreciation",
        "Attr16": "GrossProfit_Depreciation_TotalLiabilities",
        "Attr17": "Total_Assets_Total_Liabilities",
        "Attr18": "Gross_Profit_Total_Assets",
        "Attr19": "Gross_Profit_Sales",
        "Attr20": "Inventory_365_Sales",
        "Attr21": "Sales_Growth_Rate",
        "Attr22": "Profit_Operating_Activities_Total_Assets",
        "Attr23": "Net_Profit_Sales",
        "Attr24": "Gross_Profit_3Years_Total_Assets",
        "Attr25": "Equity_ShareCapital_Total_Assets",
        "Attr26": "NetProfit_Depreciation_TotalLiabilities",
        "Attr27": "Profit_Operating_Activities_Financial_Expenses",
        "Attr28": "Working_Capital_Fixed_Assets",
        "Attr29": "Logarithm_Total_Assets",
        "Attr30": "Total_Liabilities_Cash_Sales",
        "Attr31": "GrossProfit_Interest_Sales",
        "Attr32": "CurrentLiabilities_365_CostOfProductsSold",
        "Attr33": "Operating_Expenses_Short_Term_Liabilities",
        "Attr34": "Operating_Expenses_Total_Liabilities",
        "Attr35": "Profit_On_Sales_Total_Assets",
        "Attr36": "Total_Sales_Total_Assets",
        "Attr37": "CurrentAssets_Inventories_LongTermLiabilities",
        "Attr38": "Constant_Capital_Total_Assets",
        "Attr39": "Profit_On_Sales_Sales",
        "Attr40": "CurrentAssets_Inventory_Receivables_ShortTermLiabilities",
        "Attr41": "TotalLiabilities_OperatingProfit_Depreciation_12_365",
        "Attr42": "Profit_Operating_Activities_Sales",
        "Attr43": "Rotation_Receivables_Inventory_Turnover_Days",
        "Attr44": "Receivables_365_Sales",
        "Attr45": "Net_Profit_Inventory",
        "Attr46": "CurrentAssets_Inventory_ShortTermLiabilities",
        "Attr47": "Inventory_365_CostOfProductsSold",
        "Attr48": "EBITDA_ProfitOperatingActivities_Depreciation_TotalAssets",
        "Attr49": "EBITDA_ProfitOperatingActivities_Depreciation_Sales",
        "Attr50": "Current_Assets_Total_Liabilities",
        "Attr51": "Short_Term_Liabilities_Total_Assets",
        "Attr52": "ShortTermLiabilities_365_CostOfProductsSold",
        "Attr53": "Equity_Fixed_Assets",
        "Attr54": "Constant_Capital_Fixed_Assets",
        "Attr55": "Working_Capital",
        "Attr56": "Sales_CostOfProductsSold_Sales",
        "Attr57": "CurrentAssets_Inventory_STL_Sales_GrossProfit_Depreciation",
        "Attr58": "Total_Costs_Total_Sales",
        "Attr59": "Long_Term_Liabilities_Equity",
        "Attr60": "Sales_Inventory",
        "Attr61": "Sales_Receivables",
        "Attr62": "ShortTermLiabilities_365_Sales",
        "Attr63": "Sales_Short_Term_Liabilities",
        "Attr64": "Sales_Fixed_Assets",
        "class": "Target_Bankruptcy",  # Target column
    }

    df_polish = pd.DataFrame()  # Initialize as empty DataFrame
    for file_name in polish_arff_files:
        file_path = os.path.join(data_root_dir, file_name)
        log.info(f"  Loading {file_name}...")
        try:
            data, meta = arff.loadarff(file_path)
            df_temp_polish = pd.DataFrame(data)

            df_temp_polish = clean_df(
                df_temp_polish,
                target_col="class",
                column_map=polish_attr_to_desc_map,
                source_name=file_name,
            )

            # Confirm target mapping: typically 0 (non-bankrupt) and 1 (bankrupt)
            # Ensure it's numeric 0/1.
            if "Target_Bankruptcy" in df_temp_polish.columns:
                if (
                    df_temp_polish["Target_Bankruptcy"].dtype == "object"
                ):  # If loaded as string '0', '1'
                    df_temp_polish["Target_Bankruptcy"] = df_temp_polish[
                        "Target_Bankruptcy"
                    ].astype(int)

                if not df_temp_polish["Target_Bankruptcy"].isin([0, 1]).all():
                    log.warning(
                        f"  Warning: {file_name} 'Target_Bankruptcy' has values other than 0 or 1. Please verify and map manually if needed."
                    )
            else:
                log.warning(
                    f"  Warning: 'Target_Bankruptcy' column not found in {file_name} after renaming."
                )

            all_polish_dfs.append(df_temp_polish)
            log.info(f"  {file_name} loaded and processed.")

        except FileNotFoundError:
            log.warning(f"  Error: '{file_path}' not found. Skipping this file.")
        except Exception as e:
            log.error(
                f"  An unexpected error occurred during processing {file_name}: {e}. Skipping this file."
            )

    if all_polish_dfs:
        df_polish = pd.concat(all_polish_dfs, ignore_index=True)
        log.info(
            f"All Polish datasets combined. Final Polish DataFrame shape: {df_polish.shape}"
        )
    else:
        log.warning(
            "No Polish datasets were loaded. Creating empty DF for Polish data."
        )
        df_polish = pd.DataFrame()

    # --- Step 3: Load and Process "us" Dataset (Raw Financials) ---
    log.info("\n--- Processing 'us' Dataset (Raw Financials) ---")

    us_file_path = os.path.join(
        data_root_dir, "company-bankruptcy-prediction-dataset-us.csv"
    )
    df_us_derived = pd.DataFrame()  # Initialize as empty DataFrame

    # Column mapping for raw features in 'us' dataset for internal use
    us_raw_column_map = {
        "X1": "Current_Assets_Raw",
        "X2": "Cost_of_Goods_Sold_Raw",
        "X3": "Depreciation_and_Amortization_Raw",
        "X4": "EBITDA_Raw",
        "X5": "Inventory_Raw",
        "X6": "Net_Income_Raw",
        "X7": "Total_Receivables_Raw",
        "X8": "Market_Value_Raw",
        "X9": "Net_Sales_Raw",
        "X10": "Total_Assets_Raw",
        "X11": "Total_Long_Term_Debt_Raw",
        "X12": "EBIT_Raw",
        "X13": "Gross_Profit_Raw",
        "X14": "Total_Current_Liabilities_Raw",
        "X15": "Retained_Earnings_Raw",
        "X16": "Total_Revenue_Raw",
        "X17": "Total_Liabilities_Raw",
        "X18": "Total_Operating_Expenses_Raw",
        "status_label": "Target_Bankruptcy",
        "year": "Year",
    }

    try:
        df_us = pd.read_csv(us_file_path)
        df_us = clean_df(
            df_us,
            target_col="status_label",
            column_map=us_raw_column_map,
            source_name="US dataset",
        )

        # Map 'status_label' to numeric (0 for Alive, 1 for Failed/Bankrupt)
        if "Target_Bankruptcy" in df_us.columns:
            df_us["Target_Bankruptcy"] = df_us["Target_Bankruptcy"].map(
                {"alive": 0, "failed": 1}
            )
            if not df_us["Target_Bankruptcy"].isin([0, 1]).all():
                raise ValueError(
                    "Target_Bankruptcy mapping failed: Contains values other than 0 or 1."
                )
            log.info("  'status_label' mapped to 'Target_Bankruptcy' (0/1).")
        else:
            log.warning(
                "  Warning: 'Target_Bankruptcy' column not found in US data after renaming."
            )

        # --- Feature Engineering: Derive Ratios from Raw Financials ---
        log.info("  Deriving financial ratios from raw 'us' data...")
        df_us_derived = pd.DataFrame(index=df_us.index)
        if "Target_Bankruptcy" in df_us.columns:
            df_us_derived["Target_Bankruptcy"] = df_us["Target_Bankruptcy"]
        if "Year" in df_us.columns:
            df_us_derived["Year_US_Data"] = df_us[
                "Year"
            ]  # Rename to avoid conflict if 'Year' also in other datasets

        # Using .replace(0, np.nan) to prevent division by zero
        # Profitability Ratios
        df_us_derived["Net_Income_Total_Assets"] = df_us["Net_Income_Raw"] / df_us[
            "Total_Assets_Raw"
        ].replace(0, np.nan)
        df_us_derived["Gross_Profit_Sales"] = df_us["Gross_Profit_Raw"] / df_us[
            "Net_Sales_Raw"
        ].replace(0, np.nan)
        df_us_derived["Operating_Profit_Rate"] = df_us["EBIT_Raw"] / df_us[
            "Net_Sales_Raw"
        ].replace(0, np.nan)
        df_us_derived["EBITDA_Sales"] = df_us["EBITDA_Raw"] / df_us[
            "Net_Sales_Raw"
        ].replace(0, np.nan)
        df_us_derived["Return_on_Assets_Net_Income"] = df_us["Net_Income_Raw"] / df_us[
            "Total_Assets_Raw"
        ].replace(0, np.nan)

        # Liquidity Ratios
        df_us_derived["Current_Ratio"] = df_us["Current_Assets_Raw"] / df_us[
            "Total_Current_Liabilities_Raw"
        ].replace(0, np.nan)
        df_us_derived["Quick_Ratio_Acid_Test"] = (
            df_us["Current_Assets_Raw"] - df_us["Inventory_Raw"]
        ) / df_us["Total_Current_Liabilities_Raw"].replace(0, np.nan)
        df_us_derived["Working_Capital_to_Total_Assets"] = (
            df_us["Current_Assets_Raw"] - df_us["Total_Current_Liabilities_Raw"]
        ) / df_us["Total_Assets_Raw"].replace(0, np.nan)

        # Solvency/Leverage Ratios
        df_us_derived["Debt_Ratio_Percentage"] = df_us["Total_Liabilities_Raw"] / df_us[
            "Total_Assets_Raw"
        ].replace(0, np.nan)
        df_us_derived["Total_Debt_Total_Net_Worth_Ratio"] = df_us[
            "Total_Liabilities_Raw"
        ] / (df_us["Total_Assets_Raw"] - df_us["Total_Liabilities_Raw"]).replace(
            0, np.nan
        )
        df_us_derived["Long_Term_Debt_to_Equity"] = df_us[
            "Total_Long_Term_Debt_Raw"
        ] / (df_us["Total_Assets_Raw"] - df_us["Total_Liabilities_Raw"]).replace(
            0, np.nan
        )
        df_us_derived["Equity_to_Total_Assets"] = (
            df_us["Total_Assets_Raw"] - df_us["Total_Liabilities_Raw"]
        ) / df_us["Total_Assets_Raw"].replace(0, np.nan)

        # Efficiency/Activity Ratios
        df_us_derived["Total_Asset_Turnover"] = df_us["Net_Sales_Raw"] / df_us[
            "Total_Assets_Raw"
        ].replace(0, np.nan)
        df_us_derived["Inventory_Turnover_Rate"] = df_us[
            "Cost_of_Goods_Sold_Raw"
        ] / df_us["Inventory_Raw"].replace(0, np.nan)
        df_us_derived["Accounts_Receivable_Turnover"] = df_us["Net_Sales_Raw"] / df_us[
            "Total_Receivables_Raw"
        ].replace(0, np.nan)

        # Other potentially useful
        df_us_derived["Retained_Earnings_to_Total_Assets"] = df_us[
            "Retained_Earnings_Raw"
        ] / df_us["Total_Assets_Raw"].replace(0, np.nan)
        df_us_derived["Fixed_Assets_to_Total_Assets"] = (
            df_us["Total_Assets_Raw"] - df_us["Current_Assets_Raw"]
        ) / df_us["Total_Assets_Raw"].replace(0, np.nan)

        # Handle any infinites/NaNs resulting from division by zero after derivations
        df_us_derived = df_us_derived.replace([np.inf, -np.inf], np.nan)
        log.info(f"  Derived 'us' dataset shape: {df_us_derived.shape}")

    except FileNotFoundError:
        log.warning(f"Error: '{us_file_path}' not found. Skipping US dataset.")
    except Exception as e:
        log.error(
            f"An unexpected error occurred during processing 'us' dataset: {e}. Skipping US dataset."
        )

    # --- Step 4: Feature Harmonization and Merging All Datasets ---
    log.info("\n--- Harmonizing features and merging all datasets ---")

    # Combine all processed dataframes into a list
    all_dfs = [df_taiwan, df_polish, df_us_derived]

    # Filter out any empty DataFrames if files were not found or errors occurred
    all_dfs = [df for df in all_dfs if not df.empty]

    if not all_dfs:
        log.error("No valid dataframes to merge. Exiting data processing.")
        raise ValueError(
            "No valid dataframes found to process."
        )  # Raise an error to fail the Airflow task

    # Get all unique column names (excluding the target) from all dataframes
    all_feature_cols = []
    for df in all_dfs:
        all_feature_cols.extend(
            [col for col in df.columns if col != "Target_Bankruptcy"]
        )

    master_feature_cols = sorted(list(set(all_feature_cols)))
    log.info(
        f"Total unique features identified across all datasets: {len(master_feature_cols)}"
    )

    # Prepare dataframes for concatenation: add missing columns as NaN
    harmonized_dfs = []
    for df in all_dfs:
        df_temp = df.copy()
        current_cols = set(df_temp.columns)
        for col in master_feature_cols:
            if col not in current_cols:
                df_temp[col] = np.nan  # Add missing columns filled with NaN

        # Ensure all have the target column. If not present (e.g., from an error), fill with NaN
        if "Target_Bankruptcy" not in df_temp.columns:
            df_temp["Target_Bankruptcy"] = np.nan

        harmonized_dfs.append(
            df_temp[master_feature_cols + ["Target_Bankruptcy"]]
        )  # Reorder for consistency

    # Concatenate all harmonized dataframes
    merged_df = pd.concat(harmonized_dfs, ignore_index=True)

    log.info(f"All datasets merged. Final raw merged shape: {merged_df.shape}")
    log.info(
        f"Number of NaN values before final imputation: {merged_df.isnull().sum().sum()}"
    )

    # --- Step 5: Final Global Preprocessing on the Grand Merged Dataset ---
    log.info(
        "\n--- Performing final global preprocessing (Imputation, Scaling, Train-Test Split) ---"
    )

    # Separate features (X) and target (y)
    X_merged = merged_df.drop("Target_Bankruptcy", axis=1)
    y_merged = merged_df["Target_Bankruptcy"]

    # Impute Missing Values (NaNs)
    log.info("  Imputing missing values with median...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_merged), columns=X_merged.columns, index=X_merged.index
    )
    log.info(f"  NaN values after imputation: {X_imputed.isnull().sum().sum()}")

    # Save the imputer
    imputer_path = os.path.join(MODELS_DIR, "global_imputer.pkl")
    joblib.dump(imputer, imputer_path)
    log.info(f"Imputer saved to: {imputer_path}")

    # Stratified Train-Test Split (on the grand merged data)
    log.info("\n  Performing stratified train-test split on merged data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_merged, test_size=0.2, random_state=42, stratify=y_merged
    )
    log.info(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    log.info(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    log.info(f"  y_train distribution BEFORE SMOTE: {Counter(y_train)}")
    log.info(f"  y_test distribution: {Counter(y_test)}")

    # Scale Features
    log.info("\n  Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for consistency (optional, Keras accepts numpy arrays)
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )
    log.info("  Features scaled.")

    # Save the scaler
    scaler_path = os.path.join(MODELS_DIR, "global_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved to: {scaler_path}")

    # Save the list of feature names in the correct order
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    log.info(f"Feature names saved to: {feature_names_path}")

    ### Class Imbalance Handling with SMOTE
    log.info("\n--- Checking for Class Imbalance and Applying SMOTE ---")

    log.info("Class distribution in y_train BEFORE SMOTE: %s", Counter(y_train))

    # Initialize SMOTE
    smote = SMOTE(random_state=42)

    # Apply SMOTE to the training data
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled_df, y_train)

    log.info("Class distribution in y_train AFTER SMOTE: %s", Counter(y_train_smote))

    log.info(
        f"X_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}"
    )
    log.info("SMOTE applied successfully to the training data.")

    # Recalculate Class Weights for Imbalance from the grand merged dataset
    # These weights are often used in model training (e.g., Keras, XGBoost)
    neg_merged, pos_merged = np.bincount(
        y_train_smote
    )  # Use y_train_smote for weights after balancing
    total_merged = neg_merged + pos_merged
    log.info(
        f"\nFinal training set balance after SMOTE: {pos_merged} positive samples out of {total_merged} total samples ({pos_merged/total_merged*100:.2f}%)"
    )
    class_weight_merged = {
        0: (1 / neg_merged) * (total_merged / 2.0),
        1: (1 / pos_merged) * (total_merged / 2.0),
    }
    log.info(
        f"Calculated class weights for merged dataset (post-SMOTE): {class_weight_merged}"
    )

    # --- Step 6: Save Processed Data ---
    log.info("\n--- Saving Processed Data ---")

    # Define paths for saving
    x_train_path = os.path.join(data_root_dir, "X_train.csv")
    y_train_path = os.path.join(data_root_dir, "y_train.csv")
    x_test_path = os.path.join(data_root_dir, "X_test.csv")
    y_test_path = os.path.join(data_root_dir, "y_test.csv")

    # Save the preprocessed and balanced training/testing data
    X_train_smote.to_csv(x_train_path, index=False)
    y_train_smote.to_csv(y_train_path, index=False)  # y_train is now balanced

    X_test_scaled_df.to_csv(
        x_test_path, index=False
    )  # Note: X_test is scaled but not SMOTEd
    y_test.to_csv(y_test_path, index=False)

    log.info(f"Processed training and testing data saved to '{data_root_dir}'.")

    log.info("\n--- All Datasets Consolidated, Preprocessed, and Balanced! ---")

    # Return paths to the saved files for potential use in downstream Airflow tasks
    return {
        "X_train_path": x_train_path,
        "y_train_path": y_train_path,
        "X_test_path": x_test_path,
        "y_test_path": y_test_path,
    }
