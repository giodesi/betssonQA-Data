import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import re
import json
import os
import yaml


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_customer_id(customer_id):
    """Validate UUID format"""
    try:
        uuid_obj = uuid.UUID(customer_id)
        return str(uuid_obj) == customer_id
    except (ValueError, AttributeError):
        return False


def load_validation_rules(config_path=None):
    """
    Load validation rules from a config file (JSON or YAML)

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary of validation rules
    """
    default_rules = {
        "age": {
            "min": 18,
            "max": 100
        },
        "purchase_quantity": {
            "min": 1,
            "must_be_integer": True
        },
        "purchase_amount": {
            "min": 0
        },
        "validation_mode": "flag_and_correct"  # Options: flag_only, correct_only, flag_and_correct, strict
    }

    if not config_path or not os.path.exists(config_path):
        return default_rules

    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)

        elif (config_path.endswith('.yaml') or config_path.endswith('.yml')) and yaml is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        else:
            print(f"Unsupported config file format or required library not installed: {config_path}")
            return default_rules

    except Exception as e:
        print(f"Error loading config file: {e}")
        return default_rules


def validate_and_clean_data(file_path, output_path=None, config_path=None, validation_mode=None):
    """
    Validate and clean the dataset, identifying and correcting issues based on specified mode.

    Args:
        file_path: Path to the input CSV file
        output_path: Optional path to save the cleaned data
        config_path: Optional path to validation rules config
        validation_mode: Override mode from config file (flag_only, correct_only, flag_and_correct, strict)

    Returns:
        original_df: Original dataframe
        validated_df: Validated dataframe with flags and corrections
        issues_report: Dictionary with issues found and actions taken
    """
    # Load validation rules
    rules = load_validation_rules(config_path)

    # Override validation mode if specified
    if validation_mode:
        rules["validation_mode"] = validation_mode

    # Read the data
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {len(df)} records and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, {"file_error": str(e)}

    # Store original data for comparison
    original_df = df.copy()

    # Initialize issues report
    issues_report = {
        "missing_values": {},
        "invalid_values": {},
        "duplicates": {},
        "corrections": {},
        "summary": {}
    }

    # Initialize a new DataFrame to track validation status and changes
    validated_df = df.copy()

    # Initialize ValidationStatus as lists to track multiple issues
    validated_df['ValidationStatus'] = [[] for _ in range(len(validated_df))]
    validated_df['IsValid'] = True  # Will be updated based on ValidationStatus

    # Dictionary to track reasons for changes
    correction_reasons = {}

    # Check for missing values
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            issues_report["missing_values"][column] = missing_count
            # Append missing value issue to ValidationStatus
            validated_df.loc[df[column].isna(), 'ValidationStatus'] = validated_df.loc[
                df[column].isna(), 'ValidationStatus'].apply(lambda x: x + [f'Missing {column}'])

    # Check for duplicate records
    # 1. Check for exact duplicates (all columns)
    exact_duplicates = df[df.duplicated()]
    if len(exact_duplicates) > 0:
        issues_report["duplicates"]["exact_duplicates"] = len(exact_duplicates)
        print(f"Found {len(exact_duplicates)} exact duplicate records")

        # Append duplicate issue to ValidationStatus
        dup_indices = df.duplicated(keep='first')
        validated_df.loc[dup_indices, 'ValidationStatus'] = validated_df.loc[
            dup_indices, 'ValidationStatus'].apply(lambda x: x + ['Duplicate Record'])

        # Correction: Remove exact duplicates if not in flag_only mode
        if rules["validation_mode"] != "flag_only":
            validated_df = validated_df.drop_duplicates()
            issues_report["corrections"]["exact_duplicates_removed"] = len(exact_duplicates)
            correction_reasons.update({idx: "Removed duplicate record" for idx in df[dup_indices].index})

    # 2. Check for potential business duplicates (same customer buying same product on same day)
    validated_df['PurchaseDate'] = pd.to_datetime(validated_df['PurchaseDate'])
    validated_df['PurchaseDate_date'] = validated_df['PurchaseDate'].dt.date  # Extract just the date part
    business_key_columns = ['CustomerID', 'Product', 'PurchaseDate_date']
    business_duplicates = validated_df[validated_df.duplicated(subset=business_key_columns, keep=False)]

    if len(business_duplicates) > 0:
        issues_report["duplicates"]["business_duplicates"] = len(business_duplicates)
        print(f"Found {len(business_duplicates)} potential business duplicates (same customer, product, and date)")
        # Append business duplicate issue to ValidationStatus
        bus_dup_indices = validated_df.duplicated(subset=business_key_columns, keep=False)
        validated_df.loc[bus_dup_indices, 'ValidationStatus'] = validated_df.loc[
            bus_dup_indices, 'ValidationStatus'].apply(lambda x: x + ['Potential Business Duplicate'])

    # Keep PurchaseDate_date column for reference
    validated_df = validated_df.rename(columns={'PurchaseDate_date': 'PurchaseDate_Day'})

    # 1. Check and validate CustomerID (should be a valid UUID)
    invalid_ids_mask = ~validated_df['CustomerID'].apply(validate_customer_id)
    invalid_ids = validated_df[invalid_ids_mask]
    if len(invalid_ids) > 0:
        issues_report["invalid_values"]["customer_id"] = len(invalid_ids)
        print(f"Found {len(invalid_ids)} records with invalid CustomerID format")

        # Append invalid CustomerID issue to ValidationStatus
        validated_df.loc[invalid_ids_mask, 'ValidationStatus'] = validated_df.loc[
            invalid_ids_mask, 'ValidationStatus'].apply(lambda x: x + ['Invalid CustomerID'])
        validated_df.loc[invalid_ids_mask, 'CustomerID_Valid'] = False
        validated_df.loc[~invalid_ids_mask, 'CustomerID_Valid'] = True

    # 2. Validate Email addresses
    invalid_emails_mask = ~validated_df['Email'].apply(validate_email)
    invalid_emails = validated_df[invalid_emails_mask]
    if len(invalid_emails) > 0:
        issues_report["invalid_values"]["email"] = len(invalid_emails)
        print(f"Found {len(invalid_emails)} records with invalid email format")

        # Append invalid Email issue to ValidationStatus
        validated_df.loc[invalid_emails_mask, 'ValidationStatus'] = validated_df.loc[
            invalid_emails_mask, 'ValidationStatus'].apply(lambda x: x + ['Invalid Email'])
        validated_df.loc[invalid_emails_mask, 'Email_Valid'] = False
        validated_df.loc[~invalid_emails_mask, 'Email_Valid'] = True

    # 3. Check and correct Age data
    # Create raw versions of columns that might be modified
    validated_df['Age_Raw'] = validated_df['Age']

    # Flag unrealistic ages (too young)
    min_age = rules["age"].get("min", 18)
    underage_mask = validated_df['Age'] < min_age
    underage_customers = validated_df[underage_mask]
    if len(underage_customers) > 0:
        issues_report["invalid_values"]["underage_customers"] = len(underage_customers)
        print(f"Found {len(underage_customers)} records with underage customers (<{min_age})")

        # Append underage issue to ValidationStatus
        validated_df.loc[underage_mask, 'ValidationStatus'] = validated_df.loc[
            underage_mask, 'ValidationStatus'].apply(lambda x: x + [f'Invalid Age (<{min_age})'])

        # Correction based on validation mode
        if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
            validated_df.loc[underage_mask, 'Age'] = min_age
            validated_df.loc[underage_mask, 'Age_Corrected'] = True
            validated_df.loc[underage_mask, 'Age_Correction_Reason'] = f"Increased age to minimum allowed ({min_age})"
            issues_report["corrections"]["underage_corrected"] = len(underage_customers)
            correction_reasons.update({idx: f"Age below minimum {min_age}" for idx in underage_customers.index})

    # Flag unrealistically old customers
    max_age = rules["age"].get("max", 100)
    overage_mask = validated_df['Age'] > max_age
    overage_customers = validated_df[overage_mask]
    if len(overage_customers) > 0:
        issues_report["invalid_values"]["overage_customers"] = len(overage_customers)
        print(f"Found {len(overage_customers)} records with unrealistically old customers (>{max_age})")

        # Append overage issue to ValidationStatus
        validated_df.loc[overage_mask, 'ValidationStatus'] = validated_df.loc[
            overage_mask, 'ValidationStatus'].apply(lambda x: x + [f'Invalid Age (>{max_age})'])

        # Correction based on validation mode
        if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
            validated_df.loc[overage_mask, 'Age'] = max_age
            validated_df.loc[overage_mask, 'Age_Corrected'] = True
            validated_df.loc[overage_mask, 'Age_Correction_Reason'] = f"Reduced age to maximum allowed ({max_age})"
            issues_report["corrections"]["overage_corrected"] = len(overage_customers)
            correction_reasons.update({idx: f"Age above maximum {max_age}" for idx in overage_customers.index})

    # Mark records where Age wasn't corrected
    age_corrected_mask = validated_df['Age'] != validated_df['Age_Raw']
    validated_df.loc[~age_corrected_mask, 'Age_Corrected'] = False

    # 4. Check and correct PurchaseDate
    # Create raw versions
    validated_df['PurchaseDate_Raw'] = validated_df['PurchaseDate']

    # Flag future dates
    current_date = datetime.now()
    future_dates_mask = validated_df['PurchaseDate'] > current_date
    future_dates = validated_df[future_dates_mask]
    if len(future_dates) > 0:
        issues_report["invalid_values"]["future_dates"] = len(future_dates)
        print(f"Found {len(future_dates)} records with future purchase dates")

        # Append future date issue to ValidationStatus
        validated_df.loc[future_dates_mask, 'ValidationStatus'] = validated_df.loc[
            future_dates_mask, 'ValidationStatus'].apply(lambda x: x + ['Future Purchase Date'])

        # Correction based on validation mode
        if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
            validated_df.loc[future_dates_mask, 'PurchaseDate'] = current_date
            validated_df.loc[future_dates_mask, 'PurchaseDate_Corrected'] = True
            validated_df.loc[
                future_dates_mask, 'PurchaseDate_Correction_Reason'] = "Future date adjusted to current date"
            issues_report["corrections"]["future_dates_corrected"] = len(future_dates)
            correction_reasons.update({idx: "Purchase date in future" for idx in future_dates.index})

    # Mark records where PurchaseDate wasn't corrected
    date_corrected_mask = validated_df['PurchaseDate'] != validated_df['PurchaseDate_Raw']
    validated_df.loc[~date_corrected_mask, 'PurchaseDate_Corrected'] = False

    # 5. Check and correct PurchaseQuantity
    # Create raw versions
    validated_df['PurchaseQuantity_Raw'] = validated_df['PurchaseQuantity']

    min_qty = rules["purchase_quantity"].get("min", 1)
    must_be_integer = rules["purchase_quantity"].get("must_be_integer", True)

    # Flag non-integer quantities if required
    if must_be_integer:
        non_integer_mask = validated_df['PurchaseQuantity'] != validated_df['PurchaseQuantity'].astype(int)
        non_integer_qty = validated_df[non_integer_mask]
        if len(non_integer_qty) > 0:
            issues_report["invalid_values"]["non_integer_quantities"] = len(non_integer_qty)
            print(f"Found {len(non_integer_qty)} records with non-integer purchase quantities")

            # Append non-integer issue to ValidationStatus
            validated_df.loc[non_integer_mask, 'ValidationStatus'] = validated_df.loc[
                non_integer_mask, 'ValidationStatus'].apply(lambda x: x + ['Non-Integer Quantity'])

            # Correction based on validation mode
            if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
                validated_df.loc[non_integer_mask, 'PurchaseQuantity'] = validated_df.loc[
                    non_integer_mask, 'PurchaseQuantity'].round().astype(int)
                validated_df.loc[non_integer_mask, 'PurchaseQuantity_Corrected'] = True
                validated_df.loc[non_integer_mask, 'PurchaseQuantity_Correction_Reason'] = "Rounded to nearest integer"
                issues_report["corrections"]["quantities_rounded"] = len(non_integer_qty)
                correction_reasons.update({idx: "Non-integer quantity" for idx in non_integer_qty.index})

    # Flag negative or zero quantities
    invalid_qty_mask = validated_df['PurchaseQuantity'] < min_qty
    invalid_qty = validated_df[invalid_qty_mask]
    if len(invalid_qty) > 0:
        issues_report["invalid_values"]["invalid_quantities"] = len(invalid_qty)
        print(f"Found {len(invalid_qty)} records with invalid purchase quantities (< {min_qty})")

        # Append invalid quantity issue to ValidationStatus
        validated_df.loc[invalid_qty_mask, 'ValidationStatus'] = validated_df.loc[
            invalid_qty_mask, 'ValidationStatus'].apply(lambda x: x + [f'Invalid Quantity (< {min_qty})'])

        # Correction based on validation mode
        if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
            validated_df.loc[invalid_qty_mask, 'PurchaseQuantity'] = min_qty
            validated_df.loc[invalid_qty_mask, 'PurchaseQuantity_Corrected'] = True
            validated_df.loc[
                invalid_qty_mask, 'PurchaseQuantity_Correction_Reason'] = f"Set to minimum quantity ({min_qty})"
            issues_report["corrections"]["invalid_quantities_corrected"] = len(invalid_qty)
            correction_reasons.update({idx: f"Quantity below minimum {min_qty}" for idx in invalid_qty.index})

    # Mark records where PurchaseQuantity wasn't corrected
    qty_corrected_mask = validated_df['PurchaseQuantity'] != validated_df['PurchaseQuantity_Raw']
    validated_df.loc[~qty_corrected_mask, 'PurchaseQuantity_Corrected'] = False

    # 6. Check and correct PurchaseAmount
    # Create raw versions
    validated_df['PurchaseAmount_Raw'] = validated_df['PurchaseAmount']

    min_amount = rules["purchase_amount"].get("min", 0)

    # Flag negative amounts
    negative_amounts_mask = validated_df['PurchaseAmount'] < min_amount
    negative_amounts = validated_df[negative_amounts_mask]
    if len(negative_amounts) > 0:
        issues_report["invalid_values"]["negative_amounts"] = len(negative_amounts)
        print(f"Found {len(negative_amounts)} records with negative purchase amounts")

        # Append negative amount issue to ValidationStatus
        validated_df.loc[negative_amounts_mask, 'ValidationStatus'] = validated_df.loc[
            negative_amounts_mask, 'ValidationStatus'].apply(lambda x: x + ['Negative Purchase Amount'])

        # Correction based on validation mode
        if rules["validation_mode"] in ["correct_only", "flag_and_correct"]:
            validated_df.loc[negative_amounts_mask, 'PurchaseAmount'] = abs(
                validated_df.loc[negative_amounts_mask, 'PurchaseAmount'])
            validated_df.loc[negative_amounts_mask, 'PurchaseAmount_Corrected'] = True
            validated_df.loc[
                negative_amounts_mask, 'PurchaseAmount_Correction_Reason'] = "Converted negative to positive"
            issues_report["corrections"]["negative_amounts_corrected"] = len(negative_amounts)
            correction_reasons.update({idx: "Negative purchase amount" for idx in negative_amounts.index})

    # Mark records where PurchaseAmount wasn't corrected
    amount_corrected_mask = validated_df['PurchaseAmount'] != validated_df['PurchaseAmount_Raw']
    validated_df.loc[~amount_corrected_mask, 'PurchaseAmount_Corrected'] = False

    # 7. Add derived columns for insight
    # Calculate unit price (derived column)
    validated_df['UnitPrice'] = validated_df['PurchaseAmount'] / validated_df['PurchaseQuantity']

    # Add revenue classification
    validated_df['RevenueCategory'] = pd.cut(
        validated_df['PurchaseAmount'],
        bins=[0, 50, 100, 500, float('inf')],
        labels=['Low', 'Medium', 'High', 'Premium']
    )

    # 8. Check for consistency between PurchaseQuantity and PurchaseAmount
    # Identify outliers in price per unit (using IQR method)
    Q1 = validated_df['UnitPrice'].quantile(0.25)
    Q3 = validated_df['UnitPrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, 0)  # Ensure non-negative
    upper_bound = Q3 + 1.5 * IQR

    price_outliers_mask = (validated_df['UnitPrice'] < lower_bound) | (validated_df['UnitPrice'] > upper_bound)
    price_outliers = validated_df[price_outliers_mask]
    if len(price_outliers) > 0:
        issues_report["invalid_values"]["price_unit_outliers"] = len(price_outliers)
        print(f"Found {len(price_outliers)} records with unusual price per unit")

        # Append price outlier issue to ValidationStatus
        validated_df.loc[price_outliers_mask, 'ValidationStatus'] = validated_df.loc[
            price_outliers_mask, 'ValidationStatus'].apply(lambda x: x + ['Price Outlier'])
        validated_df.loc[price_outliers_mask, 'UnitPrice_IsOutlier'] = True
        validated_df.loc[~price_outliers_mask, 'UnitPrice_IsOutlier'] = False

    # 9. Update IsValid column based on ValidationStatus
    validated_df['IsValid'] = validated_df['ValidationStatus'].apply(lambda x: len(x) == 0)

    # 10. Create a display version of ValidationStatus
    validated_df['ValidationStatus_Display'] = validated_df['ValidationStatus'].apply(
        lambda x: '; '.join(x) if x else 'Valid'
    )

    # 11. Add a column with all corrections applied to the record
    validated_df['CorrectionsApplied'] = validated_df.apply(
        lambda row: "; ".join([col.replace('_Correction_Reason', '') + ": " + str(reason)
                    for col, reason in row.items()
                    if '_Correction_Reason' in col and not pd.isna(reason)]),
        axis=1
    )

    # Handle empty correction strings
    validated_df.loc[validated_df['CorrectionsApplied'] == '', 'CorrectionsApplied'] = 'None'

    # Calculate summary statistics for the issues report
    all_issues = [issue for sublist in validated_df['ValidationStatus'] for issue in sublist]
    issues_report["summary"] = {
        "total_records": len(original_df),
        "records_with_issues": len(validated_df[~validated_df['IsValid']]),
        "total_issues_found": len(all_issues),
        "total_corrections_made": sum(issues_report["corrections"].values() if issues_report["corrections"] else [0]),
        "validation_mode_used": rules["validation_mode"],
        "records_with_multiple_issues": len(validated_df[validated_df['ValidationStatus'].apply(len) > 1])
    }

    # Save validated data if output path is provided
    if output_path:
        # Convert ValidationStatus to string for CSV output
        validated_df.to_csv(output_path, index=False)
        print(f"Validated data saved to {output_path}")

    return original_df, validated_df, issues_report


def generate_report(issues_report, original_df, validated_df, output_file=None):
    """Generate a detailed report of all data quality issues and corrections with explicit logic explanations"""

    report = []
    report.append("# Data Quality Assessment and Validation Report")
    report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Validation mode used
    validation_mode = issues_report["summary"].get("validation_mode_used", "Unknown")
    report.append(f"\nValidation Mode: {validation_mode}")
    report.append("\n### Validation Mode Explanation:")
    report.append("- **flag_only**: Only identifies issues without making corrections")
    report.append("- **correct_only**: Automatically corrects issues without flagging")
    report.append("- **flag_and_correct**: Both identifies issues and makes corrections")
    report.append("- **strict**: Fails on any validation error (not implemented in this version)")

    report.append(f"\n## Dataset Overview")
    report.append(f"- Total Records: {len(original_df)}")
    report.append(f"- Total Columns: {len(original_df.columns)}")
    report.append(
        f"- Valid Records: {len(validated_df[validated_df['IsValid']])} ({len(validated_df[validated_df['IsValid']]) / len(validated_df) * 100:.2f}%)")
    report.append(
        f"- Records with Issues: {issues_report['summary']['records_with_issues']} ({issues_report['summary']['records_with_issues'] / len(validated_df) * 100:.2f}%)")
    report.append(
        f"- Records with Multiple Issues: {issues_report['summary']['records_with_multiple_issues']} ({issues_report['summary']['records_with_multiple_issues'] / len(validated_df) * 100:.2f}%)")

    # Missing values section
    report.append(f"\n## Missing Values Analysis")
    if issues_report["missing_values"]:
        for column, count in issues_report["missing_values"].items():
            report.append(f"- {column}: {count} missing values ({count / len(original_df) * 100:.2f}%)")
        report.append("\n### Missing Values Handling:")
        report.append("- Missing values are flagged but not automatically imputed")
        report.append("- Recommended actions: Investigate data collection processes or implement imputation strategies")
    else:
        report.append("- No missing values found in the dataset.")

    # Duplicates section
    report.append(f"\n## Duplicate Records Analysis")
    if "duplicates" in issues_report and issues_report["duplicates"]:
        for issue, count in issues_report["duplicates"].items():
            report.append(f"- {issue.replace('_', ' ').title()}: {count} records")

        report.append("\n### Duplicate Detection Logic:")
        report.append("- **Exact duplicates**: Rows with identical values in all columns")
        report.append("- **Business duplicates**: Same CustomerID + Product + PurchaseDate combination")
        report.append("\n### Duplicate Handling:")
        report.append("- Exact duplicates are removed in correction modes")
        report.append("- Business duplicates are flagged for manual review")
    else:
        report.append("- No duplicate records detected.")

    # Invalid values section
    report.append(f"\n## Invalid Values Analysis")
    if issues_report["invalid_values"]:
        for issue, count in issues_report["invalid_values"].items():
            report.append(f"- {issue.replace('_', ' ').title()}: {count} records")

        report.append("\n### Validation Logic Applied:")
        report.append("- **CustomerID**: Must be valid UUID format")
        report.append("- **Email**: Must match standard email pattern")
        report.append("- **Age**: Must be between configured min/max (default 18-100)")
        report.append("- **PurchaseDate**: Cannot be in the future")
        report.append("- **PurchaseQuantity**: Must be ≥ minimum (default 1) and integer if configured")
        report.append("- **PurchaseAmount**: Must be ≥ minimum (default 0)")
    else:
        report.append("- No invalid values detected.")

    # Corrections section
    report.append(f"\n## Corrections Applied")
    if issues_report["corrections"]:
        for correction, count in issues_report["corrections"].items():
            report.append(f"- {correction.replace('_', ' ').title()}: {count} records")

        report.append("\n### Correction Logic:")
        report.append("- **Age**: Set to min/max bounds when out of range")
        report.append("- **PurchaseDate**: Future dates set to current date")
        report.append("- **PurchaseQuantity**: Rounded to nearest integer or set to minimum")
        report.append("- **PurchaseAmount**: Negative values converted to positive")
        report.append("- **Duplicates**: Exact duplicates removed (except first occurrence)")
    else:
        report.append("- No corrections were applied.")

    # ValidationStatus breakdown
    report.append(f"\n## Validation Status Breakdown")
    all_issues = [issue for sublist in validated_df['ValidationStatus'] for issue in sublist]
    if all_issues:
        status_counts = pd.Series(all_issues).value_counts()
        for status, count in status_counts.items():
            report.append(f"- {status}: {count} occurrences")
        report.append(f"\n- Total issues found: {len(all_issues)}")
        report.append(
            f"- Average issues per problematic record: {len(all_issues) / issues_report['summary']['records_with_issues']:.2f}")
    else:
        report.append("- No validation issues found")

    # Derived fields and insights
    report.append(f"\n## Derived Fields and Insights")

    # Unit Price Analysis
    if 'UnitPrice' in validated_df.columns:
        report.append(f"\n### Unit Price Analysis")
        report.append(f"- Average Unit Price: ${validated_df['UnitPrice'].mean():.2f}")
        report.append(f"- Median Unit Price: ${validated_df['UnitPrice'].median():.2f}")
        report.append(f"- Min Unit Price: ${validated_df['UnitPrice'].min():.2f}")
        report.append(f"- Max Unit Price: ${validated_df['UnitPrice'].max():.2f}")
        report.append(f"- Standard Deviation: ${validated_df['UnitPrice'].std():.2f}")

        report.append("\n#### Unit Price Outlier Detection Logic:")
        report.append("1. Calculate quartiles (Q1 at 25%, Q3 at 75%)")
        report.append("2. Compute Interquartile Range (IQR = Q3 - Q1)")
        report.append("3. Define outlier bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]")
        report.append("4. Values outside these bounds are flagged as outliers")

        if 'UnitPrice_IsOutlier' in validated_df.columns:
            outlier_count = validated_df['UnitPrice_IsOutlier'].sum()
            report.append(
                f"\n- Unit Price Outliers: {outlier_count} records ({outlier_count / len(validated_df) * 100:.2f}%)")

            if outlier_count > 0:
                outlier_stats = validated_df[validated_df['UnitPrice_IsOutlier']]['UnitPrice'].describe()
                report.append("\nOutlier Statistics:")
                report.append(f"- Min outlier value: ${outlier_stats['min']:.2f}")
                report.append(f"- Max outlier value: ${outlier_stats['max']:.2f}")
                report.append(f"- Median outlier value: ${outlier_stats['50%']:.2f}")

    # Revenue Category Breakdown
    if 'RevenueCategory' in validated_df.columns:
        report.append(f"\n### Revenue Category Breakdown")
        report.append("\n#### Category Definition:")
        report.append("- **Low**: $0 - $50")
        report.append("- **Medium**: $50 - $100")
        report.append("- **High**: $100 - $500")
        report.append("- **Premium**: > $500")

        revenue_counts = validated_df['RevenueCategory'].value_counts()
        total_revenue = validated_df['PurchaseAmount'].sum()

        report.append("\n#### Distribution:")
        for category, count in revenue_counts.items():
            category_revenue = validated_df[validated_df['RevenueCategory'] == category]['PurchaseAmount'].sum()
            report.append(
                f"- {category}: {count} records ({count / len(validated_df) * 100:.2f}%) | "
                f"Revenue: ${category_revenue:,.2f} ({category_revenue / total_revenue * 100:.2f}%)")

        report.append(f"\n- Total Revenue: ${total_revenue:,.2f}")

    # Data Distribution Analysis
    report.append(f"\n## Data Distribution Analysis")

    # Age Distribution
    if 'Age' in validated_df.columns:
        report.append(f"\n### Age Distribution")
        age_stats = validated_df['Age'].describe()
        report.append(f"- Mean: {age_stats['mean']:.1f} years")
        report.append(f"- Median: {age_stats['50%']:.1f} years")
        report.append(f"- Range: {age_stats['min']:.1f} to {age_stats['max']:.1f} years")
        report.append(f"- Standard Deviation: {age_stats['std']:.1f} years")

        # Age bins analysis
        age_bins = [0, 18, 25, 35, 50, 65, 100]
        age_labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
        age_dist = pd.cut(validated_df['Age'], bins=age_bins, labels=age_labels).value_counts()
        report.append("\nAge Groups Distribution:")
        for group, count in age_dist.items():
            report.append(f"- {group}: {count} records ({count / len(validated_df) * 100:.2f}%)")

    # Purchase Amount Distribution
    if 'PurchaseAmount' in validated_df.columns:
        report.append(f"\n### Purchase Amount Distribution")
        amount_stats = validated_df['PurchaseAmount'].describe()
        report.append(f"- Mean: ${amount_stats['mean']:.2f}")
        report.append(f"- Median: ${amount_stats['50%']:.2f}")
        report.append(f"- Range: ${amount_stats['min']:.2f} to ${amount_stats['max']:.2f}")
        report.append(f"- Standard Deviation: ${amount_stats['std']:.2f}")

        # Amount percentile analysis
        percentiles = validated_df['PurchaseAmount'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        report.append("\nPurchase Amount Percentiles:")
        for p, val in percentiles.items():
            report.append(f"- {int(p * 100)}th percentile: ${val:.2f}")

    # Overall Data Quality Score
    report.append(f"\n## Overall Data Quality Assessment")

    valid_percentage = len(validated_df[validated_df['IsValid']]) / len(validated_df) * 100
    quality_score = round(valid_percentage)

    report.append(f"\n### Data Quality Score: {quality_score}/100")
    report.append("\n#### Scoring Methodology:")
    report.append("- Based on percentage of fully valid records (no issues)")
    report.append("- Each record is either valid (100%) or invalid (0%)")
    report.append("- Records with multiple issues count the same as single-issue records")
    report.append("- Corrections don't affect score (original validity is measured)")

    # Quality rating based on score
    if quality_score >= 98:
        rating = "Excellent"
        color = "🟢"
    elif quality_score >= 95:
        rating = "Very Good"
        color = "🟢"
    elif quality_score >= 90:
        rating = "Good"
        color = "🟡"
    elif quality_score >= 80:
        rating = "Fair"
        color = "🟠"
    elif quality_score >= 70:
        rating = "Poor"
        color = "🔴"
    else:
        rating = "Critical - Needs Immediate Attention"
        color = "🔴"

    report.append(f"\n- Rating: {color} {rating}")

    # Data Quality KPIs
    report.append(f"\n### Key Quality Indicators:")
    report.append(
        f"- Completeness: {100 - sum(issues_report['missing_values'].values()) / len(validated_df) * 100:.1f}%")
    report.append(f"- Accuracy: {100 - len(all_issues) / len(validated_df) * 100:.1f}%")
    report.append(
        f"- Uniqueness: {100 - issues_report['duplicates'].get('exact_duplicates', 0) / len(validated_df) * 100:.1f}%")

    # Recommendations based on findings
    report.append(f"\n## Recommendations and Next Steps")

    if issues_report["missing_values"]:
        report.append("- 🚨 **High Priority**: Implement mandatory field validation in data entry systems")
        report.append("- Consider imputation strategies for critical missing values")

    if issues_report["duplicates"]:
        report.append("- 🔄 **Medium Priority**: Add duplicate prevention in transaction processing")
        report.append("- Implement business rules to detect suspicious duplicate purchases")

    if "customer_id" in issues_report["invalid_values"]:
        report.append("- 🔧 **High Priority**: Enforce UUID validation at point of entry")

    if "email" in issues_report["invalid_values"]:
        report.append("- ✉️ **Medium Priority**: Add real-time email validation in web forms")

    if "underage_customers" in issues_report["invalid_values"] or "overage_customers" in issues_report[
        "invalid_values"]:
        report.append("- 👥 **Medium Priority**: Add age verification for age-restricted products")

    if "future_dates" in issues_report["invalid_values"]:
        report.append("- 📅 **High Priority**: Validate purchase dates against server time")

    if "price_unit_outliers" in issues_report["invalid_values"]:
        report.append("- 💰 **High Priority**: Audit pricing rules and discount policies")
        report.append("- Implement automated price anomaly detection")

    # Add general recommendations
    report.append("\n### General Improvement Suggestions:")
    report.append("- Establish a data quality SLA with data providers")
    report.append("- Implement automated data quality monitoring")
    report.append("- Create data quality dashboard for business users")
    report.append("- Schedule regular data quality audits")

    # Join the report sections and return
    full_report = "\n".join(report)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"Report saved to {output_file}")

    return full_report


def create_validation_config(output_path, custom_rules=None):
    """
    Create a validation rules configuration file

    Args:
        output_path: Path to save the configuration file
        custom_rules: Optional dictionary of custom validation rules

    Returns:
        Dictionary of validation rules
    """
    # Default validation rules
    default_rules = {
        "validation_mode": "flag_and_correct",  # Options: flag_only, correct_only, flag_and_correct, strict
        "age": {
            "min": 18,
            "max": 100
        },
        "purchase_quantity": {
            "min": 1,
            "must_be_integer": True
        },
        "purchase_amount": {
            "min": 0,
            "outlier_detection": True,
            "outlier_method": "iqr",  # Options: iqr, zscore, percentile
            "outlier_threshold": 1.5  # For IQR method
        },
        "unit_price": {
            "min": 0,
            "outlier_detection": True
        },
        "purchase_date": {
            "no_future_dates": True,
            "min_date": "2000-01-01"  # Optional minimum date
        },
        "email": {
            "validation_pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        },
        "customer_id": {
            "uuid_validation": True
        }
    }

    # Update with custom rules if provided
    if custom_rules:
        # Recursively update the default rules
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d

        default_rules = update_dict(default_rules, custom_rules)

    # Determine file format based on extension
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(default_rules, f, indent=4)
        print(f"Validation rules saved to {output_path} in JSON format")

    elif (output_path.endswith('.yaml') or output_path.endswith('.yml')) and yaml is not None:
        with open(output_path, 'w') as f:
            yaml.dump(default_rules, f, default_flow_style=False)
        print(f"Validation rules saved to {output_path} in YAML format")

    else:
        # Default to JSON if no recognized extension or yaml not available
        if not output_path.endswith('.json'):
            output_path += '.json'
        with open(output_path, 'w') as f:
            json.dump(default_rules, f, indent=4)
        print(f"Validation rules saved to {output_path} in JSON format")

    return default_rules


# Example usage
if __name__ == "__main__":
    # File paths
    input_file = "dataset.csv"
    output_file = "validated_dataset.csv"
    report_file = "data_quality_report.md"
    config_file = "validation_rules.json"

    # Create a validation config (only needed once)
    # Uncomment to create a default config file
    # create_validation_config(config_file)

    # Run validation with specified mode
    original_df, validated_df, issues_report = validate_and_clean_data(
        input_file,
        output_file,
        config_path=config_file,
        validation_mode="flag_and_correct"  # Options: flag_only, correct_only, flag_and_correct, strict
    )

    # Generate report
    if original_df is not None and validated_df is not None:
        report = generate_report(issues_report, original_df, validated_df, report_file)

        print("\nData validation and cleaning completed successfully.")
        print(f"See the detailed report in {report_file}")
        print(f"Records requiring attention: {issues_report['summary']['records_with_issues']}")
        print(f"Total issues found: {issues_report['summary']['total_issues_found']}")
        print(f"Records with multiple issues: {issues_report['summary']['records_with_multiple_issues']}")
    else:
        print("Data validation failed. Check the error messages above.")