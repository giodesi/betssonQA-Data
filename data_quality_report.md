# Data Quality Assessment and Validation Report

Analysis Date: 2025-05-03 10:20:57

Validation Mode: flag_and_correct

### Validation Mode Explanation:
- **flag_only**: Only identifies issues without making corrections
- **correct_only**: Automatically corrects issues without flagging
- **flag_and_correct**: Both identifies issues and makes corrections
- **strict**: Fails on any validation error (not implemented in this version)

## Dataset Overview
- Total Records: 250000
- Total Columns: 10
- Valid Records: 764 (0.31%)
- Records with Issues: 249236 (99.69%)
- Records with Multiple Issues: 169991 (68.00%)

## Missing Values Analysis
- No missing values found in the dataset.

## Duplicate Records Analysis
- No duplicate records detected.

## Invalid Values Analysis
- Underage Customers: 12738 records
- Overage Customers: 63125 records
- Non Integer Quantities: 247535 records
- Negative Amounts: 122715 records
- Price Unit Outliers: 27513 records

### Validation Logic Applied:
- **CustomerID**: Must be valid UUID format
- **Email**: Must match standard email pattern
- **Age**: Must be between configured min/max (default 18-100)
- **PurchaseDate**: Cannot be in the future
- **PurchaseQuantity**: Must be ≥ minimum (default 1) and integer if configured
- **PurchaseAmount**: Must be ≥ minimum (default 0)

## Corrections Applied
- Underage Corrected: 12738 records
- Overage Corrected: 63125 records
- Quantities Rounded: 247535 records
- Negative Amounts Corrected: 122715 records

### Correction Logic:
- **Age**: Set to min/max bounds when out of range
- **PurchaseDate**: Future dates set to current date
- **PurchaseQuantity**: Rounded to nearest integer or set to minimum
- **PurchaseAmount**: Negative values converted to positive
- **Duplicates**: Exact duplicates removed (except first occurrence)

## Validation Status Breakdown
- Non-Integer Quantity: 247535 occurrences
- Negative Purchase Amount: 122715 occurrences
- Invalid Age (>100): 63125 occurrences
- Price Outlier: 27513 occurrences
- Invalid Age (<18): 12738 occurrences

- Total issues found: 473626
- Average issues per problematic record: 1.90

## Derived Fields and Insights

### Unit Price Analysis
- Average Unit Price: $189.09
- Median Unit Price: $72.51
- Min Unit Price: $0.00
- Max Unit Price: $16789.65
- Standard Deviation: $503.43

#### Unit Price Outlier Detection Logic:
1. Calculate quartiles (Q1 at 25%, Q3 at 75%)
2. Compute Interquartile Range (IQR = Q3 - Q1)
3. Define outlier bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
4. Values outside these bounds are flagged as outliers

- Unit Price Outliers: 27513 records (11.01%)

Outlier Statistics:
- Min outlier value: $338.49
- Max outlier value: $16789.65
- Median outlier value: $625.56

### Revenue Category Breakdown

#### Category Definition:
- **Low**: $0 - $50
- **Medium**: $50 - $100
- **High**: $100 - $500
- **Premium**: > $500

#### Distribution:
- Premium: 229958 records (91.98%) | Revenue: $992,659,948.54 (99.50%)
- High: 16101 records (6.44%) | Revenue: $4,821,742.86 (0.48%)
- Low: 2017 records (0.81%) | Revenue: $49,443.10 (0.00%)
- Medium: 1924 records (0.77%) | Revenue: $144,744.75 (0.01%)

- Total Revenue: $997,675,879.25

## Data Distribution Analysis

### Age Distribution
- Mean: 67.2 years
- Median: 71.0 years
- Range: 18.0 to 100.0 years
- Standard Deviation: 29.0 years

Age Groups Distribution:
- 65+: 136309 records (54.52%)
- 50-64: 31672 records (12.67%)
- 35-49: 31385 records (12.55%)
- 25-34: 21127 records (8.45%)
- <18: 14850 records (5.94%)
- 18-24: 14657 records (5.86%)

### Purchase Amount Distribution
- Mean: $3990.70
- Median: $3373.75
- Range: $0.03 to $22712.76
- Standard Deviation: $3017.33

Purchase Amount Percentiles:
- 25th percentile: $1590.69
- 50th percentile: $3373.75
- 75th percentile: $5751.29
- 90th percentile: $8246.12
- 95th percentile: $9798.08
- 99th percentile: $12883.08

## Overall Data Quality Assessment

### Data Quality Score: 0/100

#### Scoring Methodology:
- Based on percentage of fully valid records (no issues)
- Each record is either valid (100%) or invalid (0%)
- Records with multiple issues count the same as single-issue records
- Corrections don't affect score (original validity is measured)

- Rating: 🔴 Critical - Needs Immediate Attention

### Key Quality Indicators:
- Completeness: 100.0%
- Accuracy: -89.5%
- Uniqueness: 100.0%

## Recommendations and Next Steps
- 👥 **Medium Priority**: Add age verification for age-restricted products
- 💰 **High Priority**: Audit pricing rules and discount policies
- Implement automated price anomaly detection

### General Improvement Suggestions:
- Establish a data quality SLA with data providers
- Implement automated data quality monitoring
- Create data quality dashboard for business users
- Schedule regular data quality audits