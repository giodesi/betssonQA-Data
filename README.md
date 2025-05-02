# Data Quality Assessment & Validation System

## Overview

This repository contains a solution for the Betsson QA Engineer code challenge, focusing on **Data Quality Assessment** and **Data Validation**. The system provides automated data quality checks, validation rules, and correction mechanisms for customer purchase data.

---

## 📊 Data Quality Report

For a detailed analysis of dataset issues, corrections, and quality metrics:  
📄 **[View Latest Data Quality Report](./data_quality_report.md)**  

*Note: This report is automatically regenerated on each dataset update by [GitHub Actions](./.github/workflows/validate_data.yml).*

---

## Challenge Requirements Addressed

✅ **Data Quality Assessment and Identification**  
✅ **Data Validation and Error Handling**  

The solution covers:
- Comprehensive data quality issue detection
- Detailed documentation of issues
- Validation rules with multiple handling modes
- Automated corrections where appropriate
- Quality reporting with metrics and recommendations

---

## Features

### Data Validation & Cleaning
- **Validation Rules** for:
  - CustomerID (UUID format)
  - Email (standard format)
  - Age (configurable min/max)
  - PurchaseDate (no future dates)
  - PurchaseQuantity (min value, integer requirement)
  - PurchaseAmount (non-negative)
- **Multiple Validation Modes**:
  - `flag_only`: Identify issues without correction
  - `correct_only`: Automatically correct issues
  - `flag_and_correct`: Both identify and correct
  - `strict`: (Future implementation) Fail on errors

### Quality Reporting
- Detailed Markdown report with:
  - Dataset overview and quality metrics
  - Issue classification and counts
  - Correction explanations
  - Statistical analysis
  - Data quality score (0-100)
  - Actionable recommendations

### Automation
- GitHub Actions workflow that:
  - Triggers on dataset changes
  - Runs validation automatically
  - Commits updated reports

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Required packages: `pandas numpy pyyaml`

### Installation
```bash
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt  # If you create a requirements file