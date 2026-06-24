# Data Directory

This folder contains sample data for testing the ML pipeline.

## Sample Data

Due to file size limitations, only small sample data is included here.

## Full Dataset Download

### Option 1: Kaggle Datasets

For Mobile Phone Price Classification:
```
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
```

For other datasets:
```
1. Visit https://www.kaggle.com/datasets
2. Search for your dataset
3. Download and extract to this folder
```

### Option 2: UCI Machine Learning Repository

```
https://archive.ics.uci.edu/ml/datasets.php
```

### Option 3: Direct Download

If you have a direct link to your dataset:

```bash
# Example: Download from URL
curl -o data.csv "https://example.com/dataset.csv"

# Or using wget
wget -O data.csv "https://example.com/dataset.csv"
```

## Data Structure

Your data file should be:
- Format: CSV, Excel (.xlsx), or JSON
- Include a target column (label/target variable)
- First row should be column headers

## Usage

1. Download your dataset
2. Place the file in this folder
3. Update paths in `app/preprocess.py` or `demo/demo.ipynb`

## Example

```
data/
├── README.md           # This file
├── sample.csv          # Small sample data (optional)
└── mobile_train.csv    # Your actual data
```

## Notes

- **Do NOT commit large datasets** to GitHub
- Add large files to `.gitignore`
- Only commit small sample/demo data
