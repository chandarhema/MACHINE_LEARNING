import pandas as pd

# === STEP 1: LOAD DATA ===
file_path = "GSE42133_non-normalized_data.txt"

print("Loading dataset...")
df = pd.read_csv(file_path, sep="\t")

# === STEP 2: KEEP ONLY AVG_SIGNAL COLUMNS ===
print("Filtering AVG_Signal columns...")

# Keep PROBE_ID + AVG_Signal columns
cols_to_keep = ["PROBE_ID"] + [col for col in df.columns if "AVG_Signal" in col]

df_clean = df[cols_to_keep]

# === STEP 3: CLEAN COLUMN NAMES ===
print("Cleaning column names...")

# Remove '.AVG_Signal' from column names
df_clean.columns = [col.replace(".AVG_Signal", "") for col in df_clean.columns]

# === STEP 4: SET INDEX ===
df_clean.set_index("PROBE_ID", inplace=True)

# === STEP 5: TRANSPOSE (IMPORTANT FOR ML) ===
print("Transposing dataset (samples as rows)...")

df_clean = df_clean.T

# === STEP 6: SAVE CLEAN DATA ===
output_file = "clean_expression_data.csv"
df_clean.to_csv(output_file)

print(f" Clean dataset saved as: {output_file}")
print("Shape:", df_clean.shape)

# === STEP 7: BASIC CHECKS ===
print("\n=== DATA SUMMARY ===")

print("Shape:", df_clean.shape)

print("\nHead:")
print(df_clean.head())

print("\nInfo:")
print(df_clean.info())

print("\nDescribe:")
print(df_clean.describe())

print("\nMissing values:")
print(df_clean.isnull().sum().sum())

print("\nUnique values (sample):")
print(df_clean.nunique().head())

# Check for zero values
print("\nZero values count:")
print((df_clean == 0).sum().sum())

# Check duplicate rows
print("\nDuplicate samples:")
print(df_clean.duplicated().sum())