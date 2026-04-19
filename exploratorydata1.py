import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("students.csv")

# Null values
print("Null Values:\n", df.isnull().sum())

# Handle missing
df.fillna(df.mean(numeric_only=True), inplace=True)

# Statistical summary
print("\nSummary:\n", df.describe())

# Count plots
cat_cols = df.select_dtypes(include='object')
for col in cat_cols:
    sns.countplot(x=col, data=df)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()

# Pie charts
df['race/ethnicity'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Race")
plt.show()

df['parental level of education'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Education")
plt.show()

# Crosstabs
print("\nGender vs Race:\n", pd.crosstab(df['gender'], df['race/ethnicity']))
print("\nGender vs Lunch:\n", pd.crosstab(df['gender'], df['lunch']))

# Distribution and skewness
num_cols = df.select_dtypes(include=np.number)
for col in num_cols:
    sns.histplot(df[col], kde=True)
    print(col, "skew:", df[col].skew())
    plt.show()

# Gender vs Math Score
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Math Score")
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()