import pandas as pd

df = pd.read_csv('student_habits_performance_dataset.csv')
print(df.head())
print(df.describe())