import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Rohit\OneDrive\Documents\pythoner\filtered_churn_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

df.drop_duplicates(inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

df = df.ffill()

df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

df.dropna(inplace=True)

df['customer_status'] = df['customer_status'].map({
    'Stayed': 0,
    'Churned': 1
})


print("\n Cleaned Data:")
print(df.head())

print("\nFinal Info:")
print(df.info())

import seaborn as sns

#                 3. EDA 

print("\nSummary Statistics:")
print(df.describe())

print("\nTarget Distribution:")
print(df['customer_status'].value_counts())

plt.figure()
df['customer_status'].value_counts().plot(kind='bar')
plt.title("Customer Churn Distribution")
plt.xlabel("Status (0=Stayed, 1=Churned)")
plt.ylabel("Count")
plt.show()



plt.figure()
df['customer_status'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Stayed', 'Churned']
)
plt.title("Customer Churn ")
plt.ylabel("")
plt.show()  

plt.figure()
sns.boxplot(x='customer_status', y='total_charges', data=df)
plt.title("Total Charges vs Churn")
plt.show()

plt.figure()
df['monthly_charge'].hist(bins=10)
plt.title("Monthly Charges Distribution")
plt.show()

plt.figure()
plt.scatter(df['monthly_charge'], df['total_charges'])
plt.title("Monthly vs Total Charges")
plt.xlabel("Monthly")
plt.ylabel("Total")
plt.show()

plt.figure()
df.groupby('customer_status')['monthly_charge'].mean().plot(marker='o')
plt.title("Average Monthly Charge by Status")
plt.show()




#          CORRELATION

print("\n🔹 Correlation Matrix:")
corr = df.corr(numeric_only=True)
print(corr)

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



print("\n🔹 Important Correlations with Target:")
print(corr['customer_status'].sort_values(ascending=False))



from statsmodels.stats.proportion import proportions_ztest

avg_charge = df['monthly_charge'].mean()

high = df[df['monthly_charge'] > avg_charge]
low = df[df['monthly_charge'] <= avg_charge]


churn_high = high['customer_status'].sum()
churn_low = low['customer_status'].sum()


total_high = len(high)
total_low = len(low)


count = [churn_high, churn_low]
nobs = [total_high, total_low]

z_stat, p_value = proportions_ztest(count, nobs)

print("\nZ-Test Result:")
print("Z value:", z_stat)
print("P value:", p_value)

print("\nH0: Charges do not affect churn")
print("H1: Charges affect churn")


if p_value < 0.05:
    print("Reject H0 → Charges have impact on churn")
else:
    print("Accept H0 → No strong impact")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler   


df = df[df['customer_status'].notna()]


df = pd.get_dummies(df, drop_first=True)


X = df.drop('customer_status', axis=1)
y = df['customer_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=2000)


model.fit(X_train, y_train)


pred = model.predict(X_test)

print("\nPredictions (first 10):")
print(pred[:10])

acc = accuracy_score(y_test, pred)

print("\n Model Evaluation:")
print("Accuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

import seaborn as sns

plt.figure()
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
plt.title(" Comparision Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


