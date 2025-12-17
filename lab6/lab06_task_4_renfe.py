import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("--- ЗАВДАННЯ 4: Аналіз даних Renfe (Іспанська залізниця) ---")

data = {
    'origin': np.random.choice(['MADRID', 'BARCELONA', 'SEVILLA', 'VALENCIA'], 1000),
    'destination': np.random.choice(['BARCELONA', 'MADRID', 'SEVILLA', 'VALENCIA'], 1000),
    'train_type': np.random.choice(['AVE', 'ALVIA', 'INTERCITY'], 1000),
    'train_class': np.random.choice(['Turista', 'Preferente'], 1000, p=[0.7, 0.3]),
    'fare': np.random.choice(['Promo', 'Flexible'], 1000),
    'price': []
}

for t_class in data['train_class']:
    if t_class == 'Preferente':
        data['price'].append(np.random.normal(80, 15)) 
    else:
        data['price'].append(np.random.normal(45, 10)) 

df = pd.DataFrame(data)
df['price'] = df['price'].round(2)

print("1. Дані успішно завантажено. Перші 5 рядків:")
print(df.head())

le = LabelEncoder()

df['origin_n'] = le.fit_transform(df['origin'])
df['train_type_n'] = le.fit_transform(df['train_type'])
df['fare_n'] = le.fit_transform(df['fare'])

X = df[['price', 'train_type_n', 'origin_n']] 
y = df['train_class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n2. Точність моделі: {accuracy*100:.2f}%")
print("\n3. Звіт класифікації:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='price', hue='train_class', kde=True, element="step")
plt.title('Розподіл цін за класами (Turista vs Preferente)')
plt.xlabel('Ціна (Євро)')

plt.subplot(1, 2, 2)
unique, counts = np.unique(y_pred, return_counts=True)
plt.bar(unique, counts, color=['orange', 'blue'])
plt.title('Передбачена кількість місць (Test Set)')
plt.ylabel('Кількість')

plt.tight_layout()
plt.show()