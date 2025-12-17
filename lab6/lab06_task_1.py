import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

print("--- ЗАВДАННЯ 3: Прогноз за погодою (Варіант 11) ---")

# Вхідні дані
data = {
    'Outlook':  ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind':     ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
                 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play':     ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Кодування ознак
le_outlook = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook_n'] = le_outlook.fit_transform(df['Outlook'])
df['Humidity_n'] = le_humidity.fit_transform(df['Humidity'])
df['Wind_n'] = le_wind.fit_transform(df['Wind'])
df['Play_n'] = le_play.fit_transform(df['Play'])

X = df[['Outlook_n', 'Humidity_n', 'Wind_n']]
y = df['Play_n']

# Навчання моделі
model = GaussianNB()
model.fit(X, y)

# Варіант 11: Overcast, High, Weak
var11_outlook = le_outlook.transform(['Overcast'])[0]
var11_humidity = le_humidity.transform(['High'])[0]
var11_wind = le_wind.transform(['Weak'])[0]

input_data = [[var11_outlook, var11_humidity, var11_wind]]

# Прогноз
predicted_code = model.predict(input_data)[0]
predicted_label = le_play.inverse_transform([predicted_code])[0]
proba = model.predict_proba(input_data)[0]

print(f"Умова: Outlook=Overcast, Humidity=High, Wind=Weak")
print(f"Числове представлення умови: {input_data}")
print(f"\n--- РЕЗУЛЬТАТ ПРОГНОЗУ ---")
print(f"Гра відбудеться? -> {predicted_label.upper()}")
print(f"Ймовірність 'No':  {proba[0]*100:.2f}%")
print(f"Ймовірність 'Yes': {proba[1]*100:.2f}%")
