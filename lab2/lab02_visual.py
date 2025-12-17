import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 1. Завантаження даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print("Будуємо графіки...")

# --- ГРАФІК 1: Діаграма розмаху (Boxplot) ---
# Це ті самі "квадратики з вусами", що показують розкид даних
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.title("Boxplot (Діаграма розмаху)")
plt.show() 
# УВАГА: Закрий вікно з графіком, щоб програма пішла далі!

# --- ГРАФІК 2: Гістограми (Histograms) ---
# Показують, як часто зустрічаються певні значення
dataset.hist()
plt.suptitle("Histograms (Гістограми)")
plt.show()
# Закрий вікно, щоб побачити наступний графік

# --- ГРАФІК 3: Матриця розсіювання (Scatter Matrix) ---
# Показує залежність всіх ознак одна від одної
scatter_matrix(dataset)
plt.suptitle("Scatter Matrix (Матриця розсіювання)")
plt.show()