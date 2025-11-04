import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./digit-recognizer/train.csv')
print(df.shape)                 # (60000, 785)
print(df.head())

# 任意一行拆成 28×28 看图
row = df.iloc[0]
label, pixels = row[0], row[1:].values.reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.title(f'Label = {label}')
plt.show()
