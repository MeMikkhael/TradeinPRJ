import numpy as np
import pandas as pd

np.random.seed(42)

tempo = np.arange(1000)

tendencia = tempo * 0.02

oscilacao = np.sin(tempo * 0.05) * 2

ruido = np.random.randn(1000) * 0.3

precos = 100 + tendencia + oscilacao + ruido

df = pd.DataFrame({
    "tempo": range(1, 1001),
    "preco": precos
})

df.to_csv("precos.csv", index=False)

print("1000 preços gerados!")

import matplotlib.pyplot as plt

plt.plot(df["preco"])
plt.title("1000 preços simulados")
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.show()