import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("precos.csv")

# Variações
df["variacao"] = df["preco"].pct_change()
df["variacao_anterior"] = df["variacao"].shift(1)
df["variacao_2_anterior"] = df["variacao"].shift(2)

# Média móvel
df["media_10"] = df["preco"].rolling(10).mean()

# Distância do preço em relação à média
df["distancia_media"] = (
    (df["preco"] - df["media_10"]) / df["media_10"]
)

# Próximo movimento
df["proxima_variacao"] = df["preco"].pct_change().shift(-1)

# Remove valores vazios
df = df.dropna()

# Alvo
df["alvo"] = 0
df.loc[df["proxima_variacao"] > 0.002, "alvo"] = 1
df.loc[df["proxima_variacao"] < -0.002, "alvo"] = -1

# 80% treino / 20% teste
limite = int(len(df) * 0.8)

features = [
    "variacao",
    "variacao_anterior",
    "variacao_2_anterior",
    "distancia_media"
]

X_train = df[features].iloc[:limite]
y_train = df["alvo"].iloc[:limite]

X_test = df[features].iloc[limite:]
y_test = df["alvo"].iloc[limite:]

# Modelo
modelo = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

# Treinar
modelo.fit(X_train, y_train)

# Prever
previsoes = modelo.predict(X_test)

probabilidades = modelo.predict_proba(X_test)

for i in range(10):
    print(
        "Previsão:", previsoes[i],
        "| Probabilidades:", probabilidades[i]
    )

# Taxa de acerto
acertos = (previsoes == y_test.to_numpy()).sum()
total = len(y_test)

taxa_acerto = acertos / total

print("Acertos:", acertos)
print("Total:", total)
print("Taxa de acerto:", taxa_acerto * 100, "%")

# Decisões
print("\nDecisões da IA:")
print(pd.Series(previsoes).value_counts())

# Backtest
saldo = 100
valor_operacao = 1
payout = 0.80
operacoes = 0
acertos_operacoes = 0

for i, (previsao, real) in enumerate(zip(previsoes, y_test)):

    probabilidade = max(probabilidades[i])

    if previsao == 0 or probabilidade < 0.80:
        continue

    operacoes += 1

    if previsao == real:
        acertos_operacoes += 1
        saldo += valor_operacao * payout
    else:
        saldo -= valor_operacao

print("\n--- BACKTEST ---")
print("Operações:", operacoes)
print("Saldo inicial: €100")
print("Saldo final: €", round(saldo, 2))
print("Lucro/prejuízo: €", round(saldo - 100, 2))

if operacoes > 0:
    print(
        "Taxa de acerto das operações:",
        round(acertos_operacoes / operacoes * 100, 2),
        "%"
    )

print("\n--- COMPARAÇÃO DE CONFIANÇA ---")

for limite_confianca in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:

    saldo = 100
    operacoes = 0
    acertos = 0

    for i, (previsao, real) in enumerate(zip(previsoes, y_test)):

        probabilidade = max(probabilidades[i])

        if previsao == 0 or probabilidade < limite_confianca:
            continue

        operacoes += 1

        if previsao == real:
            acertos += 1
            saldo += valor_operacao * payout
        else:
            saldo -= valor_operacao

    taxa = (acertos / operacoes * 100) if operacoes > 0 else 0

    print(
        f"Confiança {limite_confianca:.0%} | "
        f"Operações: {operacoes} | "
        f"Acerto: {taxa:.1f}% | "
        f"Resultado: €{saldo - 100:.2f}"
    )

    import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(
    df["tempo"].iloc[limite:],
    df["preco"].iloc[limite:],
    label="Preço"
)

for i, previsao in enumerate(previsoes):
    if previsao == 1:
        plt.scatter(
            df["tempo"].iloc[limite + i],
            df["preco"].iloc[limite + i],
            marker="^",
            s=80,
            label="CALL" if i == next(
                (j for j, x in enumerate(previsoes) if x == 1), -1
            ) else ""
        )

    elif previsao == -1:
        plt.scatter(
            df["tempo"].iloc[limite + i],
            df["preco"].iloc[limite + i],
            marker="v",
            s=80,
            label="PUT" if i == next(
                (j for j, x in enumerate(previsoes) if x == -1), -1
            ) else ""
        )

plt.title("Decisões da IA no período de teste")
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.legend()
plt.grid()
plt.show()