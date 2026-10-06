# TradeInPRJ

Projeto experimental em Python que gera uma série de preços simulados e treina uma árvore de decisão para classificar possíveis movimentos de preço. O programa também avalia previsões, simula um backtest simples e exibe um gráfico.

> Este projeto é para aprendizado e experimentação. Os dados são simulados e os resultados não representam recomendações nem previsões confiáveis para operações financeiras reais.

## Requisitos

- Python 3.10 ou superior
- `numpy`, `pandas`, `scikit-learn` e `matplotlib`

## Instalação

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install numpy pandas scikit-learn matplotlib
```

## Como executar

Para gerar uma nova série simulada e atualizar `precos.csv`:

```powershell
python gerar_dados.py
```

Para treinar e avaliar o modelo usando os dados de `precos.csv`:

```powershell
python main.py
```

Os scripts exibem gráficos; feche a janela do gráfico para encerrar a execução. `main.py` espera que o CSV tenha as colunas `tempo` e `preco`. O arquivo `precos.csv` incluído no repositório pode ser usado diretamente, sem executar primeiro o gerador.

## Estrutura

- `gerar_dados.py`: cria 1.000 preços simulados e salva o resultado em `precos.csv`.
- `main.py`: calcula variáveis a partir dos preços, treina uma árvore de decisão e mostra métricas, backtest e gráfico.
- `precos.csv`: série de preços usada pelo modelo.