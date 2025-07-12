
````markdown
# Análise Comparativa entre SVD e KNN em Sistemas de Recomendação

Este projeto realiza uma análise comparativa entre dois algoritmos de sistemas de recomendação:  
- **SVD (Singular Value Decomposition)** — fatoração de matrizes (fatores latentes)  
- **KNN (K-Nearest Neighbors)** — filtragem colaborativa baseada em vizinhança (item-based)  

Utilizando o dataset **MovieLens 100k**, o projeto avalia os algoritmos nas métricas RMSE e MAE, aplicando validação cruzada de 5 folds e divisão treino/teste.

---

## Arquivo principal

- `main01.py`: código Python para carregar dados, treinar, testar e avaliar os modelos SVD e KNN, além de gerar gráficos comparativos.

---

## Requisitos

- Python 3.6 ou superior  
- Bibliotecas Python:
  - pandas
  - matplotlib
  - scikit-surprise (surprise)

---

## Instalação

Execute o comando abaixo para instalar as dependências:

```bash
pip install pandas matplotlib scikit-surprise
````

---

## Como usar

1. Clone este repositório ou baixe o arquivo `main01.py`.
2. Execute o script:

```bash
python main01.py
```

O script irá:

* Carregar o dataset MovieLens 100k
* Dividir os dados em treino e teste
* Treinar e avaliar os algoritmos SVD e KNN (item-based)
* Realizar validação cruzada de 5 folds para comparação
* Exibir métricas RMSE e MAE no console
* Gerar gráficos comparativos de desempenho por fold

---

## Resultados esperados

* Métricas RMSE e MAE para ambos os algoritmos no conjunto de teste
* Gráficos comparativos mostrando o desempenho de SVD e KNN ao longo dos folds

---

## Referências

* MovieLens 100k dataset: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)
* Surprise Library: [http://surpriselib.com/](http://surpriselib.com/)
* Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

---

## Contato

Para dúvidas ou contribuições, abra uma issue ou envie um pull request.


