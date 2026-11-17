# Radar Docente: entrega de ML (GS 2026.2, 2TSCOA)

Modelo que prevê quais escolas públicas de SP (redes municipal e estadual) vão estar em **baixa regularidade docente (IRD < 2) no ano seguinte**.

```
modelagem/
├── README.md                         este arquivo
├── RadarDocente.ipynb                notebook principal, já executado
├── requirements.txt                  bibliotecas Python necessárias
├── data/
│   ├── base_radar_docente.csv        base do modelo: escola × ano (2022–2024), com o alvo
│   ├── base_previsao_2025.csv        dados de 2025, sem alvo, usados para prever 2026
│   └── municipios.csv                coordenadas dos municípios (para o mapa)
└── src/
    ├── inep.py                       lê e limpa as planilhas do INEP
    └── build_dataset.py              monta os CSVs de data/ a partir das planilhas
```

## Notebook
Contém todos os itens exigidos na entrega: problema, dados, pré-processamento, divisão temporal, modelos (baseline, Regressão Logística e Random Forest), métricas, gráficos, auditoria de viés, conclusão e texto analítico. A seção 12 gera os arquivos de `../data/processed/`.

Já está executado: dá para abrir e ler sem rodar nada.

## data/
Bases já tratadas. O notebook lê esses CSVs direto, sem precisar das planilhas originais do INEP.

## ../data/processed/
Saídas para o protótipo do painel no pitch (números reais):
- `radar_escolas.csv`: uma linha por escola e ano. De 2022 a 2024 tem os dados reais; em 2025, o real mais o previsto; em 2026, só o previsto (`prob_risco` e `classe_risco` = alto, médio ou baixo).
- `radar_municipios_2026.csv`: escolas em risco alto e médio por município, com latitude e longitude para o mapa de bolhas.

## src/
Scripts que transformaram as planilhas do INEP nos CSVs de `data/`. Só são necessários para refazer a base do zero. O passo a passo, os links e os nomes dos arquivos estão na seção 2 do notebook.

## Para rodar o notebook
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace RadarDocente.ipynb
```
Leva cerca de 1 minuto.

## Integração com o EduPulse

Execute o notebook com o diretório de trabalho em `modelagem/`. As bases de treino ficam em `modelagem/data/` e as saídas do painel são gravadas em `../data/processed/`, sem cópias duplicadas. Para abrir o painel, volte à raiz do projeto e execute `python -m streamlit run app.py`.
