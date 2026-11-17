# EduPulse — dashboard Streamlit

Painel interativo para a **Global Solution FIAP 2026.2**. Usa **somente dados reais e as saídas do modelo do grupo**: histórico de IRD 2022–2025 e classificação de risco de IRD < 2 em 2026 para as escolas públicas estaduais e municipais de São Paulo.

## Como executar

1. Tenha **Python 3.10 ou superior** instalado.
2. Abra o terminal na pasta do projeto e execute:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

3. Acesse no navegador o endereço indicado pelo terminal (geralmente `http://localhost:8501`).

## Conteúdo do pacote

```text
RadarDocente_Streamlit/
├── app.py                         entrada do painel Streamlit
├── requirements.txt               dependências do painel
├── rodar_dashboard.bat             inicialização no Windows
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                tema visual
│   ├── secrets.toml.example       exemplo de configuração da IA
│   └── secrets.toml               chave local (não versionada)
├── assets/
│   └── edupulse-logo.png
├── src/
│   ├── __init__.py
│   └── ai_summary.py              integração com Gemini e resumo alternativo
├── data/processed/
│   ├── radar_escolas.csv          histórico e previsões por escola
│   └── radar_municipios_2026.csv  indicadores e coordenadas municipais
├── tests/
│   └── test_ai_summary.py
└── modelagem/                     reprodução e documentação do modelo
    ├── RadarDocente.ipynb
    ├── README.md
    ├── requirements.txt           dependências da modelagem
    ├── data/                      bases de treino, previsão e municípios
    └── src/                       preparação dos dados do INEP
```

## Páginas

- **Visão geral:** cartões, evolução do IRD, distribuição dos alertas, mapa e escolas prioritárias.
- **Explorar escolas:** busca por nome ou código INEP, histórico, risco previsto, indicadores educacionais e resumo personalizado para acompanhamento com Gemini.
- **Mapa de municípios:** mapa, ranking e exportação CSV.
- **Entenda os indicadores:** guia de regularidade docente, faixas de risco, fontes, resumo por IA e demais indicadores, com metodologia e avaliação em uma seção recolhida.

Os filtros por município, rede e localização se aplicam às páginas de dados, inclusive ao mapa. Os agregados por município são recalculados a partir do CSV de escolas quando se usam filtros de rede ou localização.

## Resumo para acompanhamento com IA

A seção **Entenda o alerta desta escola** aparece entre os cartões de classificação
e o histórico do IRD. Ela separa três informações: o que o modelo previu (classe,
pontuação e faixa, descritas diretamente dos dados), os indicadores observados que
contextualizam a escola e as verificações sugeridas para a gestão. O painel não calcula
a contribuição individual das variáveis e não atribui causas à baixa regularidade.

O painel usa a API do Google Gemini para produzir quatro textos curtos: evolução do IRD,
outros indicadores observados, previsão para 2026 e próximo passo sugerido. A análise
destaca dois ou três indicadores adicionais com valores, etapas e anos, usando comparações
com observações anteriores quando disponíveis. As sugestões devem se apoiar nesses dados,
sem atribuir causas à previsão. O resumo é atualizado automaticamente na página
**Explorar escolas** conforme a escola, os filtros e os dados selecionados.

Configure a chave em `.streamlit/secrets.toml`, usando `.streamlit/secrets.toml.example`
como referência, ou nas variáveis de ambiente `GEMINI_API_KEY` e `GEMINI_MODEL`.
O modelo padrão é `gemini-3.5-flash-lite`; pode ser alterado para outro modelo Gemini
disponível no projeto. No Streamlit Community Cloud, preencha os mesmos campos na área
**Secrets** das configurações do aplicativo. O nome dado à chave no AI Studio é apenas
um identificador; a autenticação usa o valor secreto.

A chave fica no servidor e o arquivo local de segredos é ignorado pelo Git. Não inclua
esse arquivo em repositórios, capturas de tela ou pacotes de distribuição.
São enviados ao Google apenas os dados públicos da escola selecionada, seu histórico de
indicadores, classificação, pontuação, filtros e contagens agregadas do recorte.
O restante da base não é enviado. A integração não treina nem altera o modelo preditivo.

Resultados são guardados somente na sessão do usuário (até 20 contextos), evitando chamadas
repetidas ao navegar. Mudanças na escola, filtros, dados, modelo ou chave geram outro contexto.
As chamadas estão sujeitas à cota e à cobrança configuradas no Google AI Studio.
Se a chave estiver ausente, a API falhar ou a resposta for inválida, o painel continua
exibindo um resumo calculado diretamente dos dados, identificado como **sem geração por IA**.
Falhas não provocam tentativas automáticas repetidas; o usuário pode tentar novamente pelo botão.
Os resumos de IA são identificados e devem ser revisados antes de orientar decisões.

Para verificar a integração sem consumir a API: `python -m unittest discover -s tests -v`.

## Observações importantes

- **2026 é projeção, não IRD observado.** O notebook prevê se o IRD ficará abaixo de 2; ele não prevê o valor numérico do IRD.
- A pontuação (`prob_risco`) do modelo **não é uma probabilidade calibrada**. As classes foram determinadas no notebook: **alto ≥ 0,52**; **médio ≥ 0,10 e < 0,52**; **baixo < 0,10**.
- O aplicativo **não treina modelos**, apenas consulta as previsões já existentes nos CSVs. O notebook `modelagem/RadarDocente.ipynb` continua sendo a entrega técnica independente exigida na atividade.
- O IRD não mede diretamente falta de professores. Os alertas servem para **priorizar avaliações**, não para confirmar problemas ou atribuir responsabilidades.
- O mapa usa coordenadas das **sedes municipais**, não o endereço das escolas. A camada cartográfica precisa de internet para carregar o mapa base.

## Organização dos arquivos

Execute o aplicativo e os testes a partir da raiz do projeto. O painel consulta apenas `data/processed/`; as bases e o notebook em `modelagem/` são preservados para reprodução da entrega acadêmica e não são necessários para publicar o painel. O notebook deve ser executado a partir de `modelagem/` e exporta diretamente para `../data/processed/`. Não há cópias paralelas dos CSVs do painel.
