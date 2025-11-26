Sistema de Treinamento e Previsão em Nuvem com Regressão Linear
Sistema simples de analise de dados

Sistema de Análise e Previsão Remota de Séries Temporais

Nicolas Alencar de Oliveira
Matrícula: [preencher]
E-mail: [preencher]

Descrição Geral
Este projeto implementa um sistema completo em nuvem para:

-Treinar remotamente um modelo de regressão linear usando arquivos CSV enviados ao sistema;
-Processar e validar conjuntos de teste;
-Executar o modelo em produção;
-Armazenar previsões, métricas de desempenho e artefatos de modelo;
-Manter histórico completo de execuções e versões do modelo;
-Fornecer previsões e resultados para posterior análise em dashboards (ex.: Power BI).
-O sistema foi desenvolvido como solução escalável para análise de séries temporais, onde o objetivo é prever o valor da coluna time usando os cinco momentos anteriores (time1 a time5).

Dataset

-Formato: CSV
-Origem: Dataset fornecido pelo professor da disciplina
-Volume: Aproximadamente [x] linhas (preencher se quiser)

Finalidade:
Treinamento e teste de um modelo preditivo para séries temporais não estacionárias, com forte tendência.

Esquema das colunas:
Coluna	Descrição
time	Valor alvo a ser previsto
time1	Valor observado 1 passo antes
time2	Valor observado 2 passos antes
time3	Valor observado 3 passos antes
time4	Valor observado 4 passos antes
time5	Valor observado 5 passos antes

Arquitetura da Solução
A solução utiliza a arquitetura moderna de processamento distribuído em nuvem:

flowchart TD
    A[Usuário envia CSV] --> B[Azure Blob Storage]
    B --> C[Azure Functions - API]
    C -->|Treino| D[Azure SQL Database<br/>model_versions, executions]
    C -->|Previsões| D
    C -->|Artefatos| E[Blob Storage - models/, outputs/]
    D --> F[Dashboard Power BI ou front-end]

Componentes:

Azure Blob Storage
Armazena dados brutos, artefatos de modelo e arquivos de saída.

Azure Functions (Python)
Implementa endpoints:

/api/train → treina modelo
/api/test → executa previsões
/api/reset → redefine modelo
/api/processamento → teste básico da função

Azure SQL Database
Gerencia:

versões de modelo (model_versions)
execuções (executions)
métricas (RMSE, MSE, R²)
logs operacionais

Power BI (ou dashboard Web)
Leitura direta do SQL Database para visualização das previsões e métricas.

Docker + docker-compose
Ambiente de desenvolvimento local padronizado.

GitHub Actions
Pipeline CI/CD para deploy automático da Azure Function.

Fluxo de Funcionalidades
✔ 1. Upload da Base de Treino

Arquivo CSV enviado ao Blob Storage.

✔ 2. Treino Remoto

Endpoint /api/train:

Lê o CSV

Normaliza dados (min-max)
Treina regressão linear
Gera métricas do modelo (RMSE, MSE, R²)
Cria versão do modelo
Salva artefato .joblib no Blob
Escreve métricas e versão no SQL

✔ 3. Upload da Base de Teste

Arquivo CSV de teste também entra no Blob.

✔ 4. Execução / Previsão

Endpoint /api/test:

Carrega o modelo treinado
Aplica o pré-processamento
Gera previsões
Salva CSV de saída no Blob
Registra execução e métricas no SQL

✔ 5. Reset do Modelo

Endpoint /api/reset recria a tabela de modelos.

Métricas de Avaliação

O sistema calcula automaticamente:
RMSE – Root Mean Squared Error
MSE – Mean Squared Error
R² – Coeficiente de Determinação

Esses valores ficam registrados no SQL Database.

Estrutura do Repositório
/project-root
│── function_app.py
│── requirements.txt
│── host.json
│── local.settings.json (não vai para o GitHub)
│── dockerfile
│── docker-compose.yml
│── /docs
│── /sql
│── /images
│── README.md

CI/CD – GitHub Actions

Pipeline realiza:

Build da Azure Function
Testes automatizados
Deploy para Azure Function App
Limpeza de versões antigas

Como Executar Localmente (Docker)
docker build -t func-analise .
docker run -p 7071:7071 func-analise


A API ficará disponível em:
http://localhost:7071/api/train

Como Usar na Nuvem
✔ Treinar modelo:
POST https://NOME-DA-FUNCTION.azurewebsites.net/api/train

✔ Testar modelo:
POST https://NOME-DA-FUNCTION.azurewebsites.net/api/test

✔ Verificar healthcheck:
GET https://NOME-DA-FUNCTION.azurewebsites.net/api/processamento

Dashboard

O dashboard analítico foi construído em Power BI, consumindo diretamente:
Tabela model_versions
Tabela executions
Previsões armazenadas no Blob

Inclui:

Histórico de RMSE
Comparação real × previsto
Evolução das versões do modelo
Tendências da série temporal
(Inserir prints depois)

Referências

Azure Functions Python Docs
Azure Blob Storage SDK
Azure SQL Database
scikit-learn — LinearRegression
pandas, joblib, SQLAlchemy
