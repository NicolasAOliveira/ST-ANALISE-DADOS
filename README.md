Sistema de Treinamento e Previsão em Nuvem com Regressão Linear​ Sistema de Análise de Dados

Equipe
Nicolas Alencar de Oliveira — Matrícula: 23012092 — E-mail: nicolas.ao@puccampinas.edu.br​
Vitor Martins Pinto — Matrícula: 23020961 — E-mail: vitor.mp2@puccampinas.edu.br​
Lucas Gaino Sarzedo — Matrícula: - — E-mail: -

Descrição Geral
Sistema em nuvem para análise e previsão de séries temporais, com foco em um modelo de regressão linear que prevê a coluna time a partir dos cinco valores anteriores (time1 a time5).​
O objetivo é permitir treino remoto, execução em produção, versionamento de modelos e disponibilização de resultados para dashboards no Front-end.​
O sistema suporta upload de bases CSV, cálculo de métricas (RMSE, MSE, R²) e histórico completo de execuções e versões do modelo em um ambiente escalável.​

Dataset
Fonte: Dataset fornecido pelo professor Fernando da disciplina de Aprendizado Supervisionado, em formato CSV.​
https://puc-campinas.instructure.com/files/3371440/download?download_frd=1

Volume de dados esperado: aproximadamente 30 Mb no banco de dados SQL.​

Licenciamento: uso acadêmico para a disciplina, conforme diretrizes do curso e do professor (preencher se houver licença explícita do dataset).​

O dataset representa uma série temporal não estacionária, com forte tendência, usada para treinar e testar o modelo preditivo da coluna time.​
Esquema das colunas principais: time (alvo), time1, time2, time3, time4 e time5 (defasagens temporais usadas como preditores).​

Arquitetura da Solução
Diagrama em Mermaid:

<img width="1024" height="645" alt="image" src="https://github.com/user-attachments/assets/ca7ca4c1-5bba-4f1d-a399-4ce67e7d64aa" />

Principais componentes:

Azure Blob Storage: armazena dados brutos, artefatos de modelo e arquivos de saída (previsões exportadas).​

Azure Functions (Python): expõe endpoints /api/train, /api/test, /api/reset, /api/processamento para treino e /api/dashboard para dashboard, reset e healthcheck da aplicação.​

Azure SQL Database: armazena versões de modelo (model_versions), execuções (executions), métricas (RMSE, MSE, R²) e logs.​

Dashboard web: consulta diretamente o banco SQL e/ou arquivos no Blob para visualização de previsões e métricas.​

Docker + docker-compose: padroniza o ambiente local de desenvolvimento e execução da Azure Function.​

GitHub Actions: provê pipeline de CI/CD para build, testes e deploy automático da Azure Function App.​

Fluxo resumido:

Upload de base de treino em CSV para o Blob.

/api/train: leitura do CSV, normalização min-max, treino da regressão linear, cálculo de RMSE, MSE e R², salvamento do modelo (.joblib) no Blob e registro no SQL.​

/api/test: carregamento do modelo, aplicação do mesmo pré-processamento, geração de previsões, gravação de CSV de saída no Blob e registro de execução e métricas no SQL.​

/api/reset: recriação/limpeza da tabela de modelos para recomeçar o histórico.​

/api/dashboard: Gera os dashboard para visualização.

Demonstração

Capturas de tela do dashboard:

<img width="1847" height="912" alt="Captura de tela 2025-11-26 180758" src="https://github.com/user-attachments/assets/c14ff066-766d-41dc-981d-5ebfda2a6582" />

<img width="1850" height="910" alt="image" src="https://github.com/user-attachments/assets/b97de55a-a09c-4bad-88a2-9f0f12a0d152" />

Gráfico comparando valores reais × previstos da série temporal.​

Visualização do histórico de RMSE, MSE e R² por versão de modelo.​

Painel com filtro por versão do modelo e por período de tempo.​

Link para vídeo de demo:

Referências
Documentação Azure Functions Python.​ 
https://learn.microsoft.com/pt-br/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1

SDK do Azure Blob Storage e Azure SQL Database.​ 
https://learn.microsoft.com/pt-br/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1​

scikit-learn (LinearRegression, métricas de regressão).​ 
https://www.datacamp.com/pt/tutorial/sklearn-linear-regression

pandas, joblib, SQLAlchemy para manipulação de dados, serialização de modelos e acesso ao banco.​ 
https://www.datacamp.com/pt/tutorial/sklearn-linear-regression

Materiais da disciplina e documentação institucional do curso para definição do escopo acadêmico do projeto.​ 
https://ufape.edu.br/sites/default/files/2025-06/PPC%20perfil%203-2024%20de%20BCC.pdf
