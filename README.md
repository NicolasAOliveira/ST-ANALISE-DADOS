# Sistema de Treinamento e Previsão em Nuvem com Regressão Linear  
### Sistema de Análise de Dados

## Equipe

- **Nicolas Alencar de Oliveira** — Matrícula: 23012092 — E-mail: nicolas.ao@puccampinas.edu.br  
- **Vitor Martins Pinto** — Matrícula: 23020961 — E-mail: vitor.mp2@puccampinas.edu.br  
- **Lucas Gaino Sarzedo** — Matrícula: 23008146 — E-mail: Lucas.gs13@puccampinas.edu.br

---

## Descrição Geral

Sistema em nuvem para análise e previsão de séries temporais, com foco em um modelo de regressão linear que prevê a coluna `time` a partir dos cinco valores anteriores (`time1` a `time5`).  

O objetivo é permitir treino remoto, execução em produção, versionamento de modelos e disponibilização de resultados para dashboards no front-end.  

O sistema suporta upload de bases CSV, cálculo de métricas (RMSE, MSE, R²) e histórico completo de execuções e versões do modelo em um ambiente escalável.

---

## Dataset

- **Fonte:** Dataset fornecido pelo professor Fernando da disciplina de Aprendizado Supervisionado, em formato CSV.  
- **Link:** https://puc-campinas.instructure.com/files/3371440/download?download_frd=1  
- **Volume esperado:** Aproximadamente 30 MB no banco de dados SQL.  
- **Licenciamento:** Uso acadêmico para a disciplina, conforme diretrizes do curso e do professor.

O dataset representa uma série temporal não estacionária, com forte tendência, usada para treinar e testar o modelo preditivo da coluna `time`.  

**Esquema principal das colunas:**

- `time` — variável alvo  
- `time1`, `time2`, `time3`, `time4`, `time5` — defasagens temporais usadas como preditores  

---

## Arquitetura da Solução

### Diagrama (Mermaid / Arquitetura)

![Arquitetura da solução](<img width="1024" height="645" alt="image" src="https://github.com/user-attachments/assets/7dd2f7b4-b0b8-4824-8cc0-ed416b275c7a" />)

### Principais Componentes

- **Azure Blob Storage:** Armazena dados brutos, artefatos de modelo e arquivos de saída (previsões exportadas).  
- **Azure Functions (Python):** Expõe endpoints `/api/train`, `/api/test`, `/api/reset`, `/api/processamento` para treino/processamento e `/api/dashboard` para dashboard, reset e healthcheck.  
- **Azure SQL Database:** Armazena versões de modelo (`model_versions`), execuções (`executions`), métricas (RMSE, MSE, R²) e logs.
- **Dashboard web:** Consulta diretamente o banco SQL e/ou arquivos no Blob para visualização de previsões e métricas.  
- **Docker + docker-compose:** Padroniza o ambiente local de desenvolvimento e execução da Azure Function. 
- **GitHub Actions:** Pipeline de CI/CD para build, testes e deploy automático da Azure Function App.

### Fluxo Resumido

1. **Upload** da base de treino em CSV para o Blob Storage.  
2. **`/api/train`**  
   - Leitura do CSV.  
   - Normalização _min-max_.  
   - Treino da regressão linear.  
   - Cálculo de RMSE, MSE e R².  
   - Salvamento do modelo (`.joblib`) no Blob.  
   - Registro da versão do modelo e das métricas no SQL.  
3. **`/api/test`**  
   - Carregamento do modelo treinado.  
   - Aplicação do mesmo pré-processamento.  
   - Geração de previsões.  
   - Gravação do CSV de saída no Blob.  
   - Registro de execução e métricas no SQL.  
4. **`/api/reset`**  
   - Recriação/limpeza da tabela de modelos para reiniciar o histórico.  
5. **`/api/dashboard`**  
   - Geração dos dashboards para visualização de previsões, resíduos e métricas.  

---

## Demonstração

### Capturas de Tela do Dashboard

![Dashboard - Visão geral](https://github.com/user-attachments/assets/c14ff066-766d-41dc-981d-5ebfda2a6582)

![Dashboard - Métricas e previsões](https://github.com/user-attachments/assets/b97de55a-a09c-4bad-88a2-9f0f12a0d152)

**Principais visualizações:**

- Gráfico comparando valores reais × previstos da série temporal.  
- Visualização do histórico de RMSE, MSE e R² por versão de modelo.  
- Painel com filtro por versão do modelo e por período de tempo.  

### Vídeo de Demonstração

> (Inserir aqui o link do vídeo de demo quando estiver disponível.)

---

## Referências

- Documentação Azure Functions (Python).  
  - https://learn.microsoft.com/pt-br/azure/azure-functions/functions-overview  

- SDK do Azure Blob Storage e Azure SQL Database.  
  - https://learn.microsoft.com/pt-br/azure/storage/blobs/  
  - https://learn.microsoft.com/pt-br/azure/azure-sql/  

- scikit-learn (LinearRegression, métricas de regressão).  
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html  

- pandas, joblib, SQLAlchemy para manipulação de dados, serialização de modelos e acesso ao banco.  
  - https://pandas.pydata.org/  
  - https://joblib.readthedocs.io/  
  - https://docs.sqlalchemy.org/  

- Materiais da disciplina e documentação institucional do curso para definição do escopo acadêmico do projeto.  
  - https://ufape.edu.br/sites/default/files/2025-06/PPC%20perfil%203-2024%20de%20BCC.pdf
