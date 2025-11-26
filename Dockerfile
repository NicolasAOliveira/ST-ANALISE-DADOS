FROM mcr.microsoft.com/azure-functions/python:4-python3.10

# Força a versão exata do Python
# A imagem acima já usa Python 3.10, mas aqui garantimos versão mínima
RUN python --version

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

COPY . /home/site/wwwroot

RUN pip install --no-cache-dir -r /home/site/wwwroot/requirements.txt
