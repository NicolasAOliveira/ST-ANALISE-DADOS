import os
import io
import logging
import json
from datetime import datetime

import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
import plotly.graph_objs as go
import plotly.io as pio

from sqlalchemy import create_engine, text
from azure.storage.blob import BlobServiceClient
import joblib

# Optional encryption with Fernet
try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except Exception:
    HAS_FERNET = False

# configuration from env
BLOB_CONTAINER_RAW = os.environ.get("BLOB_CONTAINER_RAW", "dados")
BLOB_CONTAINER_MODELS = os.environ.get("BLOB_CONTAINER_MODELS", "models")
BLOB_CONTAINER_OUTPUTS = os.environ.get("BLOB_CONTAINER_OUTPUTS", "outputs")
ENCRYPTION_KEY = os.environ.get("FILE_ENCRYPTION_KEY")  # optional Fernet key


def get_blob_service():
    conn = os.environ["STORAGE_CONNECTION_STRING"]
    return BlobServiceClient.from_connection_string(conn)


def upload_blob_bytes(container_name: str, blob_name: str, data: bytes, encrypt: bool = False):
    bs = get_blob_service()
    container = bs.get_container_client(container_name)
    try:
        container.create_container()
    except Exception:
        pass
    if encrypt and ENCRYPTION_KEY and HAS_FERNET:
        f = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
        data = f.encrypt(data)
    blob_client = container.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)
    return f"{container_name}/{blob_name}"


def download_blob_bytes(container_name: str, blob_name: str, decrypt: bool = False) -> bytes:
    bs = get_blob_service()
    container = bs.get_container_client(container_name)
    blob_client = container.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()
    if decrypt and ENCRYPTION_KEY and HAS_FERNET:
        f = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
        stream = f.decrypt(stream)
    return stream


def ensure_model_version(engine, model_name: str, parameters: str, created_by: str = "azure-function"):
    """
    Insert a new model_versions row and return new id (incremental version).
    Use OUTPUT INSERTED.id for Azure SQL.
    """
    with engine.begin() as conn:
        version_str = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
        # INSERT + OUTPUT for ID
        insert_stmt = text("""
            INSERT INTO model_versions (model_name, version, parameters, created_by, created_at)
            OUTPUT INSERTED.id
            VALUES (:model_name, :version, :parameters, :created_by, GETDATE())
        """)
        res = conn.execute(insert_stmt, {
            "model_name": model_name,
            "version": version_str,
            "parameters": parameters,
            "created_by": created_by
        })
        row = res.fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])
        row2 = conn.execute(text("SELECT TOP 1 id FROM model_versions ORDER BY id DESC")).fetchone()
        return int(row2[0]) if row2 and row2[0] is not None else 1


def create_execution(engine, dataset_id: int, description: str, status: str = "pending", created_by: str = "azure-function"):
    with engine.begin() as conn:
        insert_stmt = text("""
            INSERT INTO executions (dataset_id, description, status, created_by, started_at, created_at)
            OUTPUT INSERTED.id
            VALUES (:dataset_id, :description, :status, :created_by, GETDATE(), GETDATE())
        """)
        res = conn.execute(insert_stmt, {
            "dataset_id": dataset_id,
            "description": description,
            "status": status,
            "created_by": created_by
        })
        row = res.fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])
        return None


def update_execution_status(engine, execution_id: int, status: str):
    with engine.begin() as conn:
        conn.execute(text("UPDATE executions SET status = :status, ended_at = GETDATE(), updated_at = GETDATE() WHERE id = :id"),
                     {"status": status, "id": execution_id})


def insert_model_metrics(engine, model_version_id, execution_id, r2, mse, mae, rmse, created_by="azure-function"):
    with engine.begin() as conn:
        stmt = text("""
            INSERT INTO model_metrics (model_version_id, execution_id, r2_score, mse, mae, rmse, created_by, created_at)
            VALUES (:mv, :ex, :r2, :mse, :mae, :rmse, :created_by, GETDATE())
        """)
        conn.execute(stmt, {"mv": model_version_id, "ex": execution_id, "r2": r2, "mse": mse, "mae": mae, "rmse": rmse, "created_by": created_by})


def insert_analysis_results(engine, execution_id, df_pred: pd.DataFrame, created_by="azure-function"):
    with engine.begin() as conn:
        for _, row in df_pred.iterrows():
            stmt = text("""
                INSERT INTO analysis_results (execution_id, category, product, average_price, predicted_price, reference_date, created_by, created_at)
                VALUES (:execution_id, :category, :product, :average_price, :predicted_price, :reference_date, :created_by, GETDATE())
            """)
            conn.execute(stmt, {
                "execution_id": execution_id,
                "category": row.get("category"),
                "product": row.get("product"),
                "average_price": float(row.get("average_price")) if pd.notna(row.get("average_price")) else None,
                "predicted_price": float(row.get("predicted_price")) if pd.notna(row.get("predicted_price")) else None,
                "reference_date": row.get("reference_date"),
                "created_by": created_by
            })


def register_dataset_if_not_exists(engine, dataset_name: str, source_path: str, created_by: str = "azure-function"):
    """
    Insert a datasets row (if not exists) and return dataset id.
    """
    with engine.begin() as conn:
        r = conn.execute(text("SELECT TOP 1 id FROM datasets WHERE dataset_name = :name ORDER BY id DESC"), {"name": dataset_name}).fetchone()
        if r and r[0] is not None:
            return int(r[0])
        insert_stmt = text("""
            INSERT INTO datasets (dataset_name, source_path, created_by, created_at)
            OUTPUT INSERTED.id
            VALUES (:name, :path, :created_by, GETDATE())
        """)
        res = conn.execute(insert_stmt, {"name": dataset_name, "path": source_path, "created_by": created_by})
        row = res.fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])
        r2 = conn.execute(text("SELECT TOP 1 id FROM datasets ORDER BY id DESC")).fetchone()
        return int(r2[0]) if r2 and r2[0] is not None else 1


def insert_dataset_records(engine, dataset_id: int, df: pd.DataFrame, created_by: str = "azure-function"):
    col_variants = {
        "time_5": ["time-5", "time_5", "time5", "time5?"],
        "time_4": ["time-4", "time_4", "time4"],
        "time_3": ["time-3", "time_3", "time3"],
        "time_2": ["time-2", "time_2", "time2"],
        "time_1": ["time-1", "time_1", "time1"],
        "time":   ["time", "target"]
    }
    rows = []
    for _, r in df.iterrows():
        rec = {
            "dataset_id": dataset_id,
            "time_5": None,
            "time_4": None,
            "time_3": None,
            "time_2": None,
            "time_1": None,
            "time": None,
            "created_by": created_by
        }
        for k, variants in col_variants.items():
            for v in variants:
                if v in df.columns:
                    val = r[v]
                    try:
                        rec[k] = float(val) if (pd.notna(val)) else None
                    except Exception:
                        rec[k] = None
                    break
        rows.append(rec)
    with engine.begin() as conn:
        stmt = text("""
            INSERT INTO dataset_records (dataset_id, time_5, time_4, time_3, time_2, time_1, time, created_by, created_at)
            VALUES (:dataset_id, :time_5, :time_4, :time_3, :time_2, :time_1, :time, :created_by, GETDATE())
        """)
        conn.execute(stmt, rows)


def insert_predictions(engine, execution_id: int, preds: np.ndarray, created_by: str = "azure-function"):
    rows = []
    for idx, p in enumerate(preds):
        try:
            val = float(p) if (p is not None and not pd.isna(p)) else None
        except Exception:
            val = None
        rows.append({
            "execution_id": execution_id,
            "predicted_time": val,
            "row_index": int(idx),
            "created_by": created_by
        })
    with engine.begin() as conn:
        stmt = text("""
            INSERT INTO predictions (execution_id, predicted_time, row_index, created_by, created_at)
            VALUES (:execution_id, :predicted_time, :row_index, :created_by, GETDATE())
        """)
        conn.execute(stmt, rows)


app = func.FunctionApp()


@app.route(route="train", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def train(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Train endpoint called")
    try:
        sql_conn = os.environ["SQL_CONNECTION_STRING"]
        try:
            body = req.get_json()
        except Exception:
            body = {}
        blob_name = body.get("blob_name") or req.params.get("blob_name") or "PI_train.csv"
        container = body.get("container") or req.params.get("container") or BLOB_CONTAINER_RAW
        model_name = body.get("model_name") or "linear_regression"
        created_by = body.get("created_by") or "azure-function"
        encrypt_artifacts = bool(body.get("encrypt", False))
        engine = create_engine(sql_conn)
        bs = get_blob_service()
        container_client = bs.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        logging.info(f"Downloading blob {container}/{blob_name}")
        stream = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(stream))
        logging.info(f"CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")

        if "time" not in df.columns and "time-5" not in df.columns:
            if not any([c.lower() == "time" or c.lower().startswith("time") for c in df.columns]):
                return func.HttpResponse("CSV must contain 'time' column (target) or lag columns.", status_code=400)

        dataset_row_id = register_dataset_if_not_exists(engine, blob_name, f"{container}/{blob_name}", created_by=created_by)
        logging.info(f"Using dataset_id={dataset_row_id}")
        execution_id = create_execution(engine, dataset_row_id, f"Training {model_name}", status="running", created_by=created_by)
        if execution_id is None:
            return func.HttpResponse("Failed to create execution record", status_code=500)
        logging.info(f"Created execution_id={execution_id}")

        try:
            insert_dataset_records(engine, dataset_row_id, df, created_by=created_by)
            logging.info("Inserted dataset_records successfully.")
        except Exception as e:
            logging.exception("Failed to insert dataset_records")
            update_execution_status(engine, execution_id, "failed")
            return func.HttpResponse(f"Failed to insert dataset_records: {str(e)}", status_code=500)

        rename_map = {}
        for i in range(1, 6):
            hyphen = f"time-{i}"
            unders = f"time_{i}"
            target_col = f"time_{i}"
            if hyphen in df.columns:
                rename_map[hyphen] = target_col
            elif unders in df.columns:
                rename_map[unders] = target_col
        if "time" not in df.columns and "time" in rename_map.values():
            pass
        if rename_map:
            df = df.rename(columns=rename_map)

        if "time" not in df.columns:
            update_execution_status(engine, execution_id, "failed")
            return func.HttpResponse("After normalization the CSV does not have a 'time' column.", status_code=400)
        lag_cols = [c for c in ["time_1", "time_2", "time_3", "time_4", "time_5"] if c in df.columns]
        if len(lag_cols) == 0:
            update_execution_status(engine, execution_id, "failed")
            return func.HttpResponse("CSV does not contain lag columns (time_1..time_5). Cannot train.", status_code=400)

        y = df["time"].astype(float).copy()
        X = df[lag_cols].astype(float).copy()

        N_SPLITS = 5 if len(X) > 5 else max(1, len(X) - 1)
        tss = TimeSeriesSplit(n_splits=N_SPLITS)
        rmse_folds = []
        r2_folds = []

        for train_idx, test_idx in tss.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            lr = LinearRegression()
            lr.fit(X_train_s, y_train)
            preds = lr.predict(X_test_s)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            rmse_folds.append(rmse)
            r2_folds.append(r2)

        mean_rmse = float(np.mean(rmse_folds)) if len(rmse_folds) > 0 else 0.0
        mean_r2 = float(np.mean(r2_folds)) if len(r2_folds) > 0 else 0.0
        mean_mse = float(mean_rmse ** 2)
        mean_mae = 0.0

        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        lr_final = LinearRegression()
        lr_final.fit(X_scaled, y)
        coefs_scaled = lr_final.coef_
        scales = scaler_final.scale_
        means = scaler_final.mean_
        coefs_original = coefs_scaled / scales
        intercept_original = lr_final.intercept_ - np.sum((coefs_scaled * means) / scales)

        parameters = "StandardScaler + LinearRegression"
        model_version_id = ensure_model_version(engine, model_name, parameters, created_by=created_by)

        try:
            insert_model_metrics(engine, model_version_id, execution_id, mean_r2, mean_mse, mean_mae, mean_rmse, created_by=created_by)
        except Exception as e:
            logging.exception("Failed to insert model_metrics")
            update_execution_status(engine, execution_id, "failed")
            return func.HttpResponse(f"Failed to insert model_metrics: {str(e)}", status_code=500)

        try:
            preds_full = lr_final.predict(X_scaled)
            insert_predictions(engine, execution_id, preds_full, created_by=created_by)
            logging.info("Inserted predictions into SQL table.")
        except Exception as e:
            logging.exception("Failed to insert predictions")
            update_execution_status(engine, execution_id, "failed")
            return func.HttpResponse(f"Failed to insert predictions: {str(e)}", status_code=500)

        df_out = X.copy()
        df_out["time"] = y
        df_out["prediction"] = preds_full
        df_out["residual"] = df_out["time"] - df_out["prediction"]
        out_csv = df_out.to_csv(index=False).encode("utf-8")
        pred_blob_name = f"execution_{execution_id}_predictions.csv"
        upload_blob_bytes(BLOB_CONTAINER_OUTPUTS, pred_blob_name, out_csv, encrypt=encrypt_artifacts)

        artifact_buf = io.BytesIO()
        joblib.dump({"model": lr_final, "scaler": scaler_final, "meta": {
            "model_version_id": model_version_id,
            "created_at": datetime.utcnow().isoformat(),
            "coefs_original": coefs_original.tolist(),
            "intercept_original": float(intercept_original)
        }}, artifact_buf)
        artifact_buf.seek(0)
        model_blob_name = f"model_v{model_version_id}.joblib"
        upload_blob_bytes(BLOB_CONTAINER_MODELS, model_blob_name, artifact_buf.read(), encrypt=encrypt_artifacts)

        try:
            needed_cols = {"category", "product", "average_price"}
            if needed_cols.issubset(set(df_out.columns)):
                df_pred_rows = pd.DataFrame({
                    "category": df_out["category"],
                    "product": df_out["product"],
                    "average_price": df_out["average_price"],
                    "predicted_price": df_out["prediction"],
                    "reference_date": df_out.get("reference_date", pd.NaT)
                })
                insert_analysis_results(engine, execution_id, df_pred_rows, created_by=created_by)
        except Exception as e:
            logging.warning("Skipping analysis_results insert: " + str(e))

        update_execution_status(engine, execution_id, "completed")

        resp = {
            "message": "Training completed",
            "model_version_id": model_version_id,
            "execution_id": execution_id,
            "predictions_blob": f"{BLOB_CONTAINER_OUTPUTS}/{pred_blob_name}",
            "model_artifact_blob": f"{BLOB_CONTAINER_MODELS}/{model_blob_name}",
            "metrics": {
                "rmse": mean_rmse,
                "mse": mean_mse,
                "r2": mean_r2
            }
        }
        return func.HttpResponse(
            json.dumps(resp),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.exception("Unhandled error in /train endpoint")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)


@app.route(route="Processamento", auth_level=func.AuthLevel.ANONYMOUS)
def Processamento(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')
    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
    
import plotly.graph_objs as go
import plotly.io as pio

@app.route(route="dashboard", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    try:
        sql_conn = os.environ["SQL_CONNECTION_STRING"]
        engine = create_engine(sql_conn)

        n_points = 264
        indices = list(range(n_points))

        with engine.connect() as conn:
            # Dados de série e previsões
            df_true = pd.read_sql("SELECT * FROM dataset_records ORDER BY id", conn).reset_index(drop=True)
            df_pred = pd.read_sql("SELECT * FROM predictions ORDER BY id", conn).reset_index(drop=True)

            # Métricas do último execution/model (ajuste se quiser filtrar por id específico)
            df_metrics = pd.read_sql(
                "SELECT TOP 1 * FROM model_metrics ORDER BY created_at DESC",
                conn
            )

        # Série real e prevista
        y_true = df_true['time'].values[:n_points]
        y_pred = df_pred['predicted_time'].values[:n_points]

        # Se não encontrar métricas, define como None
        if not df_metrics.empty:
            rmse_val = float(df_metrics['rmse'].iloc[0])
            mse_val = float(df_metrics['mse'].iloc[0])
            r2_val = float(df_metrics['r2_score'].iloc[0])
        else:
            rmse_val = mse_val = r2_val = None

        # Gráfico 1: Real vs Previsto
        fig_series = go.Figure()
        fig_series.add_trace(go.Scatter(x=indices, y=y_true, mode='lines', name='Valor Real (time)'))
        fig_series.add_trace(go.Scatter(x=indices, y=y_pred, mode='lines', name='Previsto (predicted_time)'))
        fig_series.update_layout(
            title='Regressão Linear: Real vs Previsto',
            xaxis_title='Índice (0 a 263)',
            yaxis_title='Valor'
        )

        # Gráfico 2: Resíduos
        residuals = y_true - y_pred
        fig_residuos = go.Figure()
        fig_residuos.add_trace(go.Scatter(
            x=indices,
            y=residuals,
            mode='markers',
            name='Resíduo (y - y_pred)',
            marker=dict(size=6, color='royalblue')
        ))
        fig_residuos.add_shape(type='line', x0=0, x1=n_points-1, y0=0, y1=0, line=dict(color='black', dash='dash'))
        fig_residuos.update_layout(
            title='Resíduos do Modelo (treinado em todo o dataset)',
            xaxis_title='Índice (0 a 263)',
            yaxis_title='Resíduo (y - y_pred)'
        )

        graph_html_1 = pio.to_html(fig_series, full_html=False)
        graph_html_2 = pio.to_html(fig_residuos, full_html=False)

        # Cards de métricas (formatando com 2 casas decimais se houver valor)
        rmse_txt = f"{rmse_val:.2f}" if rmse_val is not None else "N/A"
        mse_txt  = f"{mse_val:.2f}"  if mse_val  is not None else "N/A"
        r2_txt   = f"{r2_val:.3f}"   if r2_val   is not None else "N/A"

        dashboard_html = f"""
        <html>
        <head>
            <title>Dashboard de Regressão</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                    margin: 0;
                    padding: 0;
                }}
                .header {{
                    background-color: #0f4c81;
                    color: white;
                    padding: 20px 40px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 26px;
                }}
                .metrics-container {{
                    display: flex;
                    gap: 20px;
                    padding: 20px 40px;
                    flex-wrap: wrap;
                }}
                .metric-card {{
                    background-color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    min-width: 180px;
                }}
                .metric-label {{
                    font-size: 13px;
                    color: #666;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }}
                .metric-value {{
                    font-size: 22px;
                    font-weight: bold;
                    margin-top: 5px;
                }}
                .content-section {{
                    padding: 10px 40px 30px 40px;
                }}
                h2 {{
                    font-size: 20px;
                    color: #333;
                }}
                .chart-box {{
                    background-color: white;
                    padding: 10px 15px 20px 15px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                footer {{
                    text-align: center;
                    padding: 10px;
                    font-size: 12px;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dashboard: Análise de Regressão Linear</h1>
            </div>

            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{rmse_txt}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MSE</div>
                    <div class="metric-value">{mse_txt}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">R²</div>
                    <div class="metric-value">{r2_txt}</div>
                </div>
            </div>

            <div class="content-section">
                <div class="chart-box">
                    <h2>Comparação: Valor Real vs Previsto</h2>
                    {graph_html_1}
                </div>

                <div class="chart-box">
                    <h2>Resíduos do Modelo</h2>
                    {graph_html_2}
                </div>
            </div>

            <footer>
                Dashboard interativo gerado em Python / Plotly / Azure Functions.
            </footer>
        </body>
        </html>
        """

        return func.HttpResponse(dashboard_html, mimetype="text/html")
    except Exception as e:
        logging.exception("Erro ao gerar dashboard")
        return func.HttpResponse(f"<h1>Erro: {str(e)}</h1>", mimetype="text/html", status_code=500)