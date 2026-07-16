"""
monitoring_dag.py
==================
Correspond au schéma "Utilisation et monitoring" :

  Streamlit -> AI Agent (rewrite, similarité, réponse) -> logs (question+réponses+chunks) -> S3
                                                                                                |
                                                                                    Airflow (ce DAG, planifié)
                                                                                                |
                                                                                        Evidently AI
                                                                              (vérifie qualité + surveille dérive/erreurs)

Ce DAG tourne quotidiennement, agrège les logs de production stockés sur S3
par l'application Streamlit (chaque requête utilisateur + réponse + chunks
retrouvés + scores du judge), et génère un rapport Evidently AI comparant
la période courante à une période de référence. En cas de dérive
(baisse de faithfulness, hausse des réponses "non trouvé", etc.), une
alerte est déclenchée.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

S3_BUCKET = "clinical-oracle-docs"
LOGS_PREFIX = "logs/"
REFERENCE_KEY = "monitoring/reference_window.json"
REPORT_PREFIX = "monitoring/reports/"

DRIFT_ALERT_THRESHOLDS = {
    "faithfulness_drop": 1.5,   # alerte si la faithfulness moyenne chute de plus de 1.5 pts
    "no_answer_rate_max": 0.15,  # alerte si >15% des requêtes retournent "No relevant documents found"
}

default_args = {
    "owner": "clinical-oracle",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def collect_production_logs(**context):
    """Récupère les logs de la fenêtre courante (dernières 24h) depuis S3."""
    import json
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    hook = S3Hook(aws_conn_id="aws_default")
    keys = hook.list_keys(bucket_name=S3_BUCKET, prefix=LOGS_PREFIX) or []
    cutoff = datetime.utcnow() - timedelta(days=1)

    logs = []
    for key in keys:
        obj = hook.get_key(key, bucket_name=S3_BUCKET)
        if obj.last_modified.replace(tzinfo=None) < cutoff:
            continue
        record = json.loads(hook.read_key(key, bucket_name=S3_BUCKET))
        logs.append(record)

    context["ti"].xcom_push(key="logs", value=logs)
    print(f"{len(logs)} interactions de production collectées (fenêtre 24h)")


def build_evidently_report(**context):
    """Construit un rapport Evidently AI comparant la fenêtre courante à une
    fenêtre de référence, et sauvegarde le rapport HTML + JSON sur S3."""
    import json
    import pandas as pd
    from evidently.report import Report
    from evidently.metrics import ColumnSummaryMetric, ColumnDriftMetric
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    logs = context["ti"].xcom_pull(key="logs", task_ids="collect_production_logs")
    hook = S3Hook(aws_conn_id="aws_default")

    if not logs:
        print("Pas de trafic sur la fenêtre courante — rapport ignoré.")
        return

    current_df = pd.DataFrame(logs)
    current_df["no_answer"] = current_df["answer"].str.contains("No relevant documents found", na=False)

    if hook.check_for_key(REFERENCE_KEY, bucket_name=S3_BUCKET):
        reference_df = pd.DataFrame(json.loads(hook.read_key(REFERENCE_KEY, bucket_name=S3_BUCKET)))
    else:
        reference_df = current_df  # première exécution : la fenêtre courante sert de référence

    report = Report(metrics=[
        ColumnSummaryMetric(column_name="faithfulness"),
        ColumnSummaryMetric(column_name="overall"),
        ColumnDriftMetric(column_name="overall"),
    ])
    report.run(reference_data=reference_df, current_data=current_df)

    report_name = f"{REPORT_PREFIX}report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
    local_path = f"/tmp/{report_name.split('/')[-1]}"
    report.save_html(local_path)
    hook.load_file(local_path, key=report_name, bucket_name=S3_BUCKET, replace=True)

    context["ti"].xcom_push(key="current_summary", value={
        "avg_faithfulness": float(current_df["faithfulness"].mean()),
        "no_answer_rate": float(current_df["no_answer"].mean()),
        "n_requests": len(current_df),
    })
    context["ti"].xcom_push(key="reference_summary", value={
        "avg_faithfulness": float(reference_df["faithfulness"].mean()),
    })


def check_drift_and_alert(**context):
    """Compare les métriques courantes aux seuils et déclenche une alerte si nécessaire."""
    current = context["ti"].xcom_pull(key="current_summary", task_ids="build_evidently_report")
    reference = context["ti"].xcom_pull(key="reference_summary", task_ids="build_evidently_report")

    if not current:
        return

    alerts = []
    if reference and (reference["avg_faithfulness"] - current["avg_faithfulness"]) > DRIFT_ALERT_THRESHOLDS["faithfulness_drop"]:
        alerts.append(f"Faithfulness en baisse: {reference['avg_faithfulness']:.1f} -> {current['avg_faithfulness']:.1f}")
    if current["no_answer_rate"] > DRIFT_ALERT_THRESHOLDS["no_answer_rate_max"]:
        alerts.append(f"Taux de 'no answer' élevé: {current['no_answer_rate']:.0%}")

    if alerts:
        print("⚠️ ALERTE QUALITÉ RAG :")
        for a in alerts:
            print(f"  - {a}")
        # Ici : intégration Slack/email (ex: SlackWebhookOperator) à brancher en aval
        raise ValueError("Dérive détectée — voir logs Airflow pour détails: " + "; ".join(alerts))
    else:
        print(f"Pas de dérive détectée sur {current['n_requests']} requêtes analysées.")


with DAG(
    dag_id="clinical_oracle_monitoring",
    description="Monitoring qualité RAG en production via Evidently AI (dérive, hallucinations, non-réponses)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["clinical-oracle", "monitoring", "evidently"],
) as dag:

    collect = PythonOperator(task_id="collect_production_logs", python_callable=collect_production_logs)
    report = PythonOperator(task_id="build_evidently_report", python_callable=build_evidently_report)
    alert = PythonOperator(task_id="check_drift_and_alert", python_callable=check_drift_and_alert)

    collect >> report >> alert
