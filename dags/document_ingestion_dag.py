"""
document_ingestion_dag.py
==========================
Correspond au schéma "Pipeline ajout documents" :

  PDF -> Streamlit (upload) -> S3 (stockage brut)
                                  |
                          Orchestrator (Airflow)
        ETL : parsing PDF -> chunking -> embedding -> base vectorielle
                                  |
                              ChromaDB

Ce DAG est déclenché :
- automatiquement, via un S3KeySensor qui détecte tout nouveau/modifié PDF
  dans le bucket (docs/ prefix) ;
- ou manuellement via l'API Airflow depuis Streamlit après un upload.

Objectif : si un document est ajouté ou modifié, toute la chaîne RAG
(vector store) est relancée automatiquement, sans réindexation manuelle.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

S3_BUCKET = "clinical-oracle-docs"
S3_PDF_PREFIX = "docs/"
S3_TXT_PREFIX = "docs_txt/"
LOCAL_TMP_DIR = "/tmp/clinical_oracle_ingestion"
CHROMA_DIR = "/opt/airflow/data/chroma_db"

default_args = {
    "owner": "clinical-oracle",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def list_new_or_modified_pdfs(**context):
    """Compare le contenu de S3 (docs/) avec les fichiers déjà indexés
    (suivi via un fichier manifest.json sur S3) pour ne traiter que le delta."""
    import json
    hook = S3Hook(aws_conn_id="aws_default")
    keys = hook.list_keys(bucket_name=S3_BUCKET, prefix=S3_PDF_PREFIX) or []
    pdf_keys = [k for k in keys if k.endswith(".pdf")]

    manifest = {}
    if hook.check_for_key(f"{S3_TXT_PREFIX}manifest.json", bucket_name=S3_BUCKET):
        raw = hook.read_key(f"{S3_TXT_PREFIX}manifest.json", bucket_name=S3_BUCKET)
        manifest = json.loads(raw)

    to_process = []
    for key in pdf_keys:
        etag = hook.get_key(key, bucket_name=S3_BUCKET).e_tag
        if manifest.get(key) != etag:
            to_process.append({"key": key, "etag": etag})

    context["ti"].xcom_push(key="to_process", value=to_process)
    print(f"{len(to_process)} document(s) nouveaux/modifiés à traiter sur {len(pdf_keys)} au total")
    return to_process


def parse_pdfs(**context):
    """Étape 1 (ETL) : parsing des PDFs via `unstructured` -> fichiers .txt structurés."""
    import os
    from unstructured.partition.pdf import partition_pdf

    to_process = context["ti"].xcom_pull(key="to_process", task_ids="list_new_or_modified_pdfs")
    hook = S3Hook(aws_conn_id="aws_default")
    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)

    parsed_paths = []
    for item in to_process:
        key = item["key"]
        local_pdf = f"{LOCAL_TMP_DIR}/{os.path.basename(key)}"
        hook.get_key(key, bucket_name=S3_BUCKET).download_file(local_pdf)

        elements = partition_pdf(filename=local_pdf, strategy="fast", infer_table_structure=True)
        text = "\n\n".join(
            f"=== TABLE ===\n{el.metadata.text_as_html}" if el.category == "Table" else el.text
            for el in elements if el.category not in ("Header", "Footer", "PageBreak")
        )
        txt_name = os.path.basename(key).replace(".pdf", ".txt")
        local_txt = f"{LOCAL_TMP_DIR}/{txt_name}"
        with open(local_txt, "w", encoding="utf-8") as f:
            f.write(text)

        hook.load_file(local_txt, key=f"{S3_TXT_PREFIX}{txt_name}", bucket_name=S3_BUCKET, replace=True)
        parsed_paths.append(local_txt)

    context["ti"].xcom_push(key="parsed_paths", value=parsed_paths)


def chunk_and_embed_and_index(**context):
    """Étapes 2-3-4 (ETL) : chunking -> embedding -> upsert dans ChromaDB."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    parsed_paths = context["ti"].xcom_pull(key="parsed_paths", task_ids="parse_pdfs")
    if not parsed_paths:
        print("Aucun nouveau document à indexer.")
        return

    docs = []
    for path in parsed_paths:
        docs.extend(TextLoader(path, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    print(f"{len(chunks)} chunks indexés dans ChromaDB ({CHROMA_DIR})")


def update_manifest(**context):
    """Marque les documents traités comme indexés (évite un retraitement au run suivant)."""
    import json
    to_process = context["ti"].xcom_pull(key="to_process", task_ids="list_new_or_modified_pdfs")
    hook = S3Hook(aws_conn_id="aws_default")

    manifest = {}
    if hook.check_for_key(f"{S3_TXT_PREFIX}manifest.json", bucket_name=S3_BUCKET):
        manifest = json.loads(hook.read_key(f"{S3_TXT_PREFIX}manifest.json", bucket_name=S3_BUCKET))

    for item in to_process:
        manifest[item["key"]] = item["etag"]

    hook.load_string(json.dumps(manifest, indent=2), key=f"{S3_TXT_PREFIX}manifest.json",
                      bucket_name=S3_BUCKET, replace=True)


with DAG(
    dag_id="clinical_oracle_document_ingestion",
    description="Ré-indexe automatiquement ChromaDB dès qu'un PDF est ajouté ou modifié sur S3",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="*/15 * * * *",  # vérifie S3 toutes les 15 min (+ déclenchement manuel possible)
    catchup=False,
    tags=["clinical-oracle", "ingestion", "rag"],
) as dag:

    wait_for_new_pdf = S3KeySensor(
        task_id="wait_for_new_pdf",
        bucket_name=S3_BUCKET,
        bucket_key=f"{S3_PDF_PREFIX}*.pdf",
        wildcard_match=True,
        aws_conn_id="aws_default",
        timeout=60,
        soft_fail=True,  # ne bloque pas le DAG s'il n'y a rien de nouveau
    )

    list_new = PythonOperator(task_id="list_new_or_modified_pdfs", python_callable=list_new_or_modified_pdfs)
    parse = PythonOperator(task_id="parse_pdfs", python_callable=parse_pdfs)
    index = PythonOperator(task_id="chunk_and_embed_and_index", python_callable=chunk_and_embed_and_index)
    manifest = PythonOperator(task_id="update_manifest", python_callable=update_manifest)

    wait_for_new_pdf >> list_new >> parse >> index >> manifest
