# 🧬 The Clinical Oracle
### NIH Clinical Intelligence System — Agent RAG pour protocoles d'essais cliniques, avec pipeline MLOps complet

> Interroger en langage naturel 20 protocoles NIH denses et obtenir des réponses sourcées, évaluées automatiquement, avec ré-indexation et redéploiement continus.

---

## 🎯 Ce que fait le projet

**The Clinical Oracle** est un agent RAG (Retrieval-Augmented Generation) construit sur des PDFs de protocoles d'essais cliniques NIH. Au-delà du chatbot, le projet implémente une **chaîne MLOps complète en 3 pipelines automatisés** :

- ✅ Réponses ancrées exclusivement dans le contexte récupéré (anti-hallucination)
- ✅ Citations des sources avec scores de similarité par chunk
- ✅ Évaluation qualité continue (LLM-as-Judge + Evidently AI)
- ✅ **Ré-indexation automatique** dès qu'un document est ajouté/modifié
- ✅ **CI/CD** : aucun déploiement ne passe sans validation qualité
- ✅ **Monitoring de production** avec détection de dérive

---

## 🏗️ Architecture — les 3 pipelines

### 1️⃣ Pipeline d'ajout de documents (`dags/document_ingestion_dag.py`)

```
PDF → Streamlit (upload) → S3 (stockage brut)
                              │
                    Orchestrator : Airflow
        ETL : parsing (Unstructured) → chunking → embedding → base vectorielle
                              │
                          ChromaDB
```
Déclenché automatiquement (S3KeySensor, toutes les 15 min) dès qu'un PDF est ajouté ou modifié : aucune ré-indexation manuelle nécessaire.

### 2️⃣ Pipeline d'utilisation et monitoring (`monitoring/`, `dags/monitoring_dag.py`)

```
Streamlit → [Question] → Reformulation → Similarité (ChromaDB) → Réponse (Mistral AI) → Streamlit
                                                        │
                                    stockage logs + question + réponses + chunks (S3)
                                                        │
                                     Airflow (quotidien) → Evidently AI
                                     vérifie qualité + surveille dérive/erreurs/bugs
```

### 3️⃣ Pipeline déploiement et qualité (`.github/workflows/ci-cd.yml`)

```
push (code, config, requirements) → GitHub
                                        │
                    CI : test technique du code et infrastructure
              jeu de données test (JSON, questions/réponses attendues)
                    évaluation qualitative → pytest + Evidently AI
                                        │  ok
                    CD : déploiement de la nouvelle version
                        Streamlit → Hugging Face Spaces (Docker)
```
Aucun code n'est déployé si les seuils qualité (faithfulness, couverture, score global) ne sont pas atteints.

---

## 📊 Résultats d'évaluation (LLM-as-Judge, `evaluation_report.json`)

| Métrique | Score moyen |
|---|---|
| Faithfulness | 6.6/10 |
| Relevance | 8.6/10 |
| Completeness | 5.0/10 |
| Citation | 10.0/10 |
| **Overall** | **7.55/10** |

Axes d'amélioration identifiés lors de l'évaluation (voir `evaluation_report.json`) : certaines hallucinations sur les valeurs numériques absentes du contexte (sample size, endpoints) — d'où l'ajout d'un test pytest dédié (`test_faithfulness_above_threshold`) qui bloque désormais le déploiement si ce type de régression réapparaît.

---

## 📁 Structure du projet

```
clinical-oracle/
├── app.py                          # Interface Streamlit
├── rag_core.py                     # Logique RAG (retrieval, génération, judge) — testable et réutilisable
├── requirements.txt
├── Dockerfile
├── tests/
│   ├── test_dataset.json           # Jeu de questions/réponses attendues
│   ├── test_rag_quality.py         # pytest : validation qualité offline (LLM-as-Judge + Evidently)
│   └── conftest.py
├── dags/
│   ├── document_ingestion_dag.py   # Pipeline 1 : ré-indexation auto (Airflow)
│   └── monitoring_dag.py           # Pipeline 2 : monitoring qualité en production (Airflow)
├── .github/workflows/
│   └── ci-cd.yml                   # Pipeline 3 : CI (test) + CD (déploiement HF Spaces)
├── evaluation_report.json          # Rapport d'évaluation LLM-as-Judge
├── logo.png
├── archives_oracle.json            # Archives de sessions (auto-généré)
├── docs/                           # PDFs bruts (non versionnés)
├── docs_txt/                       # Textes parsés (auto-généré)
└── chroma_db/                      # Base vectorielle (auto-générée)
```

---

## 🚀 Démarrage local

```bash
git clone https://github.com/YOUR_USERNAME/clinical-oracle.git
cd clinical-oracle
pip install -r requirements.txt
```

Créer un `.env` :
```
MISTRAL_API_KEY=your_mistral_api_key_here
S3_BUCKET=clinical-oracle-docs      # optionnel, pour activer le logging monitoring
```

Placer les PDFs NIH dans `docs/`, exécuter les cellules 1 à 5 du notebook `Clinical_Oracle_FINAL.ipynb` pour construire `chroma_db/`, puis :

```bash
streamlit run app.py
```

### Tests qualité (avant tout déploiement)

```bash
pip install pytest evidently pandas
pytest tests/ -v
```
Le rapport HTML est généré dans `reports/evidently_report.html`.

---

## 🐳 Déploiement Docker / Hugging Face Spaces

```bash
docker build -t clinical-oracle .
docker run -p 8501:8501 \
  -e MISTRAL_API_KEY=your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  clinical-oracle
```

Le déploiement en production est entièrement automatisé par `.github/workflows/ci-cd.yml` : chaque push sur `main` déclenche les tests, puis (si succès) le push vers le Space Hugging Face configuré via les secrets `HF_TOKEN` et `HF_SPACE_REPO`.

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| LLM | Mistral AI (`mistral-small-latest`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Parsing PDF | Unstructured |
| Framework RAG | LangChain |
| UI | Streamlit |
| Stockage documents/logs | AWS S3 |
| Orchestration | Apache Airflow |
| Tests qualité | pytest + Evidently AI (LLM-as-Judge) |
| CI/CD | GitHub Actions |
| Hébergement | Hugging Face Spaces (Docker) |

---

## 📝 .gitignore

```
.env
chroma_db/
docs/
docs_txt/
archives_oracle.json
reports/
__pycache__/
*.pyc
```
