"""
test_rag_quality.py
====================
Étage "Test MLOPS validation qualité RAG (offline)" du pipeline CI/CD.

Déroulé :
1. Charge le jeu de questions/réponses attendues (test_dataset.json)
2. Interroge le pipeline RAG réel (rag_core.run_rag) sur chaque question
3. Note chaque réponse avec le LLM-as-Judge (rag_core.run_judge)
4. Construit un rapport Evidently AI (qualité + seuils) -> reports/evidently_report.html
5. Le job GitHub Actions "test" échoue si les seuils qualité ne sont pas atteints,
   ce qui bloque le déploiement (CD) vers Hugging Face Spaces.

Prérequis : MISTRAL_API_KEY doit être présent en variable d'environnement
(ou en secret GitHub Actions), ainsi qu'une base ChromaDB déjà indexée
(chroma_db/) accessible dans le répertoire de travail.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from rag_core import load_oracle, run_rag, run_judge

DATASET_PATH = Path(__file__).parent / "test_dataset.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Seuils minimums acceptés avant blocage du déploiement
MIN_OVERALL = 5.0
MIN_FAITHFULNESS = 4.0
MIN_KEYWORD_HIT_RATE = 0.5  # au moins 50% des mots-clés attendus doivent apparaître


@pytest.fixture(scope="session")
def oracle():
    """Charge une seule fois le vectorstore + LLM pour toute la session de tests."""
    return load_oracle()


@pytest.fixture(scope="session")
def dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def evaluation_results(oracle, dataset):
    """Exécute le pipeline RAG complet sur tout le jeu de test une seule fois,
    et partage les résultats entre tous les tests (évite de rappeler l'API)."""
    vectorstore, llm, embeddings = oracle
    results = []
    for item in dataset:
        rag_result = run_rag(item["question"], k=12, vectorstore=vectorstore, llm=llm, embeddings=embeddings)
        judge_scores = run_judge(item["question"], rag_result.get("context_preview", ""), rag_result["answer"], llm)
        hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in rag_result["answer"].lower())
        keyword_hit_rate = hits / len(item["expected_keywords"]) if item["expected_keywords"] else 1.0
        results.append({
            "question": item["question"],
            "answer": rag_result["answer"],
            "n_sources": len(rag_result["sources_details"]),
            "keyword_hit_rate": keyword_hit_rate,
            **judge_scores,
        })
    return results


# ---------------------------------------------------------------------------
# Tests unitaires par question
# ---------------------------------------------------------------------------

def test_pipeline_returns_sources(evaluation_results):
    """Chaque question doit retourner au moins une source documentaire."""
    for r in evaluation_results:
        assert r["n_sources"] > 0, f"Aucune source retrouvée pour: {r['question']}"


def test_faithfulness_above_threshold(evaluation_results):
    """Aucune réponse ne doit halluciner de façon flagrante (faithfulness < seuil)."""
    failing = [r for r in evaluation_results if r["faithfulness"] < MIN_FAITHFULNESS]
    assert not failing, f"Réponses avec faithfulness insuffisante: {[r['question'] for r in failing]}"


def test_overall_quality_above_threshold(evaluation_results):
    """Score global LLM-as-Judge moyen doit rester au-dessus du seuil qualité."""
    avg_overall = sum(r["overall"] for r in evaluation_results) / len(evaluation_results)
    assert avg_overall >= MIN_OVERALL, f"Score qualité moyen trop bas: {avg_overall}/10"


def test_keyword_coverage(evaluation_results):
    """Vérifie que les réponses contiennent les éléments cliniques attendus."""
    avg_hit_rate = sum(r["keyword_hit_rate"] for r in evaluation_results) / len(evaluation_results)
    assert avg_hit_rate >= MIN_KEYWORD_HIT_RATE, f"Couverture mots-clés trop faible: {avg_hit_rate:.0%}"


# ---------------------------------------------------------------------------
# Rapport Evidently AI (qualité + dérive) — exporté en HTML pour la CI
# ---------------------------------------------------------------------------

def test_generate_evidently_report(evaluation_results):
    """Génère un rapport Evidently AI avec tests de seuils sur les métriques
    du LLM-as-Judge. Le rapport HTML est archivé comme artefact GitHub Actions."""
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnValueMin, TestColumnValueMean

    df = pd.DataFrame(evaluation_results)

    suite = TestSuite(tests=[
        TestColumnValueMean(column_name="overall", gte=MIN_OVERALL),
        TestColumnValueMin(column_name="faithfulness", gte=MIN_FAITHFULNESS),
        TestColumnValueMean(column_name="keyword_hit_rate", gte=MIN_KEYWORD_HIT_RATE),
    ])
    suite.run(reference_data=None, current_data=df)
    suite.save_html(str(REPORTS_DIR / "evidently_report.html"))

    result_json = suite.as_dict()
    with open(REPORTS_DIR / "evidently_report.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, default=str)

    assert result_json["summary"]["all_passed"], "Evidently a détecté un échec de seuil qualité — voir reports/evidently_report.html"
