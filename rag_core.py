"""
rag_core.py — Logique RAG de The Clinical Oracle, extraite de app.py
======================================================================
Ce module isole toute la logique métier (retrieval, génération, judge)
de l'interface Streamlit. Objectif : la rendre importable et testable
par pytest, et réutilisable par les DAGs Airflow / le monitoring.
"""

import os
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral-small-latest"

combined_prompt = ChatPromptTemplate.from_template("""
You are a Senior Clinical Data Analyst working with NIH clinical trial protocols.
USER QUESTION: {question}
RETRIEVED CONTEXT:
{context}
INSTRUCTIONS:
1. Answer based ONLY on the provided context.
2. Cite the source file name for every specific claim.
3. If the context mentions scores (VAS, Constant-Murley, DASH, SF-36, etc.), extract exact values.
4. Structure your answer with clear bullet points.
5. If information is not found in the context, explicitly state it.
6. End with a confidence score (0-100%).
ANSWER:
""")

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of RAG systems for clinical research.
Evaluate the following RAG response. Respond ONLY with valid JSON, no explanation outside the JSON.
QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}
Return ONLY this JSON:
{{"faithfulness": <0-10>, "relevance": <0-10>, "completeness": <0-10>, "citation": <0-10>, "feedback": "<one sentence>"}}
""")


def load_oracle(chroma_dir: str = CHROMA_DIR):
    """Charge embeddings + vectorstore + LLM. Un seul point d'entrée
    utilisé par l'app Streamlit, les tests, et les DAGs Airflow."""
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=chroma_dir, embedding_function=emb)
    llm = ChatMistralAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.environ.get("MISTRAL_API_KEY"),
    )
    return vs, llm, emb


def run_rag(question: str, k: int, vectorstore, llm, embeddings) -> dict:
    """Pipeline retrieval + génération. Retourne réponse + sources + contexte."""
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    docs = [r[0] for r in docs_with_scores]
    l2_scores = [r[1] for r in docs_with_scores]

    if not docs:
        return {"answer": "No relevant documents found.", "sources_details": [],
                "query_used": question, "context_preview": ""}

    query_vec = np.array(embeddings.embed_query(question)).reshape(1, -1)
    doc_vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    cos_scores = cosine_similarity(query_vec, doc_vecs)[0]

    context = ""
    sources_info = []
    for i, (doc, cos, l2) in enumerate(zip(docs, cos_scores, l2_scores)):
        src = Path(doc.metadata.get("source", "Unknown")).name
        quality = "HIGH" if cos > 0.7 else ("MEDIUM" if cos > 0.5 else "LOW")
        context += f"--- DOC {i+1} | {src} | Similarity: {cos:.2%} ({quality}) ---\n{doc.page_content}\n\n"
        sources_info.append({"source": src, "similarity": f"{cos:.2%}", "quality": quality,
                              "content": doc.page_content})

    answer = (combined_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "query_used": question,
        "sources_details": sources_info,
        "context_preview": "\n".join([f"[{s['source']}]: {s['content'][:80]}" for s in sources_info]),
    }


def run_judge(question: str, context: str, answer: str, llm) -> dict:
    """LLM-as-Judge : note la réponse RAG sur 4 axes (0-10)."""
    try:
        raw = (judge_prompt | llm | StrOutputParser()).invoke(
            {"question": question, "context": context, "answer": answer}
        )
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        scores["overall"] = round(
            (scores["faithfulness"] + scores["relevance"] + scores["completeness"] + scores["citation"]) / 4, 1
        )
        return scores
    except Exception as e:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "citation": 0,
                "overall": 0, "feedback": f"Eval error: {e}"}
