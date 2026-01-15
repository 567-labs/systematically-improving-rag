"""
Evaluation metrics and functionality for RAG system.

This module implements the evaluation-first approach from Workshop Chapter 1:
- Calculate recall@k to measure retrieval quality
- Store detailed results for error analysis (Chapter 1: understanding failure modes)
- Enable systematic comparison of different strategies

Key Insight: Measure before optimizing. The 50% performance gap between v1 and v2 
queries would be invisible without systematic evaluation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from datetime import datetime

from core.db import (
    get_conversations_by_hashes,
    save_evaluations_to_sqlite,
    save_detailed_evaluation_results,
    get_engine,
    Question,
)
from core.reranking import get_reranker
from core.search import VectorSearchEngine
from sqlmodel import Session, select

console = Console()


@dataclass
class RecallMetrics:
    """Recall metrics at different k values"""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_30: float
    total_queries: int
    successful_queries: int


@dataclass
class EvaluationResult:
    """Single query evaluation result"""

    question_id: str
    query: str
    target_conversation_hash: str
    found: bool
    rank: Optional[int]
    score: Optional[float]
    top_k_results: List[str]  # List of conversation hashes in top k


def calculate_recall_at_k(
    evaluation_results: List[EvaluationResult], k_values: List[int] = [1, 5, 10, 30]
) -> Dict[str, RecallMetrics]:
    """
    Calculate recall@k metrics

    Args:
        evaluation_results: List of evaluation results
        k_values: List of k values to calculate recall for

    Returns:
        Dictionary of metrics
    """
    total = len(evaluation_results)
    if total == 0:
        return {
            "metrics": RecallMetrics(
                recall_at_1=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                recall_at_30=0.0,
                total_queries=0,
                successful_queries=0,
            )
        }

    recalls = {}
    for k in k_values:
        found_at_k = sum(
            1 for r in evaluation_results if r.rank is not None and r.rank <= k
        )
        recalls[f"recall_at_{k}"] = found_at_k / total

    successful = sum(1 for r in evaluation_results if r.found)

    return {
        "metrics": RecallMetrics(
            recall_at_1=recalls.get("recall_at_1", 0.0),
            recall_at_5=recalls.get("recall_at_5", 0.0),
            recall_at_10=recalls.get("recall_at_10", 0.0),
            recall_at_30=recalls.get("recall_at_30", 0.0),
            total_queries=total,
            successful_queries=successful,
        )
    }


async def evaluate_questions(
    question_version: str,
    embeddings_path: Path,
    db_path: Path,
    embedding_model: str = "text-embedding-3-large",
    top_k: int = 30,
    limit: Optional[int] = None,
    experiment_id: Optional[str] = None,
    save_results: bool = True,
    reranker_name: str = "none",
    reranker_n: int = 60,
    target_type: str = "conversations",
) -> Tuple[List[EvaluationResult], RecallMetrics]:
    """
    Evaluate generated questions against embedded conversations

    Args:
        question_version: Version of questions to evaluate (v1, v2, etc)
        embeddings_path: Path to conversation embeddings
        db_path: Path to SQLite database
        embedding_model: Model to use for query embedding
        top_k: Number of results to retrieve
        limit: Limit number of questions to evaluate
        experiment_id: Experiment ID for tracking
        save_results: Whether to save results to database
        reranker_name: Name of reranker to use ("none", "sentence-transformers/<model>", "cohere/<model>")
        reranker_n: Number of documents to retrieve for reranking
        target_type: Type of target documents ("conversations" or "summary")

    Returns:
        Tuple of (evaluation results, recall metrics)
    """
    console.print(f"[bold green]Evaluating {question_version} questions[/bold green]")

    # Load questions from database
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Question).where(Question.version == question_version)
        if limit:
            statement = statement.limit(limit)
        questions = session.exec(statement).all()

    if not questions:
        console.print(f"[red]No questions found for version {question_version}[/red]")
        return [], RecallMetrics(0, 0, 0, 0, 0, 0)

    console.print(f"[cyan]Loaded {len(questions)} questions[/cyan]")

    # Initialize search engine and reranker
    search_engine = VectorSearchEngine(embeddings_path, embedding_model)
    reranker = get_reranker(reranker_name)

    if reranker_name != "none":
        console.print(f"[yellow]Using reranker: {reranker.name}[/yellow]")
        console.print(
            f"[yellow]Retrieving {reranker_n} documents for reranking[/yellow]"
        )
        # For reranking, we retrieve reranker_n documents initially
        initial_top_k = reranker_n
    else:
        initial_top_k = top_k

    # Evaluate each question
    evaluation_results = []

    with Progress() as progress:
        task = progress.add_task(
            f"Evaluating {question_version} questions", total=len(questions)
        )

        # Batch process questions
        batch_size = 100
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            queries = [q.question for q in batch]

            # Search for all queries in batch
            search_results = await search_engine.batch_search(
                queries, top_k=initial_top_k, show_progress=False
            )

            # Apply reranking if requested
            if reranker_name != "none":
                search_results = await _apply_reranking(
                    queries, search_results, reranker, db_path, target_type, top_k
                )

            # Process results
            for question, results in zip(batch, search_results):
                target_hash = question.conversation_hash

                # Check if target is in results
                found = False
                rank = None
                score = None
                top_k_hashes = []

                for r in results.results:
                    # Handle ID format variations between embeddings and summaries
                    # Summary embeddings in ChromaDB may have IDs like "hash_v1" while we search for "hash"
                    # The clean conversation_hash is stored in metadata for summaries
                    result_hash = r.id

                    # First check if metadata contains conversation_hash (for summary embeddings)
                    if "conversation_hash" in r.metadata:
                        result_hash = r.metadata["conversation_hash"]
                    # Otherwise, check if ID has a technique suffix and remove it
                    elif result_hash.endswith(("_v1", "_v2", "_v3", "_v4", "_v5")):
                        result_hash = result_hash.rsplit("_", 1)[0]

                    top_k_hashes.append(result_hash)
                    if result_hash == target_hash:
                        found = True
                        rank = r.rank
                        score = r.score

                eval_result = EvaluationResult(
                    question_id=question.id,
                    query=question.question,
                    target_conversation_hash=target_hash,
                    found=found,
                    rank=rank,
                    score=score,
                    top_k_results=top_k_hashes[:top_k],
                )
                evaluation_results.append(eval_result)

            progress.update(task, advance=len(batch))

    # Calculate metrics
    metrics_dict = calculate_recall_at_k(evaluation_results)
    metrics = metrics_dict["metrics"]

    # Save results to database if requested
    if save_results:
        evaluations = []
        for result in evaluation_results:
            for k in [1, 5, 10, 30]:
                evaluations.append(
                    {
                        "id": f"{result.question_id}_recall_at_{k}_{experiment_id or 'default'}",
                        "question_id": result.question_id,
                        "metric_name": f"recall_at_{k}",
                        "metric_value": 1.0
                        if result.rank and result.rank <= k
                        else 0.0,
                        "experiment_id": experiment_id,
                    }
                )

        saved = save_evaluations_to_sqlite(evaluations, db_path)
        console.print(f"[green]Saved {saved} evaluation metrics[/green]")

    # Print results
    print_evaluation_results(metrics, question_version)

    return evaluation_results, metrics


def print_evaluation_results(metrics: RecallMetrics, version: str):
    """Print evaluation results in a nice table"""
    table = Table(title=f"Evaluation Results for {version}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Queries", str(metrics.total_queries))
    table.add_row("Successful Queries", str(metrics.successful_queries))
    table.add_row("Recall@1", f"{metrics.recall_at_1:.2%}")
    table.add_row("Recall@5", f"{metrics.recall_at_5:.2%}")
    table.add_row("Recall@10", f"{metrics.recall_at_10:.2%}")
    table.add_row("Recall@30", f"{metrics.recall_at_30:.2%}")

    console.print(table)


def analyze_failures(
    evaluation_results: List[EvaluationResult], db_path: Path, top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Analyze failed queries to understand patterns

    Args:
        evaluation_results: List of evaluation results
        db_path: Path to database
        top_n: Number of failure examples to show

    Returns:
        List of failure analysis dicts
    """
    failures = [r for r in evaluation_results if not r.found or r.rank > 10]

    if not failures:
        console.print("[green]No failures to analyze![/green]")
        return []

    console.print(f"\n[bold red]Analyzing {len(failures)} failed queries[/bold red]")

    # Get conversation details for failures
    failure_hashes = [f.target_conversation_hash for f in failures[:top_n]]
    conversations = get_conversations_by_hashes(failure_hashes, db_path)
    conv_map = {c["conversation_hash"]: c for c in conversations}

    failure_analysis = []
    for failure in failures[:top_n]:
        conv = conv_map.get(failure.target_conversation_hash, {})
        analysis = {
            "query": failure.query,
            "target_hash": failure.target_conversation_hash,
            "rank": failure.rank,
            "score": failure.score,
            "conversation_preview": conv.get("text", "")[:200] + "...",
            "top_3_results": failure.top_k_results[:3],
        }
        failure_analysis.append(analysis)

        console.print(f"\n[yellow]Query:[/yellow] {failure.query}")
        console.print(f"[yellow]Target rank:[/yellow] {failure.rank or 'Not found'}")
        if failure.score:
            console.print(f"[yellow]Score:[/yellow] {failure.score:.4f}")

    return failure_analysis


async def _apply_reranking(
    queries: List[str],
    search_results: List,
    reranker,
    db_path: Path,
    target_type: str,
    final_top_k: int,
):
    """Apply reranking to search results"""
    from core.db import (
        get_conversations_by_hashes,
        get_summaries_by_hashes_and_technique,
    )

    reranked_results = []

    for query, results in zip(queries, search_results):
        if not results.results:
            reranked_results.append(results)
            continue

        # Extract conversation hashes and get text content
        doc_info = []
        conversation_hashes = []

        for r in results.results:
            # Handle ID format variations
            result_hash = r.id
            if "conversation_hash" in r.metadata:
                result_hash = r.metadata["conversation_hash"]
            elif result_hash.endswith(("_v1", "_v2", "_v3", "_v4", "_v5")):
                result_hash = result_hash.rsplit("_", 1)[0]

            conversation_hashes.append(result_hash)
            doc_info.append((r.id, result_hash, r.score, r.rank))

        # Get document text based on target type
        if target_type == "conversations":
            # Get conversation text
            conversations = get_conversations_by_hashes(conversation_hashes, db_path)
            hash_to_text = {c["conversation_hash"]: c["text"] for c in conversations}
        else:
            # Get summary text - need to determine technique from metadata or ID
            technique = None
            if "technique" in results.results[0].metadata:
                technique = results.results[0].metadata["technique"]
            elif results.results[0].id.endswith(("_v1", "_v2", "_v3", "_v4", "_v5")):
                technique = results.results[0].id.split("_")[-1]

            if technique:
                summaries = get_summaries_by_hashes_and_technique(
                    conversation_hashes, technique, db_path
                )
                hash_to_text = {s["conversation_hash"]: s["summary"] for s in summaries}
            else:
                # Fallback to original ranking if we can't determine technique
                reranked_results.append(results)
                continue

        # Prepare documents for reranking
        documents = []
        for doc_id, conv_hash, score, rank in doc_info:
            text = hash_to_text.get(conv_hash, "")
            if text:
                documents.append((doc_id, text, score))

        if not documents:
            reranked_results.append(results)
            continue

        # Apply reranking
        rerank_results = reranker.rerank(query, documents)

        # Convert back to search result format
        from dataclasses import dataclass

        @dataclass
        class RerankedResult:
            id: str
            score: float
            rank: int
            metadata: dict

        @dataclass
        class RerankedSearchResult:
            results: List[RerankedResult]

        new_results = []
        for rerank_result in rerank_results[:final_top_k]:
            # Find original metadata
            original_metadata = {}
            for r in results.results:
                if r.id == rerank_result.document_id:
                    original_metadata = r.metadata
                    break

            new_results.append(
                RerankedResult(
                    id=rerank_result.document_id,
                    score=rerank_result.rerank_score,
                    rank=rerank_result.reranked_rank,
                    metadata=original_metadata,
                )
            )

        reranked_results.append(RerankedSearchResult(results=new_results))

    return reranked_results


def save_evaluation_report(
    evaluation_results: List[EvaluationResult],
    metrics: RecallMetrics,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
    experiment_id: Optional[str] = None,
):
    """Save detailed evaluation report to JSON and optionally to SQLite"""
    detailed_results = [
        {
            "question_id": r.question_id,
            "query": r.query,
            "target": r.target_conversation_hash,
            "found": r.found,
            "rank": r.rank,
            "score": r.score,
        }
        for r in evaluation_results
    ]

    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "recall_at_1": metrics.recall_at_1,
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "recall_at_30": metrics.recall_at_30,
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
        },
        "metadata": metadata or {},
        "detailed_results": detailed_results,
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[green]Saved evaluation report to {output_path}[/green]")

    # Save detailed results to SQLite if requested
    if db_path and experiment_id:
        try:
            inserted = save_detailed_evaluation_results(
                detailed_results, db_path, experiment_id
            )
            console.print(f"[green]Saved {inserted} detailed results to SQLite[/green]")
        except Exception as e:
            console.print(f"[red]Error saving detailed results to SQLite: {e}[/red]")
