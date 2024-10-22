import json
import lancedb
import numpy as np
from pydantic import BaseModel
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Tuple

# Constants
FIRST_STAGE_LIMIT = 50
BASE_MODEL_PATH = "cross-encoder/stsb-distilroberta-base"
FINE_TUNED_MODEL_PATH = "./fine_tuned_reranker"


class EvalQuestion(BaseModel):
    question: str
    answer: str
    chunk_id: str
    question_with_context: str


# Load data and setup
with open("../week1_bootstrap_evals/synthetic_eval_dataset.json", "r") as f:
    synthetic_questions = json.load(f)
eval_questions = [EvalQuestion(**q) for q in synthetic_questions]

db = lancedb.connect("../week1_bootstrap_evals/lancedb")
reviews_table = db.open_table("reviews")


def score_question(eval_question: EvalQuestion, model: CrossEncoder) -> float:
    """
    Score a question using the given model.

    Args:
        eval_question (EvalQuestion): The question to evaluate.
        model (CrossEncoder): The model to use for ranking.

    Returns:
        float: The rank of the desired result, or inf if not found.
    """
    query = f"Answer the following question: {eval_question.question_with_context}\n."
    target_id = int(eval_question.chunk_id)
    first_stage = reviews_table.search(query).limit(FIRST_STAGE_LIMIT).to_pandas()
    first_stage_ids = first_stage.id.astype(int).values
    review_text = first_stage.review.values
    reranked_results = model.rank(query, review_text)
    is_right_result = lambda x: first_stage_ids[x["corpus_id"]] == target_id
    try:
        rank_of_desired_result = next(
            i + 1 for i, d in enumerate(reranked_results) if is_right_result(d)
        )
    except StopIteration:
        return np.inf
    return rank_of_desired_result


def mean_reciprocal_rank(ranks: List[float]) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) from a list of ranks.

    Args:
        ranks (List[float]): List of ranks for each query.

    Returns:
        float: The Mean Reciprocal Rank.
    """
    reciprocal_ranks = [1 / rank if rank != np.inf else 0 for rank in ranks]
    return np.mean(reciprocal_ranks)


def evaluate_model(
    model: CrossEncoder, model_name: str
) -> Tuple[List[float], float, float]:
    """
    Evaluate a model on the evaluation questions.

    Args:
        model (CrossEncoder): The model to evaluate.
        model_name (str): The name of the model for printing results.

    Returns:
        Tuple[List[float], float, float]: Ranks, recall at 5, and recall at 10.
    """
    ranks = [score_question(eval_question, model) for eval_question in eval_questions]
    recall_at_5 = np.mean([rank <= 5 for rank in ranks])
    recall_at_10 = np.mean([rank <= 10 for rank in ranks])
    mrr = mean_reciprocal_rank(ranks)

    print(f"{model_name} results:")
    print(f"Recall at 5: {recall_at_5}")
    print(f"Recall at 10: {recall_at_10}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")

    return ranks, recall_at_5, recall_at_10, mrr


# Evaluate base model
base_model = CrossEncoder(BASE_MODEL_PATH)
base_ranks, base_recall_at_5, base_recall_at_10, base_mrr = evaluate_model(
    base_model, "Base model"
)

# Evaluate fine-tuned model
fine_tuned_model = CrossEncoder(FINE_TUNED_MODEL_PATH)
fine_tuned_ranks, fine_tuned_recall_at_5, fine_tuned_recall_at_10, fine_tuned_mrr = (
    evaluate_model(fine_tuned_model, "Fine-tuned model")
)
