from pydantic import BaseModel, Field
from typing import List


class RelevanceScore(BaseModel):
    """LLM judgment of query-document relevance"""
    is_relevant: bool = Field(
        description="Whether the document is relevant to the query (True=1, False=0)"
    )
    reasoning: str = Field(
        description="Brief explanation of why the document is or isn't relevant"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the judgment (0.0 to 1.0)"
    )


class SearchResult(BaseModel):
    """A single search result document"""
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class RelevanceEvaluation(BaseModel):
    """Complete evaluation of a query-document pair"""
    query: str
    document: SearchResult
    llm_score: RelevanceScore
    human_score: bool
    agreement: bool = Field(
        description="Whether LLM and human agree on relevance"
    )


class EvaluationResults(BaseModel):
    """Aggregated results from the relevance evaluation"""
    query: str
    total_documents: int
    evaluations: List[RelevanceEvaluation]
    agreement_rate: float = Field(
        description="Percentage of documents where LLM and human agree"
    )
    llm_precision: float = Field(
        description="Of documents LLM marked relevant, how many humans agreed"
    )
    llm_recall: float = Field(
        description="Of documents humans marked relevant, how many LLM found"
    )
    confusion_matrix: dict = Field(
        description="2x2 confusion matrix: {tp, fp, tn, fn}"
    )
    confidence_analysis: dict = Field(
        description="Analysis of LLM confidence vs accuracy correlation"
    )