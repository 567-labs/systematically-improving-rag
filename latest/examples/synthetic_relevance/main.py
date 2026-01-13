import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict

import instructor
import typer
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import SearchResult, RelevanceScore, RelevanceEvaluation, EvaluationResults

console = Console()
app = typer.Typer()

# Initialize instructor with OpenAI
client = instructor.from_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def get_synthetic_questions() -> List[str]:
    """Return 4 diverse synthetic questions for evaluation"""
    return [
        "What are the most effective machine learning algorithms for classification tasks?",
        "What are some healthy breakfast recipes that are quick to prepare?",
        "How is climate change affecting global weather patterns and ecosystems?",
        "What are the essential software engineering best practices for large codebases?"
    ]


def mock_search(question: str) -> List[SearchResult]:
    """
    Mock search system that returns hardcoded results for demonstration.
    Returns 10 documents per question with mixed relevance levels.
    """

    # Document pool - mix of relevant and irrelevant content for each question
    all_documents = {
        "What are the most effective machine learning algorithms for classification tasks?": [
            SearchResult(id="ml_1", content="Support Vector Machines (SVMs) are powerful supervised learning algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that separates different classes with maximum margin.", metadata={"relevance": "high"}),
            SearchResult(id="ml_2", content="Random forests are ensemble learning methods that combine multiple decision trees to improve prediction accuracy and reduce overfitting. Each tree votes on the final prediction.", metadata={"relevance": "high"}),
            SearchResult(id="ml_3", content="Neural networks, inspired by biological neurons, consist of interconnected nodes that process information. Deep learning uses multi-layer neural networks to learn complex patterns from data.", metadata={"relevance": "high"}),
            SearchResult(id="ml_4", content="Gradient boosting algorithms like XGBoost and LightGBM build models sequentially, where each new model corrects errors from previous models. They're highly effective for structured data.", metadata={"relevance": "high"}),
            SearchResult(id="ml_5", content="K-means clustering is an unsupervised machine learning algorithm that partitions data into k clusters by minimizing within-cluster sum of squares.", metadata={"relevance": "medium"}),
            SearchResult(id="ml_6", content="The weather today is sunny with a high of 75 degrees. Perfect for outdoor activities like hiking or having a picnic in the park.", metadata={"relevance": "none"}),
            SearchResult(id="ml_7", content="Cooking pasta requires boiling water with salt, adding the pasta, and cooking for 8-12 minutes depending on the type. Always taste test for doneness.", metadata={"relevance": "none"}),
            SearchResult(id="ml_8", content="Stock market performance this quarter shows mixed results across sectors. Technology stocks gained 5% while energy stocks declined 3%.", metadata={"relevance": "none"}),
            SearchResult(id="ml_9", content="Python is a popular programming language often used in machine learning projects due to its extensive libraries like scikit-learn and TensorFlow.", metadata={"relevance": "medium"}),
            SearchResult(id="ml_10", content="The history of artificial intelligence dates back to the 1950s when researchers first began exploring computational approaches to intelligence.", metadata={"relevance": "low"}),
        ],
        "What are some healthy breakfast recipes that are quick to prepare?": [
            SearchResult(id="breakfast_1", content="Greek yogurt parfait with berries and granola provides protein, probiotics, and antioxidants. Layer Greek yogurt with fresh blueberries, strawberries, and a sprinkle of homemade granola.", metadata={"relevance": "high"}),
            SearchResult(id="breakfast_2", content="Overnight oats are a nutritious make-ahead breakfast. Combine rolled oats with milk, chia seeds, and your favorite fruits. Refrigerate overnight and enjoy in the morning.", metadata={"relevance": "high"}),
            SearchResult(id="breakfast_3", content="Avocado toast on whole grain bread topped with a poached egg creates a balanced meal with healthy fats, fiber, and protein to keep you energized all morning.", metadata={"relevance": "high"}),
            SearchResult(id="breakfast_4", content="Smoothie bowls made with frozen fruits, spinach, and protein powder offer a vitamin-packed start to your day. Top with nuts, seeds, and fresh fruit.", metadata={"relevance": "high"}),
            SearchResult(id="breakfast_5", content="Whole grain pancakes made with oat flour and topped with fresh berries provide complex carbohydrates and fiber for sustained energy.", metadata={"relevance": "medium"}),
            SearchResult(id="breakfast_6", content="Machine learning algorithms like neural networks require large datasets to train effectively and achieve good performance on new data.", metadata={"relevance": "none"}),
            SearchResult(id="breakfast_7", content="Climate change affects global weather patterns, leading to more frequent extreme weather events and rising sea levels worldwide.", metadata={"relevance": "none"}),
            SearchResult(id="breakfast_8", content="Software testing is crucial for ensuring code quality and preventing bugs from reaching production environments.", metadata={"relevance": "none"}),
            SearchResult(id="breakfast_9", content="Eggs are a complete protein source containing all essential amino acids. They can be prepared in many ways for breakfast.", metadata={"relevance": "medium"}),
            SearchResult(id="breakfast_10", content="Coffee is one of the most popular morning beverages worldwide, containing caffeine which can help improve alertness and focus.", metadata={"relevance": "low"}),
        ],
        "How is climate change affecting global weather patterns and ecosystems?": [
            SearchResult(id="climate_1", content="Rising global temperatures are causing polar ice caps to melt at unprecedented rates, contributing to sea level rise that threatens coastal communities worldwide.", metadata={"relevance": "high"}),
            SearchResult(id="climate_2", content="Extreme weather events like hurricanes, droughts, and heatwaves are becoming more frequent and intense due to climate change, affecting agriculture and human settlements.", metadata={"relevance": "high"}),
            SearchResult(id="climate_3", content="Ocean acidification caused by increased atmospheric CO2 is damaging coral reefs and marine ecosystems, threatening biodiversity and fishing industries.", metadata={"relevance": "high"}),
            SearchResult(id="climate_4", content="Climate change is shifting precipitation patterns globally, leading to more severe droughts in some regions and increased flooding in others.", metadata={"relevance": "high"}),
            SearchResult(id="climate_5", content="Arctic wildlife like polar bears and seals face habitat loss as sea ice continues to decline due to warming temperatures.", metadata={"relevance": "medium"}),
            SearchResult(id="climate_6", content="Artificial intelligence and machine learning are revolutionizing how we process and analyze large datasets in various industries.", metadata={"relevance": "none"}),
            SearchResult(id="climate_7", content="Mediterranean diet rich in olive oil, fish, and vegetables has been linked to numerous health benefits including reduced heart disease risk.", metadata={"relevance": "none"}),
            SearchResult(id="climate_8", content="Agile software development methodologies emphasize iterative development, collaboration, and responding to change over following a fixed plan.", metadata={"relevance": "none"}),
            SearchResult(id="climate_9", content="Solar and wind energy technologies are becoming more affordable as governments and companies seek to reduce carbon emissions.", metadata={"relevance": "medium"}),
            SearchResult(id="climate_10", content="Weather prediction models use complex mathematical equations to forecast short-term atmospheric conditions.", metadata={"relevance": "low"}),
        ],
        "What are the essential software engineering best practices for large codebases?": [
            SearchResult(id="software_1", content="Code reviews are essential for maintaining code quality, catching bugs early, and sharing knowledge among team members. They should be constructive and thorough.", metadata={"relevance": "high"}),
            SearchResult(id="software_2", content="Version control systems like Git enable developers to track changes, collaborate effectively, and maintain a complete history of code modifications.", metadata={"relevance": "high"}),
            SearchResult(id="software_3", content="Test-driven development (TDD) involves writing tests before implementing features, leading to better code design and higher test coverage.", metadata={"relevance": "high"}),
            SearchResult(id="software_4", content="Continuous integration and continuous deployment (CI/CD) pipelines automate testing and deployment processes, reducing manual errors and improving release velocity.", metadata={"relevance": "high"}),
            SearchResult(id="software_5", content="Clean code principles emphasize readable, maintainable code with meaningful variable names, small functions, and clear documentation.", metadata={"relevance": "medium"}),
            SearchResult(id="software_6", content="Quinoa salad with roasted vegetables makes a nutritious lunch option packed with complete proteins and essential vitamins.", metadata={"relevance": "none"}),
            SearchResult(id="software_7", content="Renewable energy sources like solar panels are becoming more efficient and cost-effective for residential use.", metadata={"relevance": "none"}),
            SearchResult(id="software_8", content="Deep learning models require careful hyperparameter tuning to achieve optimal performance on specific tasks.", metadata={"relevance": "none"}),
            SearchResult(id="software_9", content="Database design principles include normalization, proper indexing, and choosing appropriate data types for optimal performance.", metadata={"relevance": "medium"}),
            SearchResult(id="software_10", content="Programming languages like Python and JavaScript are popular choices for web development due to their extensive ecosystems.", metadata={"relevance": "low"}),
        ]
    }

    return all_documents.get(question, [])


async def get_llm_relevance_score(query: str, document: SearchResult) -> RelevanceScore:
    """
    Use LLM as a judge to score document relevance to query.
    Returns structured binary relevance score with reasoning.
    """
    prompt = f"""
    You are an expert information retrieval judge tasked with evaluating document relevance.

    Your job is to determine if a document is relevant to a user's search query. A document is RELEVANT if it contains information that directly helps answer the query or provides useful context for the query topic.

    ## Evaluation Criteria:

    **RELEVANT (1) if the document:**
    - Directly answers the query or provides information requested
    - Contains key concepts, terms, or topics mentioned in the query
    - Provides useful background or context for understanding the query topic
    - Offers practical information someone searching this query would find valuable

    **NOT RELEVANT (0) if the document:**
    - Is about a completely different topic with no connection to the query
    - Only mentions query terms in passing without substantive content
    - Contains information that wouldn't help someone with this information need
    - Is too general/vague to provide meaningful value for the specific query

    ## Instructions:
    1. Read the query carefully to understand the user's information need
    2. Analyze the document content for relevance to that need
    3. Make a binary decision: relevant (1) or not relevant (0)
    4. Provide clear reasoning for your decision
    5. Rate your confidence in this judgment (0.0 = very uncertain, 1.0 = completely certain)

    <query>
    {query}
    </query>

    <document>
    {document.content}
    </document>

    Evaluate this document's relevance to the query using the criteria above.
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_model=RelevanceScore,
        temperature=0.1
    )

    return response


def calculate_ndcg(predicted_relevance: List[bool], true_relevance_scores: List[str], k: int = 10) -> float:
    """Calculate NDCG@k score"""
    # Convert true relevance scores to gains (high/medium/low/none -> 3/2/1/0)
    relevance_to_gain = {"high": 3, "medium": 2, "low": 1, "none": 0}
    true_gains = [relevance_to_gain.get(score, 0) for score in true_relevance_scores]

    # Calculate DCG for predicted ranking
    dcg = 0.0
    for i, (pred, gain) in enumerate(zip(predicted_relevance[:k], true_gains[:k])):
        if pred:  # Only count if LLM predicted relevant
            dcg += gain / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate ideal DCG (best possible ranking)
    sorted_gains = sorted(true_gains, reverse=True)
    idcg = 0.0
    for i, gain in enumerate(sorted_gains[:k]):
        if gain > 0:
            idcg += gain / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def analyze_llm_performance(all_results: Dict) -> None:
    """Analyze LLM performance against ground truth from metadata"""
    console.print("\n📈 [bold green]LLM Performance Analysis[/bold green]")

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    all_ndcg_scores = []

    # Per-question analysis
    question_table = Table(title="📋 Per-Question Analysis")
    question_table.add_column("Question", style="blue", width=40)
    question_table.add_column("Search P@10", style="green")
    question_table.add_column("LLM P", style="magenta")
    question_table.add_column("LLM R", style="yellow")
    question_table.add_column("NDCG@10", style="red")

    for question, results in all_results.items():
        # Extract ground truth from metadata
        true_relevance = []
        llm_predictions = []
        true_relevance_scores = []

        for result in results:
            # Get documents to check metadata
            documents = mock_search(question)
            doc = next((d for d in documents if d.id == result["document_id"]), None)

            if doc:
                true_rel_level = doc.metadata.get("relevance", "none")
                true_relevant = true_rel_level in ["high", "medium", "low"]
                true_relevance.append(true_relevant)
                true_relevance_scores.append(true_rel_level)

                llm_relevant = result["llm_score"]["is_relevant"]
                llm_predictions.append(llm_relevant)

        # Calculate metrics for this question
        tp = sum(1 for t, p in zip(true_relevance, llm_predictions) if t and p)
        fp = sum(1 for t, p in zip(true_relevance, llm_predictions) if not t and p)
        tn = sum(1 for t, p in zip(true_relevance, llm_predictions) if not t and not p)
        fn = sum(1 for t, p in zip(true_relevance, llm_predictions) if t and not p)

        # Search system precision@10 (relevant docs in top 10 results)
        search_precision_at_10 = sum(true_relevance) / len(true_relevance) if true_relevance else 0

        # LLM judge precision (of docs LLM marked relevant, how many were actually relevant)
        llm_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # LLM judge recall (of truly relevant docs, how many did LLM find)
        llm_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate NDCG
        ndcg = calculate_ndcg(llm_predictions, true_relevance_scores)
        all_ndcg_scores.append(ndcg)

        question_table.add_row(
            question[:37] + "..." if len(question) > 40 else question,
            f"{search_precision_at_10:.1%}",
            f"{llm_precision:.1%}",
            f"{llm_recall:.1%}",
            f"{ndcg:.3f}"
        )

        # Add to totals
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    console.print(question_table)

    # Overall metrics
    overall_search_precision = (total_tp + total_fn) / (len(all_results) * 10)  # Total relevant / Total docs (40)
    overall_llm_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_llm_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_llm_precision * overall_llm_recall) / (overall_llm_precision + overall_llm_recall) if (overall_llm_precision + overall_llm_recall) > 0 else 0
    avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0

    summary_table = Table(title="📊 Overall Performance")
    summary_table.add_column("Metric", style="bold blue")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Search System P@10", f"{overall_search_precision:.1%}")
    summary_table.add_row("LLM Judge Precision", f"{overall_llm_precision:.1%}")
    summary_table.add_row("LLM Judge Recall", f"{overall_llm_recall:.1%}")
    summary_table.add_row("LLM Judge F1", f"{overall_f1:.1%}")
    summary_table.add_row("Average NDCG@10", f"{avg_ndcg:.3f}")
    summary_table.add_row("Total Relevant Docs", str(total_tp + total_fn))
    summary_table.add_row("LLM Found Relevant", str(total_tp + total_fp))

    console.print(summary_table)

    # Confusion matrix
    console.print(f"\n📈 [bold]LLM Judge Confusion Matrix:[/bold]")
    console.print(f"  True Positives:  {total_tp}")
    console.print(f"  False Positives: {total_fp}")
    console.print(f"  True Negatives:  {total_tn}")
    console.print(f"  False Negatives: {total_fn}")


def get_human_relevance_score(query: str, document: SearchResult) -> bool:
    """
    Prompt human annotator for relevance judgment using arrow keys.
    Returns binary relevance score.
    """
    console.print("\n" + "="*80)
    console.print(Panel(f"[bold blue]Question:[/bold blue] {query}", title="🔍 Question"))
    console.print(Panel(f"[dim]{document.content}[/dim]", title=f"📄 Document {document.id}"))

    try:
        import keyboard

        console.print("\n[bold yellow]Is this document relevant to the question?[/bold yellow]")
        console.print("Use [bold green]→ (Right Arrow)[/bold green] for Relevant or [bold red]← (Left Arrow)[/bold red] for Not Relevant")
        console.print("Or press [bold cyan]y/n[/bold cyan] keys")

        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == 'right' or event.name == 'y':
                    console.print("✅ [bold green]RELEVANT[/bold green]")
                    return True
                elif event.name == 'left' or event.name == 'n':
                    console.print("❌ [bold red]NOT RELEVANT[/bold red]")
                    return False
                elif event.name == 'esc' or event.name == 'q':
                    console.print("⏹️ Exiting...")
                    sys.exit(0)

    except ImportError:
        # Fallback to simple y/n input if keyboard library not available
        console.print("\n[bold yellow]Is this document relevant to the question? (y/n):[/bold yellow]")
        while True:
            response = input().lower().strip()
            if response in ['y', 'yes', '1', 'true']:
                return True
            elif response in ['n', 'no', '0', 'false']:
                return False
            else:
                console.print("Please enter 'y' for relevant or 'n' for not relevant:")


def calculate_metrics(evaluations: List[RelevanceEvaluation]) -> dict:
    """Calculate evaluation metrics from LLM vs human comparisons"""
    # Count agreements and disagreements
    agreements = sum(1 for eval in evaluations if eval.agreement)
    total = len(evaluations)
    agreement_rate = agreements / total if total > 0 else 0

    # Calculate confusion matrix
    tp = sum(1 for eval in evaluations if eval.llm_score.is_relevant and eval.human_score)
    fp = sum(1 for eval in evaluations if eval.llm_score.is_relevant and not eval.human_score)
    tn = sum(1 for eval in evaluations if not eval.llm_score.is_relevant and not eval.human_score)
    fn = sum(1 for eval in evaluations if not eval.llm_score.is_relevant and eval.human_score)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Confidence correlation analysis
    correct_predictions = [eval for eval in evaluations if eval.agreement]
    incorrect_predictions = [eval for eval in evaluations if not eval.agreement]

    avg_confidence_correct = sum(eval.llm_score.confidence for eval in correct_predictions) / len(correct_predictions) if correct_predictions else 0
    avg_confidence_incorrect = sum(eval.llm_score.confidence for eval in incorrect_predictions) / len(incorrect_predictions) if incorrect_predictions else 0

    # Confidence calibration: are high-confidence predictions more accurate?
    high_confidence_evals = [eval for eval in evaluations if eval.llm_score.confidence >= 0.8]
    low_confidence_evals = [eval for eval in evaluations if eval.llm_score.confidence < 0.8]

    high_confidence_accuracy = sum(1 for eval in high_confidence_evals if eval.agreement) / len(high_confidence_evals) if high_confidence_evals else 0
    low_confidence_accuracy = sum(1 for eval in low_confidence_evals if eval.agreement) / len(low_confidence_evals) if low_confidence_evals else 0

    return {
        "agreement_rate": agreement_rate,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "confidence_analysis": {
            "avg_confidence_when_correct": avg_confidence_correct,
            "avg_confidence_when_incorrect": avg_confidence_incorrect,
            "high_confidence_accuracy": high_confidence_accuracy,
            "low_confidence_accuracy": low_confidence_accuracy,
            "confidence_calibration_gap": high_confidence_accuracy - low_confidence_accuracy
        }
    }


def display_results(results: EvaluationResults):
    """Display evaluation results in a formatted table"""
    console.print("\n🎯 [bold green]Evaluation Results[/bold green]")

    # Summary table
    summary_table = Table(title="📊 Summary Metrics")
    summary_table.add_column("Metric", style="bold blue")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Agreement Rate", f"{results.agreement_rate:.1%}")
    summary_table.add_row("LLM Precision", f"{results.llm_precision:.1%}")
    summary_table.add_row("LLM Recall", f"{results.llm_recall:.1%}")

    console.print(summary_table)

    # Confidence analysis
    conf_analysis = results.confidence_analysis
    conf_table = Table(title="🎯 Confidence Analysis")
    conf_table.add_column("Metric", style="bold blue")
    conf_table.add_column("Value", style="cyan")

    conf_table.add_row("Avg Confidence (Correct)", f"{conf_analysis['avg_confidence_when_correct']:.3f}")
    conf_table.add_row("Avg Confidence (Incorrect)", f"{conf_analysis['avg_confidence_when_incorrect']:.3f}")
    conf_table.add_row("High Confidence Accuracy", f"{conf_analysis['high_confidence_accuracy']:.1%}")
    conf_table.add_row("Low Confidence Accuracy", f"{conf_analysis['low_confidence_accuracy']:.1%}")
    conf_table.add_row("Calibration Gap", f"{conf_analysis['confidence_calibration_gap']:+.1%}")

    console.print(conf_table)

    # Confusion matrix
    cm = results.confusion_matrix
    console.print(f"\n📈 [bold]Confusion Matrix:[/bold]")
    console.print(f"  True Positives:  {cm['tp']}")
    console.print(f"  False Positives: {cm['fp']}")
    console.print(f"  True Negatives:  {cm['tn']}")
    console.print(f"  False Negatives: {cm['fn']}")


@app.command()
def generate_llm_scores():
    """
    Phase 1: Generate LLM relevance scores for all query-document pairs.

    This runs async batch processing of all LLM judgments without blocking
    on human input. Results are saved to JSON files for later annotation.
    """
    async def run_llm_generation():
        console.print("🤖 [bold green]Generating LLM Relevance Scores[/bold green]\n")

        questions = get_synthetic_questions()
        all_llm_results = {}

        for question in questions:
            console.print(f"📝 Processing question: [bold blue]{question}[/bold blue]")
            documents = mock_search(question)

            # Process all documents for this question concurrently
            tasks = [get_llm_relevance_score(question, doc) for doc in documents]
            llm_scores = await asyncio.gather(*tasks)

            # Store results
            question_results = []
            for doc, score in zip(documents, llm_scores):
                question_results.append({
                    "question": question,
                    "document_id": doc.id,
                    "document_content": doc.content,
                    "llm_score": {
                        "is_relevant": score.is_relevant,
                        "reasoning": score.reasoning,
                        "confidence": score.confidence
                    }
                })

            all_llm_results[question] = question_results
            console.print(f"✅ Completed {len(documents)} documents for question: {question}\n")

        # Save to JSON file
        output_file = "llm_scores.json"
        with open(output_file, "w") as f:
            json.dump(all_llm_results, f, indent=2)

        console.print(f"💾 Saved LLM scores to: [bold green]{output_file}[/bold green]")
        console.print(f"📊 Total evaluations: [bold yellow]{sum(len(results) for results in all_llm_results.values())}[/bold yellow]")

        # Analyze LLM predictions vs ground truth (from metadata)
        analyze_llm_performance(all_llm_results)

        console.print("\n🎯 Next step: Run '[bold cyan]uv run python main.py label[/bold cyan]' to add human annotations")

    asyncio.run(run_llm_generation())


@app.command()
def label():
    """
    Phase 2: Add human labels to pre-generated LLM scores.

    Loads LLM scores from JSON and presents each query-document pair
    for human annotation. Saves completed evaluations as you go.
    """

    # Load LLM scores
    llm_file = Path("llm_scores.json")
    if not llm_file.exists():
        console.print("❌ No LLM scores found. Run '[bold cyan]generate-llm-scores[/bold cyan]' first.")
        return

    with open(llm_file, "r") as f:
        llm_data = json.load(f)

    # Load existing human annotations if any
    human_file = Path("human_annotations.json")
    completed_annotations = {}
    if human_file.exists():
        with open(human_file, "r") as f:
            completed_annotations = json.load(f)

    console.print("👤 [bold green]Human Annotation Interface[/bold green]\n")

    total_items = sum(len(results) for results in llm_data.values())
    completed_items = sum(len(results) for results in completed_annotations.values())

    console.print(f"📊 Progress: {completed_items}/{total_items} completed\n")

    for query, documents in llm_data.items():
        if query not in completed_annotations:
            completed_annotations[query] = []

        completed_ids = {item["document_id"] for item in completed_annotations[query]}

        for doc_data in documents:
            doc_id = doc_data["document_id"]

            # Skip if already annotated
            if doc_id in completed_ids:
                continue

            # Show LLM's judgment first
            llm_score = doc_data["llm_score"]
            console.print("="*80)
            console.print(Panel(f"[bold blue]Query:[/bold blue] {query}", title="🔍 Search Query"))
            console.print(Panel(f"[dim]{doc_data['document_content']}[/dim]", title=f"📄 Document {doc_id}"))

            # Show LLM judgment
            llm_relevant = "✅ Relevant" if llm_score["is_relevant"] else "❌ Not Relevant"
            console.print(f"\n🤖 [bold cyan]LLM Judge:[/bold cyan] {llm_relevant} (confidence: {llm_score['confidence']:.2f})")
            console.print(f"💭 [italic]LLM Reasoning:[/italic] {llm_score['reasoning']}")

            # Get human annotation
            human_score = Confirm.ask(
                "\n[bold yellow]Do YOU think this document is relevant to the query?[/bold yellow]",
                default=False
            )

            # Store annotation
            annotation = {
                "document_id": doc_id,
                "human_score": human_score,
                "llm_score": llm_score["is_relevant"],
                "agreement": human_score == llm_score["is_relevant"],
                "llm_confidence": llm_score["confidence"]
            }
            completed_annotations[query].append(annotation)

            # Save progress
            with open(human_file, "w") as f:
                json.dump(completed_annotations, f, indent=2)

            # Show agreement
            agreement = "✅ Agree" if annotation["agreement"] else "❌ Disagree"
            console.print(f"🎯 {agreement}\n")

    console.print("🎉 [bold green]All annotations completed![/bold green]")
    console.print("🎯 Next step: Run '[bold cyan]uv run python main.py analyze[/bold cyan]' to see results")


@app.command()
def analyze():
    """
    Phase 3: Analyze correlation between LLM and human judgments.

    Loads completed annotations and generates comprehensive analysis
    including confidence correlation and per-query breakdowns.
    """

    # Load annotations
    human_file = Path("human_annotations.json")
    if not human_file.exists():
        console.print("❌ No human annotations found. Complete labeling first.")
        return

    with open(human_file, "r") as f:
        annotations = json.load(f)

    console.print("📊 [bold green]Relevance Evaluation Analysis[/bold green]\n")

    # Aggregate all evaluations
    all_evaluations = []
    for query, query_annotations in annotations.items():
        for annotation in query_annotations:
            all_evaluations.append({
                "query": query,
                "agreement": annotation["agreement"],
                "llm_score": annotation["llm_score"],
                "human_score": annotation["human_score"],
                "confidence": annotation["llm_confidence"]
            })

    # Calculate overall metrics
    total = len(all_evaluations)
    agreements = sum(1 for eval in all_evaluations if eval["agreement"])

    # Confusion matrix
    tp = sum(1 for eval in all_evaluations if eval["llm_score"] and eval["human_score"])
    fp = sum(1 for eval in all_evaluations if eval["llm_score"] and not eval["human_score"])
    tn = sum(1 for eval in all_evaluations if not eval["llm_score"] and not eval["human_score"])
    fn = sum(1 for eval in all_evaluations if not eval["llm_score"] and eval["human_score"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Confidence analysis
    correct_evals = [eval for eval in all_evaluations if eval["agreement"]]
    incorrect_evals = [eval for eval in all_evaluations if not eval["agreement"]]

    avg_conf_correct = sum(eval["confidence"] for eval in correct_evals) / len(correct_evals) if correct_evals else 0
    avg_conf_incorrect = sum(eval["confidence"] for eval in incorrect_evals) / len(incorrect_evals) if incorrect_evals else 0

    high_conf_evals = [eval for eval in all_evaluations if eval["confidence"] >= 0.8]
    low_conf_evals = [eval for eval in all_evaluations if eval["confidence"] < 0.8]

    high_conf_acc = sum(1 for eval in high_conf_evals if eval["agreement"]) / len(high_conf_evals) if high_conf_evals else 0
    low_conf_acc = sum(1 for eval in low_conf_evals if eval["agreement"]) / len(low_conf_evals) if low_conf_evals else 0

    # Display results
    summary_table = Table(title="📊 Overall Results")
    summary_table.add_column("Metric", style="bold blue")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Evaluations", str(total))
    summary_table.add_row("Agreement Rate", f"{agreements/total:.1%}")
    summary_table.add_row("LLM Precision", f"{precision:.1%}")
    summary_table.add_row("LLM Recall", f"{recall:.1%}")

    console.print(summary_table)

    # Confidence analysis table
    conf_table = Table(title="🎯 Confidence Analysis")
    conf_table.add_column("Metric", style="bold blue")
    conf_table.add_column("Value", style="cyan")

    conf_table.add_row("Avg Confidence (Correct)", f"{avg_conf_correct:.3f}")
    conf_table.add_row("Avg Confidence (Incorrect)", f"{avg_conf_incorrect:.3f}")
    conf_table.add_row("High Confidence Accuracy", f"{high_conf_acc:.1%}")
    conf_table.add_row("Low Confidence Accuracy", f"{low_conf_acc:.1%}")
    conf_table.add_row("Calibration Gap", f"{high_conf_acc - low_conf_acc:+.1%}")

    console.print(conf_table)

    # Per-query breakdown
    query_table = Table(title="📋 Per-Query Results")
    query_table.add_column("Query", style="blue")
    query_table.add_column("Agreements", style="green")
    query_table.add_column("Total", style="dim")
    query_table.add_column("Rate", style="cyan")

    for query, query_annotations in annotations.items():
        query_agreements = sum(1 for ann in query_annotations if ann["agreement"])
        query_total = len(query_annotations)
        query_rate = query_agreements / query_total if query_total > 0 else 0

        query_table.add_row(
            query[:40] + "..." if len(query) > 40 else query,
            str(query_agreements),
            str(query_total),
            f"{query_rate:.1%}"
        )

    console.print(query_table)

    # Key insights
    console.print("\n💡 [bold yellow]Key Insights:[/bold yellow]")

    if avg_conf_correct > avg_conf_incorrect + 0.1:
        console.print("✅ LLM shows good confidence calibration - more confident when correct")
    else:
        console.print("⚠️  LLM confidence doesn't correlate well with accuracy")

    if high_conf_acc > low_conf_acc + 0.2:
        console.print("🎯 High-confidence predictions are significantly more accurate")
    else:
        console.print("🤔 Confidence level doesn't predict accuracy well")

    if agreements/total >= 0.8:
        console.print("🎉 High agreement! LLM judge aligns well with human judgment")
    elif agreements/total >= 0.6:
        console.print("👍 Moderate agreement. Consider prompt improvements")
    else:
        console.print("⚠️  Low agreement. LLM prompt needs significant work")


@app.command()
def evaluate(
    query: str = typer.Option(
        "machine learning algorithms",
        help="Single query to evaluate (legacy mode - use generate-llm-scores instead)"
    )
):
    """
    Legacy single-query evaluation mode.

    For better workflow, use the 3-phase approach:
    1. generate-llm-scores (async batch processing)
    2. label (human annotation)
    3. analyze (correlation analysis)
    """
    async def run_evaluation():
        console.print("🚀 [bold green]Single Query Evaluation (Legacy Mode)[/bold green]")
        console.print("💡 [yellow]Tip: Use 'generate-llm-scores' → 'label' → 'analyze' for better workflow[/yellow]\n")
        console.print(f"📝 Query: [bold blue]{query}[/bold blue]\n")

        # Get mock search results
        documents = mock_search(query)
        console.print(f"🔍 Retrieved {len(documents)} documents from mock search\n")

        evaluations = []

        for i, doc in enumerate(documents, 1):
            console.print(f"⏳ Processing document {i}/{len(documents)}...")

            # Get LLM score
            llm_score = await get_llm_relevance_score(query, doc)

            # Get human score
            human_score = get_human_relevance_score(query, doc)

            # Create evaluation record
            evaluation = RelevanceEvaluation(
                query=query,
                document=doc,
                llm_score=llm_score,
                human_score=human_score,
                agreement=(llm_score.is_relevant == human_score)
            )
            evaluations.append(evaluation)

            console.print(f"✅ LLM: {'Relevant' if llm_score.is_relevant else 'Not Relevant'} | "
                         f"Human: {'Relevant' if human_score else 'Not Relevant'} | "
                         f"{'✅ Agree' if evaluation.agreement else '❌ Disagree'}\n")

        # Calculate final metrics
        metrics = calculate_metrics(evaluations)

        # Create results object
        results = EvaluationResults(
            query=query,
            total_documents=len(documents),
            evaluations=evaluations,
            **metrics
        )

        # Display results
        display_results(results)

        # Final insights
        console.print("\n💡 [bold yellow]Key Insights:[/bold yellow]")
        if results.agreement_rate >= 0.8:
            console.print("🎉 High agreement! LLM judge aligns well with human judgment.")
        elif results.agreement_rate >= 0.6:
            console.print("👍 Moderate agreement. LLM judge shows promise but may need tuning.")
        else:
            console.print("⚠️  Low agreement. Consider improving LLM prompts or training data.")

        console.print(f"\n📊 Binary scoring (0/1) simplifies annotation and improves reliability compared to 5-star scales.")

    # Run the async evaluation
    asyncio.run(run_evaluation())


@app.command()
def demo():
    """Show example of what the evaluation looks like without running it"""
    console.print("🎯 [bold green]Synthetic Relevance Evaluation Demo[/bold green]\n")

    console.print("This tool demonstrates LLM-as-a-judge for relevance scoring:\n")

    console.print("1. 🔍 [bold blue]Mock Search:[/bold blue] Query returns 7 hardcoded documents")
    console.print("2. 🤖 [bold cyan]LLM Judge:[/bold cyan] GPT-4 scores each doc as 0 (not relevant) or 1 (relevant)")
    console.print("3. 👤 [bold magenta]Human Judge:[/bold magenta] You score each doc with y/n")
    console.print("4. 📊 [bold green]Analysis:[/bold green] Compare agreement rates and calculate metrics\n")

    console.print("💡 [bold yellow]Why Binary Scoring?[/bold yellow]")
    console.print("• Simpler decisions than 5-star scales")
    console.print("• Higher inter-annotator agreement")
    console.print("• Easier to analyze and debug")
    console.print("• Perfect starting point for relevance evaluation\n")

    console.print("🚀 Run: [bold green]uv run python main.py evaluate[/bold green]")


if __name__ == "__main__":
    app()