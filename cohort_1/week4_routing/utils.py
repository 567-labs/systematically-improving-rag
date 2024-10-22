import asyncio
from typing import List, Tuple, Any
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI


async_client = instructor.from_openai(AsyncOpenAI())


class FunctionList(BaseModel):
    """A model representing a list of function names."""

    func_names: List[str]


class QuestionWithTools(BaseModel):
    """A model representing a question and its required tools."""

    question: str
    required_tools: FunctionList


class ToolCallEvaluation(BaseModel):
    """A model representing the evaluation of a tool call."""

    question: str
    expected: FunctionList
    predicted: FunctionList


def describe_tools(tools: List[Any]) -> str:
    """
    Generate a string description of the given tools.

    Args:
        tools (List[Any]): A list of tool objects.

    Returns:
        str: A string containing the name and docstring of each tool.
    """

    def get_name(tool):
        return tool.__name__ if hasattr(tool, "__name__") else tool.__class__.__name__

    def get_doc(tool):
        return tool.__doc__ if hasattr(tool, "__doc__") else ""

    return "\n".join([f"{get_name(tool)}: {get_doc(tool)}" for tool in tools])


def calculate_precision_recall(
    desired_function_calls: List[FunctionList],
    actual_function_calls: List[FunctionList],
) -> Tuple[float, float]:
    """
    Calculate precision and recall for function calls.

    Args:
        desired_function_calls (List[FunctionList]): List of desired function calls.
        actual_function_calls (List[FunctionList]): List of actual function calls.

    Returns:
        Tuple[float, float]: A tuple containing (precision, recall).
    """
    true_positives = sum(
        len(set(desired.func_names) & set(actual.func_names))
        for desired, actual in zip(desired_function_calls, actual_function_calls)
    )
    false_positives = sum(
        len(set(actual.func_names) - set(desired.func_names))
        for desired, actual in zip(desired_function_calls, actual_function_calls)
    )
    false_negatives = sum(
        len(set(desired.func_names) - set(actual.func_names))
        for desired, actual in zip(desired_function_calls, actual_function_calls)
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    return precision, recall


async def get_one_tool_call_eval(
    q: QuestionWithTools, tool_list: str
) -> ToolCallEvaluation:
    """
    Get a single tool call evaluation.

    Args:
        q (QuestionWithTools): The question with required tools.
        tool_list (str): A string describing available tools.

    Returns:
        ToolCallEvaluation: The evaluation result.
    """
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""Identify the tools that will help you answer the user's question.
                    Respond with the names of 0, 1 or 2 tools to use. The available tools are
                    {tool_list}.

                    Don't make unnecessary function calls.
                    """,
                },
                {"role": "user", "content": q.question},
            ],
            temperature=0.0,
            response_model=FunctionList,
        )
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None

    return ToolCallEvaluation(
        question=q.question,
        expected=q.required_tools,
        predicted=response,
    )


async def get_all_tool_call_evals(
    synthetic_questions: List[QuestionWithTools],
    tool_list: str,
    max_concurrency: int = 40,
) -> Tuple[List[FunctionList], List[FunctionList]]:
    """
    Get all tool call evaluations for a list of synthetic questions.

    Args:
        synthetic_questions (List[QuestionWithTools]): List of synthetic questions.
        tool_list (str): A string describing available tools.
        max_concurrency (int, optional): Maximum number of concurrent API calls. Defaults to 40.

    Returns:
        Tuple[List[FunctionList], List[FunctionList]]: A tuple containing lists of desired and actual function calls.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_get_tool_call_evals(q: QuestionWithTools):
        async with semaphore:
            return await get_one_tool_call_eval(q, tool_list)

    tasks = [bounded_get_tool_call_evals(q) for q in synthetic_questions]
    eval_results = await asyncio.gather(*tasks)
    eval_results = [result for result in eval_results if result is not None]

    desired_function_calls = [q.expected for q in eval_results]
    actual_function_calls = [e.predicted for e in eval_results]
    return desired_function_calls, actual_function_calls
