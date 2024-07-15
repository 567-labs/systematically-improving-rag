import json
import asyncio
from typing import List
import logging
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch the AsyncOpenAI client
client = instructor.from_openai(AsyncOpenAI())


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class ChunkEval(QuestionAnswer):
    chunk_id: str


class TextChunk(BaseModel):
    id: str
    content: str


class ChunkProcessingError(Exception):
    pass


async def generate_evals(
    chunk: TextChunk, n_questions: int, example_questions: List[str]
) -> List[ChunkEval]:
    prompt = f"""
        Generate `{n_questions}` question-answer pairs based on the following content:

        <content>
        {chunk.content}
        </content>

        Example questions:
        {chr(10).join(f'- {q}' for q in example_questions)}

        Generate diverse questions that probe different aspects of the content. 
        Provide a concise answer for each question.
        Do not use the exact example questions, but use them as inspiration for the types of questions to generate.
        Do not include answers that are not in the content.
        """

    try:
        pairs = client.chat.completions.create_iterable(
            model="gpt-4o",
            response_model=QuestionAnswer,
            messages=[{"role": "user", "content": prompt}],
        )
        return [
            ChunkEval(question=pair.question, answer=pair.answer, chunk_id=chunk.id)
            async for pair in pairs
        ]
    except Exception as e:
        logger.error(f"Error generating evals: {str(e)}")
        return []


async def process_chunk(
    chunk: TextChunk,
    n_questions: int,
    example_questions: List[str],
    semaphore: asyncio.Semaphore,
) -> List[ChunkEval]:
    async with semaphore:
        try:
            return await generate_evals(chunk, n_questions, example_questions)
        except Exception as e:
            logger.error(f"Unexpected error processing chunk {chunk.id}: {str(e)}")
            raise ChunkProcessingError(f"Failed to process chunk {chunk.id}") from e


async def create_synthetic_dataset(
    chunks: List[TextChunk],
    n_questions: int,
    example_questions: List[str],
    max_concurrency: int = 10,
) -> List[ChunkEval]:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        process_chunk(chunk, n_questions, example_questions, semaphore)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    dataset = []
    for result in results:
        if isinstance(result, ChunkProcessingError):
            logger.error(str(result))
        elif isinstance(result, list):
            dataset.extend(result)
        else:
            logger.error(f"Unexpected result type: {type(result)}")

    return dataset


def save_dataset(dataset: List[ChunkEval], filename: str):
    with open(filename, "w") as f:
        json.dump([chunk_eval.dict() for chunk_eval in dataset], f, indent=2)


async def main():
    # Sample text chunks (replace with your actual data)
    sample_chunks = [
        TextChunk(
            id="chunk1",
            content="Machine learning is a method of data analysis that automates analytical model building.",
        ),
        TextChunk(
            id="chunk2",
            content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        ),
        TextChunk(
            id="chunk3",
            content="Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities.",
        ),
    ]

    # Example questions and number of questions to generate
    n_questions = 3
    example_questions = [
        "What is the main topic of this text?",
        "Can you summarize the key points in this content?",
        "How does this information relate to current trends in the field?",
    ]

    try:
        # Generate the dataset
        synthetic_dataset = await create_synthetic_dataset(
            sample_chunks, n_questions, example_questions
        )

        # Save the dataset
        save_dataset(synthetic_dataset, "synthetic_eval_dataset.json")

        logger.info(f"Generated {len(synthetic_dataset)} ChunkEvals.")
        logger.info("Dataset saved as 'synthetic_eval_dataset.json'")
    except Exception as e:
        logger.error(f"An error occurred during dataset creation: {str(e)}")


# Example usage
if __name__ == "__main__":
    asyncio.run(main())
