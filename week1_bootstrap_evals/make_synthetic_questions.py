import json
import asyncio
from typing import List
import logging
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
import lancedb


db = lancedb.connect("./lancedb")
reviews_table = db.open_table("reviews")
sample_reviews = reviews_table.to_pandas()
sample_reviews.review

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
        Generate `{n_questions}` question-answer pairs about the Sawzall PX-1000. The answers should be derived from information in this product review:

        <content>
        {chunk.content}
        </content>

        Example questions:
        {chr(10).join(f'- {q}' for q in example_questions)}

        Provide a concise and specific answer for each question.
        Do not use the exact example questions. Use them only as inspiration for the types of more specific questions to generate.
        Do not include answers that are not in the content.
        Questions should ask about product characteristics (e.g. durability) and answers should refer to product characteristics without referring to the reviewer specifically.

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
            content="""I've enjoyed using this saw. It is lightweight and the battery lasts longer than other brands.
            I've been using it for 3 years now and it has been very durable. It was twice as expensive as the PX-500. But
            it is comfortable to hold because of the light weight.""",
        ),
        TextChunk(
            id="chunk2",
            content="I thought it would cut through tile, and it doesn't. But it goes through plastics and wood like butter.",
        ),
        TextChunk(
            id="chunk3",
            content="I've used this saw almost every day for a year. It's incredibly reliable. But I recommend buying the spare battery for $20. I only get 2 hours per charge.",
        ),
    ]

    n_questions = 2  # number of questions to get in each LLM call
    example_questions = [
        "What does the reviewer like about the product?",
        "What does the reviewer think could be improved?",
    ]
    try:
        # Generate the dataset
        synthetic_dataset = await create_synthetic_dataset(
            sample_chunks, n_questions, example_questions
        )

        # Save the dataset
        save_dataset(synthetic_dataset, "synthetic_eval_questions.json")

        logger.info(f"Generated {len(synthetic_dataset)} ChunkEvals.")
        logger.info("Dataset saved as 'synthetic_eval_questions.json'")
    except Exception as e:
        logger.error(f"An error occurred during dataset creation: {str(e)}")


# Example usage
if __name__ == "__main__":
    asyncio.run(main())
