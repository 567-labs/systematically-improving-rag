import instructor
import openai
from pydantic import BaseModel, Field
from typing import List


client = instructor.from_openai(openai.OpenAI())


class RelevancyLabel(BaseModel):
    chunk_id: int
    reasoning: str
    is_relevant: bool
    confidence_score: float = Field(ge=0, le=1)


def predict_relevance(query: str, chunks: list[dict]) -> List[RelevancyLabel]:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=List[RelevancyLabel],
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant specialized in assessing the relevance of text chunks to specific queries. Your task is to meticulously analyze the content, considering various aspects such as key concepts, direct matches, contextual information, and supporting details. Provide a comprehensive and nuanced reasoning for your relevance determination, taking into account the depth, breadth, and novelty of the information in relation to the query. Your assessment should be thorough, unbiased, and well-justified.",
            },
            {
                "role": "user",
                "content": f"""
            Determine if the following chunk is relevant to the query: '{query}'

            <chunk>
            {chunks}
            </chunk>

            Instructions for reasoning about relevancy:
            1. Identify key concepts and topics in both the query and the chunk.
            2. Look for direct matches or closely related ideas between the query and chunk.
            3. Consider if the chunk provides context, background, or supporting information for the query.
            4. Evaluate if the chunk contains specific details, examples, or explanations that address the query.
            5. Assess whether the information in the chunk is necessary or helpful in answering the query.
            6. Think about the depth and breadth of information provided in relation to the query.
            7. Consider any temporal or causal relationships between the query and the chunk content.
            8. Determine if the chunk introduces new, relevant information or merely repeats what's already known.
            9. Be very careful with logic and reasoning.

            Based on these criteria, decide if the chunk is relevant or not, and provide a detailed explanation for your decision.
            """,
            },
        ],
    )


if __name__ == "__main__":
    chunks = [
        {
            "chunk_id": 1,
            "chunk": "Climate change is primarily caused by human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy, deforestation, and industrial processes are major contributors. These activities increase the concentration of carbon dioxide, methane, and other heat-trapping gases in the atmosphere, leading to a rise in global temperatures. This human-induced warming is occurring at a rate much faster than natural climate variability, resulting in various environmental impacts such as sea-level rise, extreme weather events, and changes in ecosystems.",
        },
        {
            "chunk_id": 2,
            "chunk": "The Earth's climate has always been in flux, with periods of warming and cooling throughout its history. Milankovitch cycles, which describe variations in the Earth's orbit and axis, play a role in long-term climate patterns. Additionally, solar activity fluctuations and volcanic eruptions can influence global temperatures. However, recent anthropogenic factors have accelerated these changes, leading to debates about the relative importance of natural versus human-induced climate forcings. Some researchers argue that cosmic rays and cloud formation mechanisms might be underestimated in current climate models, while others focus on the complex feedback loops within the Earth system, such as albedo effects and ocean circulation patterns.",
        },
        {
            "chunk_id": 3,
            "chunk": "The Earth's climate system is complex and influenced by various factors. While human activities, such as burning fossil fuels and deforestation, contribute to greenhouse gas emissions, the extent of their impact on global climate patterns remains a subject of ongoing research. Natural phenomena, including solar cycles, volcanic eruptions, and long-term oceanic oscillations, also play roles in climate variability. The interplay between these anthropogenic and natural factors creates a nuanced picture of climate change, with scientists continually refining models to better understand the relative contributions of each. This complexity sometimes leads to debates about the primary drivers of observed climate trends and the most effective strategies for mitigation.",
        },
    ]

    predictions = predict_relevance(
        "What are the primary causes of climate change?", chunks
    )

    for prediction in predictions:
        print(prediction.model_dump_json(indent=2))
