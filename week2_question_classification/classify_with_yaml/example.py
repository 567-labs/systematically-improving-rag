import openai
import instructor
from yaml_classifier import YamlClassifier
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List

client = instructor.from_openai(openai.OpenAI())

classifier = YamlClassifier.load("example.yaml")


class Prediction(BaseModel):
    correct_labels: List[str] = Field(
        description="The predicted label(s) as a list of strings"
    )

    @field_validator("correct_labels")
    def validate_labels(cls, v, info: ValidationInfo):
        labels = info.context["labels"]
        for label in v:
            if label not in labels:
                raise ValueError(f"Label {label} not in {labels}")
        return v


class PredictionWithReasoning(Prediction):
    reasoning: str = Field(
        description="A detailed explanation of the thought process leading to the prediction, including key factors considered, comparisons to label descriptions and examples, and how the query's content and intent align with the chosen label"
    )


# Example without reasoning

resp = classifier.predict(
    query="When was the last time i ask you about dinner?",
    response_model=Prediction,
    client=client,
    model="gpt-4o-mini",
)

print(resp.model_dump_json(indent=2))
# > {"correct_labels": ["time_filter_requirement"]}

# Example with batch prediction with asyncio
from asyncio import run, gather

client = instructor.from_openai(openai.AsyncOpenAI())

examples = [
    "When was the last time I asked you about dinner?",
    "Can you draft an email to my team about the project deadline?",
    "Do I have permission to access the financial reports?",
    "I need to draft a report on our current stock prices for the board meeting tomorrow.",
    "Can you give me access to the server room and write an email to IT about it?",
    "What were yesterday's headlines about our company's financial performance?",
]


async def run_predictions():
    tasks = [
        classifier.apredict(
            client=client,  # type: ignore
            model="gpt-4o-mini",
            query=query,
            response_model=Prediction,
        )
        for query in examples
    ]
    return await gather(*tasks)


resp = run(run_predictions())

for r in resp:
    print(r.model_dump_json(indent=2))
