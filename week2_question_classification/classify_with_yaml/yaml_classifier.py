from pydantic import BaseModel, Field
from typing import List
from pydantic import BaseModel, field_validator
from instructor import Instructor, AsyncInstructor
from jinja2 import Template
from textwrap import dedent
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)


class Example(BaseModel):
    positive: List[str]
    negative: List[str]


class Label(BaseModel):
    name: str
    description: str
    examples: Example

    @field_validator("name")
    def name_must_be_snake_case(cls, v):
        import re

        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("Label name must be in snake_case")
        return v


class YamlClassifier(BaseModel):
    task: str
    description: str
    labels: List[Label]
    n_examples: int | None = Field(
        default=100, description="Number of examples to use for each label"
    )

    @classmethod
    def load(cls, fn: str):
        import yaml

        with open(fn, "r") as file:
            data = yaml.safe_load(file)

        return cls(**data)

    def to_system_messages(self) -> str:
        template = Template(
            dedent(
                """
        <task>
            {{ task }}
        </task>

        <description>
            {{ description }}
        </description>

        <labels>
        {% for label in labels %}
            <label>
                <name>
                    {{ label.name }}
                </name>

                <description>
                    {{ label.description }}
                </description>

                <examples>
                    <positive>
                    {% for example in label.examples.positive[:n_examples] %}
                        <example>
                            {{ example }}
                        </example>
                    {% endfor %}
                    </positive>

                    <negative>
                    {% for example in label.examples.negative[:n_examples] %}
                        <example>
                            {{ example }}
                        </example>
                    {% endfor %}
                    </negative>
                </examples>
            </label>
        {% endfor %}
        </labels>

        Instructions:
        1. Carefully read the user's query.
        2. Compare the query to the descriptions and examples for each label.
        3. Use the provided examples as a guide:
           - Positive examples show queries that should be classified under that label.
           - Negative examples show queries that should not be classified under that label.
        4. Consider both the content and the intent of the query when matching to a label.
        5. Choose the most appropriate label that matches the query's intent and content.
        6. If the query doesn't clearly fit any label, choose the closest match based on similarity to the examples and description.
        7. Provide your classification as a single word matching the chosen label's name.
        8. Do not assume any specific task unless it's explicitly mentioned in the 'task' variable.
        """
            )
        )
        return template.render(self.model_dump())

    def get_user_query(self, query: str) -> str:
        return f"Correctly Classify:\n\n{query}"

    def get_labels(self) -> List[str]:
        return [label.name for label in self.labels]

    def set_client(self, client: Instructor):
        self._client = client

    def predict(
        self, query: str, model: str, response_model: Type[T], client: Instructor
    ):
        system_message = self.to_system_messages()
        user_query = self.get_user_query(query)
        return client.create(
            model=model,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
            ],
            validation_context={
                "labels": self.get_labels(),
            },
        )

    async def apredict(
        self, query: str, model: str, response_model: Type[T], client: AsyncInstructor
    ):
        system_message = self.to_system_messages()
        user_query = self.get_user_query(query)
        return await client.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
            ],
            model=model,
            response_model=response_model,
            validation_context={
                "labels": self.get_labels(),
            },
        )
