from enum import Enum
from pydantic import BaseModel
from typing import List


class Question(BaseModel):
    text: str


class QuestionType(BaseModel):
    title: str
    description: str
    examples: List[str]


class Product(BaseModel):
    title: str
    description: str


class UntypedQuestion(BaseModel):
    question: Question
    product: Product
    thumbs_up: bool
    days_ago: int


class TypedQuestion(BaseModel):
    question: Question
    question_type: "QuestionTypes"
    product: Product
    thumbs_up: bool
    days_ago: int


class QuestionTypes(str, Enum):
    COMPARISON = "Comparison"
    VAGUE = "Vague"
    TYPICAL_PRICE = "TypicalPrice"
    CUSTOMER_SERVICE = "Customer Service"
    VISUAL = "Visual"
    ACCESSORIES = "Accessories"
    COMPATIBILITY = "Compatibility"
    COUNTRY_OF_ORIGIN = "Country of Origin"
    ENVIRONMENTAL = "Environmental Impact"
    AUTHENTIC = "Authenticity and counterfeits"
    MATERIALS = "Materials"
    TIME_SENSITIVE = "Time Sensitive"
    TREND = "Trend"


question_type_details = {
    QuestionTypes.COMPARISON: QuestionType(
        title="Comparison",
        description="Comparison to other specific products in the same category",
        examples=[
            "Is this more durable than the MX-500?",
            "Is this bigger or smaller than the K-20?",
        ],
    ),
    QuestionTypes.VAGUE: QuestionType(
        title="Vague",
        description="Overall vague (non-specific) product evaluation",
        examples=[
            "Were most people happy with this product overall?",
            "How is the quality of this product?",
        ],
    ),
    QuestionTypes.TYPICAL_PRICE: QuestionType(
        title="TypicalPrice",
        description="Typical price of the product",
        examples=[
            "How much do other online retailers sell this for?",
            "What is the average price of this product?",
        ],
    ),
    QuestionTypes.CUSTOMER_SERVICE: QuestionType(
        title="Customer Service",
        description="Questions about customer service related to the product",
        examples=[
            "Were people able to get customer service help if this broke?",
            "Is there a help-line for this product?",
        ],
    ),
    QuestionTypes.VISUAL: QuestionType(
        title="Visual",
        description="Questions about the look of the product",
        examples=["What are the colors available?", "How shiny is it?"],
    ),
    QuestionTypes.ACCESSORIES: QuestionType(
        title="Accessories",
        description="Questions about accessories related to the product",
        examples=[
            "What else do you recommend to buy with this from the same manufacturer?",
            "What other products are sold with this?",
        ],
    ),
    QuestionTypes.COMPATIBILITY: QuestionType(
        title="Compatibility",
        description="Questions about compatibility of the product with other products",
        examples=[
            "What standards will this work with?",
            "Will this work with my JT-1000?",
        ],
    ),
    QuestionTypes.COUNTRY_OF_ORIGIN: QuestionType(
        title="Country of Origin",
        description="Questions about the country of origin of the product",
        examples=["Is this made in the USA?", "Where is this made?"],
    ),
    QuestionTypes.ENVIRONMENTAL: QuestionType(
        title="Environmental Impact",
        description="Questions about the environmental impact of the product",
        examples=[
            "Did they use renewable energy to manufacture this?",
            "Is this environmentally friendly?",
        ],
    ),
    QuestionTypes.AUTHENTIC: QuestionType(
        title="Authenticity and counterfeits",
        description="Questions related to ensuring the shipped product is what's advertised",
        examples=[
            "Are reviewers sure they got the real thing and it wasn't counterfeit?",
            "Is this a genuine product?",
        ],
    ),
    QuestionTypes.MATERIALS: QuestionType(
        title="Materials",
        description="Questions about the materials used in the product",
        examples=["What is this made of?", "Is the inner part made of carbon steel?"],
    ),
    QuestionTypes.TIME_SENSITIVE: QuestionType(
        title="Time Sensitive",
        description="A question whose answer depends on when it is asked",
        examples=["When will this be back in stock?", "Is this a new product?"],
    ),
    QuestionTypes.TREND: QuestionType(
        title="Trend",
        description="A question related to a current trend",
        examples=[
            "Are recent purchasers happier than the people who purchased longer ago?",
            "Is this getting more popular these days?",
        ],
    ),
}

assert set(QuestionTypes.__members__.keys()) == set(
    [q.name for q in question_type_details.keys()]
)
