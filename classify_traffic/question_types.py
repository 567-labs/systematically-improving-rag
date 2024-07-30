from enum import Enum
from pydantic import BaseModel


class Question(BaseModel):
    text: str


class QuestionType(BaseModel):
    title: str
    description: str
    example: str


class Product(BaseModel):
    title: str
    description: str


class UntypedQuestion(BaseModel):
    question: Question
    product: Product
    thumbs_up: bool


class TypedQuestion(BaseModel):
    question: Question
    question_type: "QuestionTypes"  # Updated to use the Enum
    product: Product
    thumbs_up: bool


class QuestionTypes(str, Enum):
    COMPARISON = "Comparison"
    GENERAL = "General"
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


question_type_details = {
    QuestionTypes.COMPARISON: QuestionType(
        title="Comparison",
        description="Comparison to other specific products in the same category",
        example="Is this more durable than the MX-500?",
    ),
    QuestionTypes.GENERAL: QuestionType(
        title="General",
        description="Overall vague (non-specific) product evaluation",
        example="Were most people happy with this product overall?",
    ),
    QuestionTypes.TYPICAL_PRICE: QuestionType(
        title="TypicalPrice",
        description="Typical price of the product",
        example="How much do other online retailers sell this for?",
    ),
    QuestionTypes.CUSTOMER_SERVICE: QuestionType(
        title="Customer Service",
        description="Questions about customer service related to the product",
        example="Were people able to get customer service help if this broke?",
    ),
    QuestionTypes.VISUAL: QuestionType(
        title="Visual",
        description="Questions about the look of the product",
        example="What are the colors available?",
    ),
    QuestionTypes.ACCESSORIES: QuestionType(
        title="Accessories",
        description="Questions about accessories related to the product",
        example="What else do you recommend to buy with this from the same manufacturer?",
    ),
    QuestionTypes.COMPATIBILITY: QuestionType(
        title="Compatibility",
        description="Questions about compatibility of the product with other products",
        example="What standards will this work with?",
    ),
    QuestionTypes.COUNTRY_OF_ORIGIN: QuestionType(
        title="Country of Origin",
        description="Questions about the country of origin of the product",
        example="Is this made in the USA?",
    ),
    QuestionTypes.ENVIRONMENTAL: QuestionType(
        title="Environmental Impact",
        description="Questions about the environmental impact of the product",
        example="Did they use renewable energy to manufacture this?",
    ),
    QuestionTypes.AUTHENTIC: QuestionType(
        title="Authenticity and counterfeits",
        description="Questions related to ensuring the shipped product is what's advertised",
        example="Are reviewers sure they got the real thing and it wasn't counterfeit?",
    ),
    QuestionTypes.MATERIALS: QuestionType(
        title="Materials",
        description="Questions about the materials used in the product",
        example="What is this made of?",
    ),
    QuestionTypes.TIME_SENSITIVE: QuestionType(
        title="Time Sensitive",
        description="A question whose answer depends on when it is asked",
        example="When will this be back in stock?",
    ),
}
