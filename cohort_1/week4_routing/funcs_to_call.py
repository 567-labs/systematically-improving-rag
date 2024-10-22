from pydantic import BaseModel, Field
from typing import Union


class ShippingDateRequest(BaseModel):
    """Check when a product will be shipped"""

    call_name: str = "ShippingDateRequest"
    sku: str = Field(..., description="SKU of the product to check shipping date for")


class ShippingCostRequest(BaseModel):
    """Check the cost of shipping a product"""

    sku: str = Field(..., description="SKU of the product to check shipping cost for")
    shipping_location: str = Field(..., description="Location to ship to")


class ProductDimensionsRequest(BaseModel):
    """Check the dimensions of a product"""

    sku: str = Field(..., description="SKU of the product to check dimensions for")


class PriceHistoryRequest(BaseModel):
    """Check the price history of a product (e.g. identifying historical price fluctuations)"""

    sku: str = Field(..., description="SKU of the product to check price history for")


class ProductComparisonRequest(BaseModel):
    """Compare two products"""

    sku1: str = Field(..., description="SKU of the first product to compare")
    sku2: str = Field(..., description="SKU of the second product to compare")


class LogDesiredFeatureRequest(BaseModel):
    """Record a user's desire for a certain product feature"""

    sku: str = Field(..., description="SKU of the product to log a desired feature for")
    user_id: str = Field(..., description="User ID to log the desired feature for")
    desired_feature: str = Field(..., description="Desired feature to log")


class ExtractDataFromImageRequest(BaseModel):
    """Use our product images with multimodal llm to extract info about the product"""

    image_url: str = Field(..., description="URL of the image to examine")
    question: str = Field(..., description="Question to answer about the image")


class ProductMaterialsRequest(BaseModel):
    """Check what materials a product is made of"""

    sku: str = Field(..., description="SKU of the product to check materials for")


FunctionOption = Union[
    ShippingDateRequest,
    ShippingCostRequest,
    ProductDimensionsRequest,
    PriceHistoryRequest,
    ProductComparisonRequest,
    LogDesiredFeatureRequest,
    ExtractDataFromImageRequest,
    ProductMaterialsRequest,
]
