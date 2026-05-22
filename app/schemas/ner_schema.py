from __future__ import annotations

from pydantic import BaseModel, Field


class NerRequestSchema(BaseModel):
    article_id: int = Field(ge=1)
    title: str
    summary: str | None = None


class NerBatchRequestSchema(BaseModel):
    items: list[NerRequestSchema] = Field(min_length=1, max_length=256)


class NerEntityRead(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class NerBatchItemRead(BaseModel):
    index: int = Field(ge=0)
    article_id: int = Field(ge=1)
    entities: list[NerEntityRead] = Field(default_factory=list)


class NerBatchResponseRead(BaseModel):
    data: list[NerBatchItemRead] = Field(default_factory=list)


class InternalServiceHealthRead(BaseModel):
    service: str
    status: str
