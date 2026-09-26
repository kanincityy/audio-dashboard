from typing import Literal

from pydantic import BaseModel


class AnalysisOut(BaseModel):
    name: str
    group: str
    label: str
    note: str = ""
    requires: list[str] = []
    prefers: list[str] = []
    cost: Literal["fast", "slow", "paid"]


class AnalysesOut(BaseModel):
    analyses: list[AnalysisOut] = []


class FileOut(BaseModel):
    digest: str
    size_bytes: int
    filename: str


class RunIn(BaseModel):
    analyses: list[str]
    force: bool = False
    allow_billing: bool = False
