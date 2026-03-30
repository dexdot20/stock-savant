from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel


class FetchUrlContentResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    is_summarized: Optional[bool] = None
    error: Optional[str] = None
    url: Optional[str] = None


class GoogleNewsItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    link: str
    time: Optional[str] = None


class GoogleNewsResult(RootModel[List[GoogleNewsItem]]):
    pass


class KapDisclosureItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    disclosureIndex: int
    stockCode: str
    companyTitle: str
    disclosureClass: str
    publishDate: str


class KapSearchResult(RootModel[List[KapDisclosureItem]]):
    pass


class KapDisclosureDetailResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    disclosureIndex: Optional[int] = None
    stockCode: Optional[str] = None
    companyTitle: Optional[str] = None
    disclosureClass: Optional[str] = None
    publishDate: Optional[str] = None
    summary: Optional[str] = None
    disclosureBodyText: Optional[str] = None
    signatures: Optional[List[Dict[str, Any]]] = None
    attachmentCount: Optional[int] = None
    error: Optional[str] = None


class KapBatchDetailsResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: List[Optional[KapDisclosureDetailResult]] = Field(default_factory=list)
    errors: Dict[str, str] = Field(default_factory=dict)


class PythonExecResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    stdout: Optional[str] = None
    stderr: Optional[str] = None
    result: Optional[Any] = None
    result_repr: Optional[str] = None
    execution_ms: Optional[int] = None
    timed_out: bool = False
    files: List[str] = Field(default_factory=list)
    sandbox_ephemeral: bool = True


class ReportToolResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    report_id: Optional[str] = None
    fingerprint: Optional[str] = None
    duplicate_suppressed: bool = False
    path: Optional[str] = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    data_format: str = "json"
    is_ephemeral: bool = False
    token_cost_hint: Optional[str] = None
