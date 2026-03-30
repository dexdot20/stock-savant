from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


def _parse_turkish_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None

    try:
        return datetime.strptime(value, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


class DisclosureItem(BaseModel):
    disclosure_index: int = Field(alias="disclosureIndex")
    stock_code: Optional[str] = Field(default=None, alias="stockCode")
    company_title: Optional[str] = Field(default=None, alias="companyTitle")
    disclosure_class: Optional[str] = Field(default=None, alias="disclosureClass")
    disclosure_type: Optional[str] = Field(default=None, alias="disclosureType")
    disclosure_category: Optional[str] = Field(default=None, alias="disclosureCategory")
    publish_date: Optional[datetime] = Field(default=None, alias="publishDate")
    related_stocks: Optional[List[str]] = Field(default=None, alias="relatedStocks")
    detail_url: Optional[str] = Field(default=None)

    model_config = {"populate_by_name": True}

    @field_validator("publish_date", mode="before")
    @classmethod
    def parse_publish_date(cls, value: Any) -> Optional[datetime]:
        return _parse_turkish_datetime(value)


class SignatureInfo(BaseModel):
    user_name: Optional[str] = Field(default=None, alias="userName")
    title: Optional[str] = Field(default=None)
    company_name: Optional[str] = Field(default=None, alias="companyName")
    sign_date: Optional[str] = Field(default=None, alias="signDate")

    model_config = {"populate_by_name": True}


class DisclosureDetail(BaseModel):
    disclosure_index: Optional[int] = Field(default=None, alias="disclosureIndex")
    mkk_member_oid: Optional[str] = Field(default=None, alias="mkkMemberOid")
    company_title: Optional[str] = Field(default=None, alias="companyTitle")
    stock_code: Optional[str] = Field(default=None, alias="stockCode")
    disclosure_class: Optional[str] = Field(default=None, alias="disclosureClass")
    disclosure_type: Optional[str] = Field(default=None, alias="disclosureType")
    disclosure_category: Optional[str] = Field(default=None, alias="disclosureCategory")
    publish_date: Optional[datetime] = Field(default=None, alias="publishDate")
    summary: Optional[str] = Field(default=None)
    attachment_count: Optional[int] = Field(default=None, alias="attachmentCount")
    year: Optional[int] = Field(default=None)
    period: Optional[str] = Field(default=None)
    is_late: Optional[bool] = Field(default=None, alias="isLate")
    is_changed: Optional[bool] = Field(default=None, alias="isChanged")
    is_blocked: Optional[bool] = Field(default=None, alias="isBlocked")
    sender_type: Optional[str] = Field(default=None, alias="senderType")
    related_disclosure_oid: Optional[str] = Field(default=None, alias="relatedDisclosureOid")

    ft_niteligi: Optional[str] = Field(default=None, alias="ftNiteligi")
    opinion: Optional[str] = Field(default=None)
    opinion_type: Optional[Any] = Field(default=None, alias="opinionType")
    audit_type: Optional[Any] = Field(default=None, alias="auditType")
    main_disclosure_document_id: Optional[str] = Field(
        default=None,
        alias="mainDisclosureDocumentId",
    )
    member_type: Optional[str] = Field(default=None, alias="memberType")

    signatures: Optional[list[SignatureInfo]] = Field(default=None)
    disclosure_body: Optional[str] = Field(default=None, alias="disclosureBody")
    disclosure_body_text: Optional[str] = Field(default=None, alias="disclosureBodyText")

    model_config = {"populate_by_name": True}

    @field_validator("publish_date", mode="before")
    @classmethod
    def parse_publish_date(cls, value: Any) -> Optional[datetime]:
        return _parse_turkish_datetime(value)