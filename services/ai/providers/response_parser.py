import json
import re
from typing import Any, Dict, Optional, List


class ResponseParser:
    """Parses model responses into structured investment outputs."""

    _DECISION_ALIASES = {
        "BUY": "BUY",
        "STRONG_BUY": "BUY",
        "BULLISH": "BUY",
        "LONG": "BUY",
        "SELL": "SELL",
        "STRONG_SELL": "SELL",
        "BEARISH": "SELL",
        "SHORT": "SELL",
        "WAIT": "WAIT",
        "HOLD": "WAIT",
        "NEUTRAL": "WAIT",
        "NO_ACTION": "WAIT",
    }

    @classmethod
    def _normalize_decision(cls, value: Any, default: str = "WAIT") -> str:
        if not isinstance(value, str):
            return default
        normalized = re.sub(r"[^A-Z_]", "_", value.strip().upper())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            return default
        return cls._DECISION_ALIASES.get(normalized, default)

    @staticmethod
    def _normalize_risk_score(value: Any, default: int = 50) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default

        if numeric <= 1:
            numeric *= 100
        elif numeric <= 10:
            numeric *= 10

        numeric = max(0.0, min(100.0, numeric))
        return int(round(numeric))

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_-]*", "", stripped)
            stripped = re.sub(r"```$", "", stripped)
            stripped = stripped.strip()
        return stripped

    def parse_json_from_text(self, text: str) -> Optional[Any]:
        if not text:
            return None

        candidate = self._strip_code_fences(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, candidate, re.DOTALL)
            if not match:
                continue
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
        return None

    def parse_indices(self, content: str) -> List[int]:
        """Parses a list of indices from AI response."""
        if not content:
            return []

        candidate = self._strip_code_fences(content)

        # Extract anything that looks like a JSON array [1, 2, 3]
        array_match = re.search(r"\[[\d\s,]+\]", candidate)
        if array_match:
            candidate = array_match.group(0)

        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [int(x) for x in data if str(x).isdigit()]
            elif isinstance(data, dict) and "indices" in data:
                return [int(x) for x in data["indices"] if str(x).isdigit()]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: find all digits
        return [int(d) for d in re.findall(r"\d+", candidate)]

    def parse_ai_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse free-form or JSON responses from the reasoner."""
        result = {
            "decision": "WAIT",
            "risk_score": 50,
            "reasoning": response_text,
            "raw_response": response_text,
        }

        if not response_text or not isinstance(response_text, str):
            return result

        def _merge_structured_data(payload: Any) -> bool:
            if not isinstance(payload, dict):
                return False

            result["structured"] = payload

            decision_candidate = payload.get("decision")
            if isinstance(decision_candidate, str) and decision_candidate.strip():
                result["decision"] = self._normalize_decision(decision_candidate)

            risk_candidate = payload.get("risk_score")
            result["risk_score"] = self._normalize_risk_score(
                risk_candidate, result["risk_score"]
            )

            for key in (
                "scores",
                "analysis_steps",
                "total_score",
                "thesis",
                "sentiment_score",
                "impact_level",
                "executive_summary",
                "key_themes",
                "risks",
                "opportunities",
                "sources",
                "decision_strength",
            ):
                if key in payload:
                    result[key] = payload[key]

            # Normalize decision_strength if present
            if "decision_strength" in result:
                strength = result["decision_strength"]
                if isinstance(strength, str):
                    result["decision_strength"] = strength.strip().upper()
                    if result["decision_strength"] not in (
                        "STRONG",
                        "MODERATE",
                        "WEAK",
                    ):
                        result["decision_strength"] = "MODERATE"

            reasoning_candidate = payload.get("reasoning") or payload.get("summary")

            if not reasoning_candidate:
                thesis_block = (
                    payload.get("thesis") if isinstance(payload, dict) else None
                )
                if isinstance(thesis_block, dict):
                    reasoning_candidate = thesis_block.get("big_picture")

            if not reasoning_candidate and isinstance(
                payload.get("analysis_steps"), list
            ):
                insights = [
                    step.get("insight")
                    for step in payload["analysis_steps"]
                    if isinstance(step, dict)
                ]
                reasoning_candidate = ". ".join(filter(None, insights))

            if isinstance(reasoning_candidate, str) and reasoning_candidate.strip():
                result["reasoning"] = reasoning_candidate.strip()

            return True

        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                parsed_json = None
            if _merge_structured_data(parsed_json):
                return result

        try:
            parsed_json = json.loads(response_text)
        except (TypeError, json.JSONDecodeError):
            parsed_json = None

        if _merge_structured_data(parsed_json):
            return result

        decision_match = re.search(
            r"(?:decision|recommendation)[\s:]*\**\s*([A-Z]+)",
            response_text,
            re.IGNORECASE,
        )
        if decision_match:
            result["decision"] = self._normalize_decision(decision_match.group(1))

        risk_match = re.search(
            r"(?:risk\s+score|risk_score)[\s:]*(\d+)",
            response_text,
            re.IGNORECASE,
        )
        if risk_match:
            result["risk_score"] = self._normalize_risk_score(
                risk_match.group(1), result["risk_score"]
            )

        result["reasoning"] = response_text
        return result
