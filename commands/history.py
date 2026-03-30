"""History commands"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime

from core import get_standard_logger

logger = get_standard_logger(__name__)


# Persistent history storage
import json

from core.paths import get_history_dir

HISTORY_DIR = get_history_dir()


def _get_next_id() -> int:
    """Get the next available history ID by checking files."""
    files = list(HISTORY_DIR.glob("*.json"))
    if not files:
        return 1

    ids = []
    for f in files:
        try:
            # Filename format: {id}_{symbol}_{timestamp}.json
            file_id = int(f.stem.split("_")[0])
            ids.append(file_id)
        except (ValueError, IndexError):
            continue

    return max(ids, default=0) + 1


def add_to_history(
    symbol: str,
    analysis_type: str,
    data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Add an analysis to persistent history

    Args:
        symbol: Stock ticker symbol
        analysis_type: Type of analysis (e.g., 'stock')
        data: Analysis data
        config: Configuration dictionary

    Returns:
        History entry ID
    """
    entry_id = _get_next_id()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save to JSON file
    filename = f"{entry_id}_{symbol.upper()}_{timestamp}.json"
    file_path = HISTORY_DIR / filename

    # Save both a lightweight projection for quick listing/inspection and
    # the full original analysis artifact for complete context preservation.
    final_data_to_save = data
    if analysis_type == "stock" and isinstance(data, dict):
        workflow = data.get("workflow", {})

        # Extract News Analysis
        news_analysis_node = workflow.get("news_analysis", {})
        news_analysis_text = ""
        if isinstance(news_analysis_node, dict):
            news_analysis_text = news_analysis_node.get("analysis", "")
        elif isinstance(news_analysis_node, str):
            news_analysis_text = news_analysis_node

        # Extract Final Analysis
        investment_decision_node = workflow.get("investment_decision", {})
        final_analysis_text = ""
        if isinstance(investment_decision_node, dict):
            final_analysis_text = investment_decision_node.get("content", "")
        elif isinstance(investment_decision_node, str):
            final_analysis_text = investment_decision_node

        # Append technical info to the end of the analysis text
        model_name = investment_decision_node.get("model", "unknown")
        usage_info = investment_decision_node.get("usage")

        tech_suffix = f"\n\n---\n**Technical Info:**\n- **Model:** {model_name}"
        if usage_info:
            in_t = usage_info.get("input_tokens", 0)
            out_t = usage_info.get("output_tokens", 0)
            tech_suffix += f"\n- **Tokens:** {in_t} Input / {out_t} Output (Total: {usage_info.get('total_tokens', 0)})"

        final_analysis_text += tech_suffix

        # Extract Links
        # 'news_articles' in workflow is the list sent to AI
        news_articles = workflow.get("news_articles", [])
        links = []
        if isinstance(news_articles, list):
            for article in news_articles:
                if isinstance(article, dict):
                    url = article.get("link") or article.get("url")
                    if url:
                        links.append(url)

        summary_projection = {
            "news_analysis_output": news_analysis_text,
            "final_analysis_output": final_analysis_text,
            "site_links": links,
            "model_used": investment_decision_node.get("model", "unknown"),
            "usage": {
                "news": news_analysis_node.get("usage"),
                "reasoner": investment_decision_node.get("usage"),
            },
            "symbol": symbol.upper(),
        }

        final_data_to_save = {
            **summary_projection,
            "summary_projection": summary_projection,
            "full_artifact": data,
            "history_format_version": 2,
        }

    entry = {
        "id": entry_id,
        "symbol": symbol.upper(),
        "type": analysis_type,
        "data": final_data_to_save,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Analysis saved to file: %s", file_path)
    except Exception as e:
        logger.error("Error while saving analysis (%s): %s", symbol, e)
        # Fallback to older behavior if needed, but here we just raise or log
        raise

    return entry_id


def list_history(
    limit: int = 10,
    symbol: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    List analysis history from files

    Args:
        limit: Maximum number of entries to return
        symbol: Filter by symbol (optional)
        config: Configuration dictionary

    Returns:
        List of history entries summary
    """
    files = list(HISTORY_DIR.glob("*.json"))

    entries = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                entry = json.load(file)
                # Keep only summary info for listing
                entries.append(
                    {
                        "id": entry["id"],
                        "symbol": entry["symbol"],
                        "type": entry["type"],
                        "timestamp": entry["timestamp"],
                        "file_path": str(f),
                    }
                )
        except Exception as e:
            logger.warning("History file could not be read %s: %s", f, e)
            continue

    # Filter by symbol if provided
    if symbol:
        entries = [e for e in entries if e["symbol"].upper() == symbol.upper()]

    # Sort by timestamp (newest first)
    entries = sorted(entries, key=lambda x: x["timestamp"], reverse=True)

    # Limit results
    return entries[:limit]


def show_history_entry(
    analysis_id: int, config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a specific history entry from file

    Args:
        analysis_id: History entry ID
        config: Configuration dictionary

    Returns:
        History entry or None if not found
    """
    files = list(HISTORY_DIR.glob(f"{analysis_id}_*.json"))

    if not files:
        return None

    # Take the first one matching the ID
    file_path = files[0]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Analysis record could not be read %s: %s", file_path, e)
        return None
