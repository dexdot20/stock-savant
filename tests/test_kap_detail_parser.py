"""
detail_parser modülündeki tüm fonksiyonlar için birim testler.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from services.kap.proxy_manager import ProxyManager
from services.kap.scraper.detail_parser import (
    _extract_json_array,
    _extract_json_object,
    _extract_rsc_strings,
    _find_chunk,
    _get_array_ref_key,
    _get_ref_key,
    _resolve_refs,
    fetch_and_parse,
    parse_disclosure,
)

from tests.kap_test_utils import build_disclosure_rsc_html


# --------------------------------------------------------------------------- #
# _extract_rsc_strings
# --------------------------------------------------------------------------- #

class TestExtractRscStrings:
    def test_extracts_plain_string(self):
        html = '<script>self.__next_f.push([1,"hello world"])</script>'
        assert _extract_rsc_strings(html) == "hello world"

    def test_extracts_json_escaped_content(self):
        html = '<script>self.__next_f.push([1,"\\"key\\":\\"value\\""])</script>'
        result = _extract_rsc_strings(html)
        assert '"key":"value"' in result

    def test_ignores_non_nextf_scripts(self):
        html = '<script>console.log("test")</script><script>window.foo=1</script>'
        assert _extract_rsc_strings(html) == ""

    def test_multiple_pushes_joined_with_newline(self):
        html = (
            '<script>self.__next_f.push([1,"part1"])</script>'
            '<script>self.__next_f.push([1,"part2"])</script>'
        )
        result = _extract_rsc_strings(html)
        assert result == "part1\npart2"

    def test_empty_html_returns_empty_string(self):
        assert _extract_rsc_strings("") == ""

    def test_mixed_scripts_only_extracts_nextf(self):
        html = (
            '<script>ga("send")</script>'
            '<script>self.__next_f.push([1,"data"])</script>'
        )
        assert _extract_rsc_strings(html) == "data"


# --------------------------------------------------------------------------- #
# _extract_json_object
# --------------------------------------------------------------------------- #

class TestExtractJsonObject:
    def test_extracts_flat_object(self):
        text = '"key":{"a":1,"b":2}'
        assert _extract_json_object(text, "key") == {"a": 1, "b": 2}

    def test_extracts_nested_object(self):
        text = '"outer":{"inner":{"x":true}}'
        assert _extract_json_object(text, "outer") == {"inner": {"x": True}}

    def test_returns_none_when_key_missing(self):
        assert _extract_json_object('"other":{"a":1}', "key") is None

    def test_handles_string_values_with_braces(self):
        text = '"key":{"msg":"has {braces} inside"}'
        result = _extract_json_object(text, "key")
        assert result == {"msg": "has {braces} inside"}

    def test_extracts_correct_key_among_multiple(self):
        text = '"first":{"v":1},"second":{"v":2}'
        assert _extract_json_object(text, "second") == {"v": 2}

    def test_returns_none_for_invalid_json(self):
        text = '"key":{"bad": json}'
        assert _extract_json_object(text, "key") is None

    def test_extracts_empty_object(self):
        text = '"key":{}'
        assert _extract_json_object(text, "key") == {}


# --------------------------------------------------------------------------- #
# _extract_json_array
# --------------------------------------------------------------------------- #

class TestExtractJsonArray:
    def test_extracts_simple_array(self):
        text = '"items":[1,2,3]'
        assert _extract_json_array(text, "items") == [1, 2, 3]

    def test_extracts_array_of_objects(self):
        text = '"sigs":[{"name":"A"},{"name":"B"}]'
        assert _extract_json_array(text, "sigs") == [{"name": "A"}, {"name": "B"}]

    def test_returns_none_when_key_missing(self):
        assert _extract_json_array('"other":[1,2]', "items") is None

    def test_extracts_empty_array(self):
        text = '"items":[]'
        assert _extract_json_array(text, "items") == []

    def test_returns_none_for_invalid_json(self):
        text = '"items":[bad]'
        assert _extract_json_array(text, "items") is None


# --------------------------------------------------------------------------- #
# _find_chunk
# --------------------------------------------------------------------------- #

class TestFindChunk:
    def test_finds_t_type_chunk(self):
        rsc = "\nabc:T5,hello\nxyz:other"
        assert _find_chunk(rsc, "abc") == "hello"

    def test_finds_t_type_chunk_at_start(self):
        rsc = "abc:T5,hello\nrest"
        assert _find_chunk(rsc, "abc") == "hello"

    def test_finds_json_chunk(self):
        rsc = '\nabc:{"key":"val"}\n'
        result = _find_chunk(rsc, "abc")
        assert result == '{"key":"val"}'

    def test_returns_none_when_not_found(self):
        assert _find_chunk("other:T4,data", "abc") is None

    def test_t_chunk_length_is_respected(self):
        rsc = "\nk:T10,helloworld extra"
        assert _find_chunk(rsc, "k") == "helloworld"


# --------------------------------------------------------------------------- #
# _resolve_refs
# --------------------------------------------------------------------------- #

class TestResolveRefs:
    def test_returns_plain_value_unchanged(self):
        assert _resolve_refs("plain", "") == "plain"
        assert _resolve_refs(42, "") == 42
        assert _resolve_refs(None, "") is None

    def test_returns_non_ref_dollar_string_unchanged(self):
        assert _resolve_refs("$notfound", "") == "$notfound"

    def test_resolves_t_chunk_ref(self):
        rsc = "\nabc:T5,world"
        result = _resolve_refs("$abc", rsc)
        assert result == "world"

    def test_resolves_dict_values(self):
        rsc = "\nabc:T5,world"
        result = _resolve_refs({"key": "$abc"}, rsc)
        assert result == {"key": "world"}

    def test_resolves_list_values(self):
        rsc = "\nabc:T3,yes"
        result = _resolve_refs(["$abc", "plain"], rsc)
        assert result == ["yes", "plain"]

    def test_depth_limit_prevents_infinite_recursion(self):
        result = _resolve_refs("$abc", "\nabc:T4,test", depth=10)
        assert result == "$abc"

    def test_nested_dict_resolved(self):
        rsc = "\nk1:T4,DATA"
        result = _resolve_refs({"outer": {"inner": "$k1"}}, rsc)
        assert result == {"outer": {"inner": "DATA"}}


# --------------------------------------------------------------------------- #
# _get_ref_key / _get_array_ref_key
# --------------------------------------------------------------------------- #

class TestGetRefKey:
    def test_extracts_ref_key(self):
        rsc = '"disclosureDetail":"$1a"'
        assert _get_ref_key(rsc, "disclosureDetail") == "1a"

    def test_returns_none_when_not_found(self):
        assert _get_ref_key('"other":"$1a"', "disclosureDetail") is None

    def test_extracts_array_ref_key(self):
        rsc = '"disclosureBody":["$2b"]'
        assert _get_array_ref_key(rsc, "disclosureBody") == "2b"

    def test_array_ref_returns_none_when_not_found(self):
        assert _get_array_ref_key('"other":["$2b"]', "disclosureBody") is None


# --------------------------------------------------------------------------- #
# parse_disclosure — entegrasyon
# --------------------------------------------------------------------------- #

class TestParseDisclosure:
    def test_returns_none_for_empty_html(self):
        assert parse_disclosure("<html><body></body></html>") is None

    def test_returns_none_when_disclosure_basic_missing(self):
        html = '<script>self.__next_f.push([1,"no relevant data"])</script>'
        assert parse_disclosure(html) is None

    def test_parses_disclosure_index(self):
        html = build_disclosure_rsc_html(disclosureBasic={"disclosureIndex": 12345})
        result = parse_disclosure(html)
        assert result is not None
        assert result.disclosure_index == 12345

    def test_parses_stock_code(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={"disclosureIndex": 1, "stockCode": "AKBNK"}
        )
        result = parse_disclosure(html)
        assert result.stock_code == "AKBNK"

    def test_parses_company_title(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={"disclosureIndex": 1, "companyTitle": "AKBANK T.A.Ş."}
        )
        result = parse_disclosure(html)
        assert result.company_title == "AKBANK T.A.Ş."

    def test_parses_publish_date_turkish_format(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={"disclosureIndex": 1, "publishDate": "19.02.2026 16:52:20"}
        )
        result = parse_disclosure(html)
        assert result.publish_date == datetime(2026, 2, 19, 16, 52, 20)

    def test_parses_boolean_flags(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={
                "disclosureIndex": 1,
                "isLate": False,
                "isChanged": True,
                "isBlocked": False,
            }
        )
        result = parse_disclosure(html)
        assert result.is_late is False
        assert result.is_changed is True
        assert result.is_blocked is False

    def test_merges_basic_and_detail(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={"disclosureIndex": 1, "stockCode": "THYAO"},
            disclosureDetail={"opinion": "Olumlu", "memberType": "IGS"},
        )
        result = parse_disclosure(html)
        assert result.stock_code == "THYAO"
        assert result.opinion == "Olumlu"
        assert result.member_type == "IGS"

    def test_extracts_disclosure_body_text(self):
        html = build_disclosure_rsc_html(
            disclosureBasic={"disclosureIndex": 1},
            disclosureDetail={"disclosureBody": "<p>Açıklama metni</p>"},
        )
        result = parse_disclosure(html)
        assert result is not None
        if result.disclosure_body_text:
            assert "Açıklama metni" in result.disclosure_body_text


# --------------------------------------------------------------------------- #
# fetch_and_parse — HTTP mock
# --------------------------------------------------------------------------- #

class TestFetchAndParse:
    def test_calls_network_and_returns_detail(self):
        html = build_disclosure_rsc_html(disclosureBasic={"disclosureIndex": 999})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = html

        with patch(
            "services.kap.scraper.detail_parser.fetch_with_retry", return_value=mock_resp
        ):
            result = fetch_and_parse(999)

        assert result is not None
        assert result.disclosure_index == 999

    def test_returns_none_when_html_unparseable(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "<html><body>Hata</body></html>"

        with patch(
            "services.kap.scraper.detail_parser.fetch_with_retry", return_value=mock_resp
        ):
            result = fetch_and_parse(1)

        assert result is None

    def test_passes_proxy_manager(self):
        html = build_disclosure_rsc_html(disclosureBasic={"disclosureIndex": 1})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = html

        with patch(
            "services.kap.scraper.detail_parser.fetch_with_retry", return_value=mock_resp
        ) as mock_fetch:
            fake_pm = MagicMock(spec=ProxyManager)
            fetch_and_parse(1, proxy_manager=fake_pm)
            _, call_kwargs = mock_fetch.call_args
            assert call_kwargs.get("proxy_manager") is fake_pm
