from __future__ import annotations

from marauders_map.app import _parse_names


def test_parse_names_empty():
    assert _parse_names(None) == {}
    assert _parse_names("") == {}


def test_parse_names_orders_by_global_id():
    assert _parse_names("Vlad,Hermione,Ron") == {1: "Vlad", 2: "Hermione", 3: "Ron"}


def test_parse_names_strips_whitespace_and_blanks():
    assert _parse_names(" Vlad , , Ron ") == {1: "Vlad", 3: "Ron"}
