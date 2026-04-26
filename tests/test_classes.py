from __future__ import annotations

import pytest

from marauders_map.classes import class_name, resolve_classes


def test_resolve_single_name():
    assert resolve_classes("person") == {0}


def test_resolve_multiple_names():
    assert resolve_classes("person,cat,dog") == {0, 15, 16}


def test_resolve_numeric_ids():
    assert resolve_classes("0,15,16") == {0, 15, 16}


def test_resolve_mixed_with_whitespace():
    assert resolve_classes(" person , 15 , dog ") == {0, 15, 16}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="unknown class"):
        resolve_classes("wizard")


def test_class_name_known():
    assert class_name(0) == "person"
    assert class_name(15) == "cat"


def test_class_name_out_of_range():
    assert class_name(999) == "cls999"
