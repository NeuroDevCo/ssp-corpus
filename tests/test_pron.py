"""Test corpus functions."""

import numpy as np
from src import phonology


def test_count_clusters():
    """Test `count_valic_clusters`."""
    assert phonology.count_clusters("CCC") == 0
    assert phonology.count_clusters("VCC") == 1
    assert phonology.count_clusters("CVC") == 1
    assert phonology.count_clusters("CCV") == 1
    assert phonology.count_clusters("VVV") == 1
    assert phonology.count_clusters("V") == 1
    assert phonology.count_clusters("VCC") == 1
    assert phonology.count_clusters("CVVCV") == 2
    assert phonology.count_clusters("CVVCVCCVVV") == 3
    assert phonology.count_clusters("CVVCVCCVVV"), "C" == 3
    assert phonology.count_clusters("CVV CVCC VVV"), "C" == 3


def test_has_one_peak():
    """Test `has_one_peak`."""
    assert phonology.has_one_peak([0])
    assert phonology.has_one_peak([0, 1])
    assert isinstance(phonology.has_one_peak([0, 1, 9]), bool)
    assert phonology.has_one_peak([3, 5, 3])
    assert phonology.has_one_peak(np.array([3, 5, 3]))
    assert phonology.has_one_peak([3, 5, 6, 5, 6]) is False
    assert phonology.has_one_peak([5, 3, 1])
    assert not phonology.has_one_peak([5, 1, 4])
    assert phonology.has_one_peak([5, 1, 4, 1]) is False


def test_classify_segments():
    """Test `classify_segments`."""
    assert phonology.classify_segments("ˈli.β̞ɾo")
    assert isinstance(phonology.classify_segments("ˈli.β̞ɾo"), str)
    assert phonology.classify_segments("ˈli.β̞ɾo") == "CVCCV"
    assert phonology.classify_segments("[ˈli.β̞ɾo]") == "CVCCV"
    assert phonology.classify_segments("/ˈli.β̞ɾo/") == "CVCCV"
    assert phonology.classify_segments("fi.las.tˤiːn") == "CVCVCCVC"
    assert phonology.classify_segments("cʰyᵝ.n̟e̞.fɛ") == "CVCVCV"
    assert phonology.classify_segments("ɡwɛiθjɔ") == "CVVVCVV"


def test_extract_corpusority():
    """Test `extract_corpusority`."""
    assert phonology.extract_corpusority("ˈli.β̞ɾo")
    assert isinstance(phonology.extract_corpusority("ˈli.β̞ɾo"), list)
    assert all(isinstance(s, int) for s in phonology.extract_corpusority("ˈli.β̞ɾo"))
    assert all(s >= 1 for s in phonology.extract_corpusority("ˈli.β̞ɾo"))
    assert all(s <= 9 for s in phonology.extract_corpusority("ˈli.β̞ɾo"))
    assert phonology.extract_corpusority("ˈli.β̞ɾo") == [6, 8, 4, 6, 9]
