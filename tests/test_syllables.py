"""Test corpus functions."""

import numpy as np
from src import syllables as syll


def test_count_clusters():
    """Test `count_valic_clusters`."""
    assert syll.count_clusters("CCC") == 0
    assert syll.count_clusters("VCC") == 1
    assert syll.count_clusters("CVC") == 1
    assert syll.count_clusters("CCV") == 1
    assert syll.count_clusters("VVV") == 1
    assert syll.count_clusters("V") == 1
    assert syll.count_clusters("VCC") == 1
    assert syll.count_clusters("CVVCV") == 2
    assert syll.count_clusters("CVVCVCCVVV") == 3
    assert syll.count_clusters("CVVCVCCVVV"), "C" == 3
    assert syll.count_clusters("CVV CVCC VVV"), "C" == 3


def test_classify_segments():
    """Test `classify_segments`."""
    assert syll.classify_segments("ˈli.β̞ɾo")
    assert isinstance(syll.classify_segments("ˈli.β̞ɾo"), str)
    assert syll.classify_segments("ˈli.β̞ɾo") == "CVCCV"
    assert syll.classify_segments("[ˈli.β̞ɾo]") == "CVCCV"
    assert syll.classify_segments("/ˈli.β̞ɾo/") == "CVCCV"
    assert syll.classify_segments("fi.las.tˤiːn") == "CVCVCCVC"
    assert syll.classify_segments("cʰyᵝ.n̟e̞.fɛ") == "CVCVCV"
    assert syll.classify_segments("ɡwɛiθjɔ") == "CVVCV"
    assert syll.classify_segments("ɡwɛiθjɔ", other="V") == "CVVVCVV"
    assert syll.classify_segments("ˈku") == "CV"


def test_sonority():
    """Test `extract_sonority`."""
    assert syll.extract_sonority("ˈli.β̞ɾo")
    assert isinstance(syll.extract_sonority("ˈli.β̞ɾo"), list)
    assert all(isinstance(s, int) for s in syll.extract_sonority("ˈli.β̞ɾo"))
    assert all(s >= 1 for s in syll.extract_sonority("ˈli.β̞ɾo"))
    assert all(s <= 9 for s in syll.extract_sonority("ˈli.β̞ɾo"))
    assert syll.extract_sonority("ˈli.β̞ɾo") == [6, 8, 4, 6, 9]
    assert syll.extract_sonority("ˈku") == [1, 8]


def test_has_one_peak():
    """Test `has_one_peak`."""
    assert syll.has_one_peak([0])
    assert syll.has_one_peak([0, 1])
    assert isinstance(syll.has_one_peak([0, 1, 9]), bool)
    assert syll.has_one_peak([3, 5, 3])
    assert syll.has_one_peak(np.array([3, 5, 3]))
    assert syll.has_one_peak([5, 3, 1])
    assert syll.has_one_peak([1, 8])
    assert not syll.has_one_peak([3, 5, 6, 5, 6])
    assert not syll.has_one_peak([5, 1, 4])
    assert not syll.has_one_peak([5, 1, 4, 1])
