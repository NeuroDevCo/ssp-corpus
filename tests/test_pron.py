"""Test corpus functions."""

import numpy as np
from src import corpus


def test_count_clusters():
    """Test `count_valic_clusters`."""
    assert corpus.count_clusters("CCC") == 0
    assert corpus.count_clusters("VCC") == 1
    assert corpus.count_clusters("CVC") == 1
    assert corpus.count_clusters("CCV") == 1
    assert corpus.count_clusters("VVV") == 1
    assert corpus.count_clusters("V") == 1
    assert corpus.count_clusters("VCC") == 1
    assert corpus.count_clusters("CVVCV") == 2
    assert corpus.count_clusters("CVVCVCCVVV") == 3
    assert corpus.count_clusters("CVVCVCCVVV"), "C" == 3
    assert corpus.count_clusters("CVV CVCC VVV"), "C" == 3


def test_has_one_peak():
    """Test `has_one_peak`."""
    assert corpus.has_one_peak([0])
    assert corpus.has_one_peak([0, 1])
    assert isinstance(corpus.has_one_peak([0, 1, 9]), bool)
    assert corpus.has_one_peak([3, 5, 3])
    assert corpus.has_one_peak(np.array([3, 5, 3]))
    assert corpus.has_one_peak([3, 5, 6, 5, 6]) is False
    assert corpus.has_one_peak([5, 3, 1])
    assert corpus.has_one_peak([5, 1, 4])
    assert corpus.has_one_peak([5, 1, 4, 1]) is False


def test_classify_segments():
    """Test `classify_segments`."""
    assert corpus.classify_segments("ˈli.β̞ɾo")
    assert isinstance(corpus.classify_segments("ˈli.β̞ɾo"), str)
    assert corpus.classify_segments("ˈli.β̞ɾo") == "CVCCV"
    assert corpus.classify_segments("[ˈli.β̞ɾo]") == "CVCCV"
    assert corpus.classify_segments("/ˈli.β̞ɾo/") == "CVCCV"
    assert corpus.classify_segments("fi.las.tˤiːn") == "CVCVCCVC"
    assert corpus.classify_segments("cʰyᵝ.n̟e̞.fɛ") == "CVCVCV"
    assert corpus.classify_segments("ɡwɛiθjɔ") == "CVVVCVV"


def test_extract_corpusority():
    """Test `extract_corpusority`."""
    assert corpus.extract_corpusority("ˈli.β̞ɾo")
    assert isinstance(corpus.extract_corpusority("ˈli.β̞ɾo"), list)
    assert all(isinstance(s, int) for s in corpus.extract_corpusority("ˈli.β̞ɾo"))
    assert all(s >= 1 for s in corpus.extract_corpusority("ˈli.β̞ɾo"))
    assert all(s <= 9 for s in corpus.extract_corpusority("ˈli.β̞ɾo"))
    assert corpus.extract_corpusority("ˈli.β̞ɾo") == [6, 8, 4, 6, 9]
