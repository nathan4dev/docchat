from core.ranking import rank_results


def test_sorts_by_distance_ascending():
    results = [{"distance": 0.9}, {"distance": 0.1}, {"distance": 0.5}]
    ranked = rank_results(results)
    assert [r["distance"] for r in ranked] == [0.1, 0.5, 0.9]


def test_top_k_truncates():
    results = [{"distance": d} for d in [0.1, 0.2, 0.3, 0.4]]
    ranked = rank_results(results, top_k=2)
    assert len(ranked) == 2


def test_max_distance_filters_out_weak_matches():
    results = [{"distance": 0.1}, {"distance": 0.9}]
    ranked = rank_results(results, max_distance=0.5)
    assert ranked == [{"distance": 0.1}]


def test_empty_input():
    assert rank_results([]) == []
