"""Orders and filters raw retrieval results before they're handed to the prompt builder."""


def rank_results(results: list[dict], top_k: int | None = None, max_distance: float | None = None) -> list[dict]:
    """Sorts retrieved chunks by distance (lower = more similar) and optionally filters/truncates.

    Chroma already returns results roughly sorted, but we sort explicitly so this
    doesn't silently depend on store implementation details.
    """
    sorted_results = sorted(results, key=lambda r: r["distance"])

    if max_distance is not None:
        sorted_results = [r for r in sorted_results if r["distance"] <= max_distance]

    if top_k is not None:
        sorted_results = sorted_results[:top_k]

    return sorted_results
