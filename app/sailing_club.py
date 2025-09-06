from typing import List, Dict, Any, Tuple


def merge_bookings(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge overlapping/adjacent intervals and return sorted merged slots.

    Rules:
    - Intervals are [start, end] with integer hours, end is exclusive.
    - Treat touching intervals as a single busy period: merge when last_end >= start.
    - After merging, sort merged slots by end time ascending (tie-break by start).
    """
    if not intervals:
        return []

    # Filter invalid/zero-length intervals defensively
    cleaned: List[List[int]] = []
    for pair in intervals:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        start, end = pair
        if start is None or end is None:
            continue
        if start == end:
            # Ignore zero-length per constraints (min 1 hour)
            continue
        cleaned.append([start, end])

    if not cleaned:
        return []

    # Sort by start time first for correct merging
    cleaned.sort(key=lambda x: (x[0], x[1]))

    merged: List[List[int]] = []
    for start, end in cleaned:
        if not merged:
            merged.append([start, end])
            continue
        last_start, last_end = merged[-1]
        if last_end >= start:  # overlap or contiguous boundary
            merged[-1][1] = max(last_end, end)
            # last_start is already <= start due to sorting; no need to adjust
        else:
            merged.append([start, end])

    # Sort final merged intervals by end time ascending (tie-break by start)
    # merged.sort(key=lambda x: (x[1], x[0]))
    return merged


def min_boats_needed(intervals: List[List[int]]) -> int:
    """
    Compute the minimum number of boats to satisfy all bookings.

    Uses sweep line over start/end events. Adjacent [a,b] and [b,c] do not overlap
    if end is exclusive. However, the prompt's merging includes boundary touch as
    a busy period; for boat count, only true overlaps need multiple boats.
    We'll treat end as non-inclusive: decrement before increment at same time.
    """
    if not intervals:
        return 0

    events: List[Tuple[int, int]] = []
    for start, end in intervals:
        if start is None or end is None or start == end:
            continue
        events.append((start, 1))   # boat needed starts
        events.append((end, -1))    # boat freed at end

    # Sort events: at the same time, process end (-1) before start (+1)
    events.sort(key=lambda e: (e[0], e[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        if current > peak:
            peak = current
    return peak


def sailing_club(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by FastAPI endpoint.
    Expects payload of the form:
    {
        "testCases": [
            {"id": "0001", "input": [[1,8], [17,28], ...]},
            ...
        ]
    }

    Returns:
    {
        "solutions": [
            {"id": "0001", "sortedMergedSlots": [[...]], "minBoatsNeeded": N},
            ...
        ]
    }
    """
    test_cases = payload.get("testCases", [])
    solutions = []
    for case in test_cases:
        case_id = case.get("id", "")
        intervals = case.get("input", [])

        merged = merge_bookings(intervals)
        boats = min_boats_needed(intervals)

        solutions.append({
            "id": case_id,
            "sortedMergedSlots": merged,
            "minBoatsNeeded": boats,
        })

    return {"solutions": solutions}


