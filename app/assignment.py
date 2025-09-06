from typing import Dict, List, Any


def _validate_input(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")
    customers = payload.get("customers")
    concerts = payload.get("concerts")
    priority = payload.get("priority", {})

    if not isinstance(customers, list) or not isinstance(concerts, list):
        raise ValueError("customers and concerts must be arrays")
    if len(customers) < 1 or len(concerts) < 1:
        raise ValueError("customers and concerts must be non-empty")
    if priority is not None and not isinstance(priority, dict):
        raise ValueError("priority must be an object")


def _squared_distance(a: List[int], b: List[int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def assign_concerts(payload: Dict[str, Any]) -> Dict[str, str]:
    _validate_input(payload)
    customers: List[Dict[str, Any]] = payload["customers"]
    concerts: List[Dict[str, Any]] = payload["concerts"]
    priority: Dict[str, str] = payload.get("priority", {})

    concert_name_to_location: Dict[str, List[int]] = {
        c["name"]: c["booking_center_location"] for c in concerts
    }

    result: Dict[str, str] = {}

    for customer in customers:
        customer_name = customer.get("name")
        customer_location = customer.get("location")
        credit_card = customer.get("credit_card")

        preferred_concert = priority.get(credit_card)
        if preferred_concert and preferred_concert in concert_name_to_location:
            result[customer_name] = preferred_concert
            continue

        best_concert_name = None
        best_dist = float("inf")
        for concert in concerts:
            dist = _squared_distance(customer_location, concert["booking_center_location"])
            if dist < best_dist:
                best_dist = dist
                best_concert_name = concert["name"]

        result[customer_name] = best_concert_name  # type: ignore[assignment]

    return result


