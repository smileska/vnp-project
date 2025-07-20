from Levenshtein import distance


def calculate_text_similarity(original: str, predicted: str) -> float:
    if original == predicted:
        return 1.0

    if not original or not predicted:
        return 0.0

    original_norm = original.strip().lower()
    predicted_norm = predicted.strip().lower()

    levenshtein_distance = distance(original_norm, predicted_norm)

    max_length = max(len(original_norm), len(predicted_norm))
    if max_length == 0:
        return 1.0

    similarity = 1 - (levenshtein_distance / max_length)
    return round(similarity, 4)