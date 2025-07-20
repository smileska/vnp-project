from typing import Dict, Any, Tuple


def count_total_fields(obj: Any) -> int:
    if obj is None:
        return 1

    if isinstance(obj, (str, int, float, bool)):
        return 1

    if isinstance(obj, list):
        return sum(count_total_fields(item) for item in obj)

    if isinstance(obj, dict):
        return sum(count_total_fields(value) for value in obj.values())

    return 0


def compare_values(actual: Any, predicted: Any) -> Tuple[int, int]:

    if actual is None and predicted is None:
        return 1, 1

    if actual is None or predicted is None:
        total_fields = count_total_fields(actual) if actual is not None else count_total_fields(predicted)
        return 0, total_fields

    if isinstance(actual, (str, int, float, bool)) and isinstance(predicted, (str, int, float, bool)):
        return (1, 1) if actual == predicted else (0, 1)

    if isinstance(actual, list) and isinstance(predicted, list):
        if len(actual) != len(predicted):
            total_actual = sum(count_total_fields(item) for item in actual)
            total_predicted = sum(count_total_fields(item) for item in predicted)
            total_fields = max(total_actual, total_predicted)
            correct = 0
            for i in range(min(len(actual), len(predicted))):
                item_correct, _ = compare_values(actual[i], predicted[i])
                correct += item_correct
            return correct, total_fields

        correct = 0
        total = 0
        for a_item, p_item in zip(actual, predicted):
            item_correct, item_total = compare_values(a_item, p_item)
            correct += item_correct
            total += item_total
        return correct, total

    if isinstance(actual, dict) and isinstance(predicted, dict):
        all_keys = set(actual.keys()) | set(predicted.keys())
        correct = 0
        total = 0

        for key in all_keys:
            a_val = actual.get(key)
            p_val = predicted.get(key)
            item_correct, item_total = compare_values(a_val, p_val)
            correct += item_correct
            total += item_total

        return correct, total

    total_actual = count_total_fields(actual)
    total_predicted = count_total_fields(predicted)
    total_fields = max(total_actual, total_predicted)
    return 0, total_fields


def calculate_json_accuracy(actual: Dict[str, Any], predicted: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:

    if actual == predicted:
        return 1.0, {"correct_fields": count_total_fields(actual), "total_fields": count_total_fields(actual)}

    correct_fields, total_fields = compare_values(actual, predicted)

    if total_fields == 0:
        accuracy = 1.0
    else:
        accuracy = correct_fields / total_fields

    diff_info = {
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "accuracy": round(accuracy, 4)
    }

    return round(accuracy, 4), diff_info