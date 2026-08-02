import unittest

from prr_che.parsing import parse_che, parse_prr
from universal_score.parsing import parse_universal


class ParsingTests(unittest.TestCase):
    def test_prr_parser_preserves_checks(self) -> None:
        parsed = parse_prr(
            '{"assessability":"sufficient","checks":{"display_mode":"poster_first"},'
            '"print_ready":true,"reason":"Complete poster."}'
        )
        self.assertTrue(parsed["print_ready"])
        self.assertEqual(parsed["checks"]["display_mode"], "poster_first")

    def test_prr_insufficient_is_never_ready(self) -> None:
        parsed = parse_prr(
            '{"assessability":"insufficient","print_ready":true,"reason":"Image cannot be assessed."}'
        )
        self.assertFalse(parsed["print_ready"])
        self.assertTrue(parsed["warnings"])

    def test_che_parser_computes_mean(self) -> None:
        parsed = parse_che(
            '{"assessability":"sufficient","craftsmanship":{"score":4,"reason":"a"},'
            '"harmony":{"score":3,"reason":"b"},"expressiveness":{"score":2,"reason":"c"}}'
        )
        self.assertEqual(parsed["che_score"], 3.0)

    def test_universal_parser_orders_scores(self) -> None:
        checklist = [f"criterion {index}" for index in range(1, 11)]
        items = [
            {"criterion_index": index, "description": checklist[index - 1], "reason": "ok", "score": index % 6}
            for index in reversed(range(1, 11))
        ]
        import json

        scores, _ = parse_universal(json.dumps({"criteria": items}), checklist)
        self.assertEqual([item["criterion_index"] for item in scores], list(range(1, 11)))
        self.assertEqual(scores[0]["score"], 1)


if __name__ == "__main__":
    unittest.main()
