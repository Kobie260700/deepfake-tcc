import unittest
import csv
import tempfile
from pathlib import Path

from evaluate_sessions import conclusion_from_assessment_file, period_summary_lines, summarize


class EvaluationTests(unittest.TestCase):
    def test_summarizes_assessments(self):
        rows = [
            {
                "sessao": "s1", "risco_acumulado": "10", "distancia_identidade": "3",
                "identidade_compativel": "True", "liveness_valido": "True",
                "probabilidade_deepfake": "0.1", "nivel_risco": "BAIXO", "rotulo": "real",
            },
            {
                "sessao": "s2", "risco_acumulado": "80", "distancia_identidade": "30",
                "identidade_compativel": "False", "liveness_valido": "False",
                "probabilidade_deepfake": "0.9", "nivel_risco": "ALTO", "rotulo": "fake",
            },
            {
                "sessao": "s2", "risco_acumulado": "70", "distancia_identidade": "25",
                "identidade_compativel": "False", "liveness_valido": "False",
                "probabilidade_deepfake": "0.8", "nivel_risco": "MODERADO", "rotulo": "fake",
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["sessoes"], 2)
        self.assertEqual(result["identidade_compativel_percentual"], 33.33)
        self.assertEqual(result["alertas_moderados_ou_altos"], 2)
        self.assertEqual(result["classificacao"]["unidade"], "sessao")
        self.assertEqual(result["classificacao"]["sessoes_rotuladas"], 2)

    def test_rebuilds_conclusion_from_legacy_assessment_file(self):
        headers = [
            "identidade_compativel",
            "probabilidade_deepfake",
            "liveness_valido",
            "risco_acumulado",
            "nivel_risco",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "avaliacoes.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerow({
                    "identidade_compativel": "False",
                    "probabilidade_deepfake": "0.185734",
                    "liveness_valido": "True",
                    "risco_acumulado": "75.0",
                    "nivel_risco": "MODERADO",
                })
            result = conclusion_from_assessment_file(path, "Guilherme")

        self.assertEqual(result["cycles"], 1)
        self.assertEqual(result["identity_compatible_percent"], 0.0)
        self.assertEqual(result["liveness_valid_percent"], 100.0)
        self.assertEqual(result["deepfake_average_percent"], 18.57)
        self.assertEqual(result["maximum_accumulated_risk"], 75.0)

    def test_formats_period_details(self):
        data = {
            "periods": [{"periodo": 1, "analise_completa": True}],
            "assessments": [{
                "identidade_compativel": "False",
                "liveness_valido": "True",
                "probabilidade_deepfake": "0.0153",
                "risco_acumulado": "75",
            }],
        }
        line = period_summary_lines(data)[0]
        self.assertIn("referência divergente", line)
        self.assertIn("alteração 1.53%", line)


if __name__ == "__main__":
    unittest.main()
