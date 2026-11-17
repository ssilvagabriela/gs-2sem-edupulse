import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.ai_summary import SUMMARY_FIELDS, SummaryError, fallback_summary, generate_summary, indicator_evidence


def context(rows=None, risk="alto"):
    return {"historico": rows if rows is not None else [
        {"ano": 2022, "ird": 1.5}, {"ano": 2025, "ird": 0.0}],
        "previsao_2026": {"classe_risco": risk}}


def response(summary=None, finish="STOP"):
    text = json.dumps(summary if summary is not None else dict.fromkeys(SUMMARY_FIELDS, "Texto válido."))
    return io.StringIO(json.dumps({"candidates": [{
        "finishReason": finish, "content": {"parts": [{"text": text}]}}]}))


class SummaryTests(unittest.TestCase):
    def test_other_indicators_keep_missing_zero_and_comparable_years(self):
        data = context([
            {"ano": 2022, "ird": 3, "ied_alto_fun": 50, "aband_fun": 1, "atu_med": 30},
            {"ano": 2024, "ird": 2.8, "ied_alto_fun": None, "aband_fun": 1, "atu_med": 30},
            {"ano": 2025, "ird": 2.6, "ied_alto_fun": 74.4, "aband_fun": 0, "atu_med": None},
        ])
        data["significado_dos_indicadores"] = {
            "ied_alto_fun": "Esforço de trabalho · Ensino Fundamental (%)",
            "aband_fun": "Abandono · Ensino Fundamental (%)",
            "atu_med": "Alunos por turma · Ensino Médio",
        }
        evidence = indicator_evidence(data)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["ano_anterior_disponivel"], 2022)
        self.assertEqual(evidence[0]["variacao"], 24.4)
        self.assertEqual(evidence[0]["unidade"], "pontos percentuais")
        self.assertEqual(evidence[1]["valor"], 0)
        self.assertIn("74,4 em 2025", fallback_summary(data)["indicadores"])

    def test_missing_is_distinct_from_zero(self):
        result = fallback_summary(context([{"ano": 2024, "ird": None}, {"ano": 2025, "ird": 0.0}]))
        self.assertIn("0,00", result["historico"])
        self.assertIn("1 dos 1", result["historico"])
        self.assertIn("Não há valores", fallback_summary(context([]))["historico"])

    def test_recommendation_changes_with_risk(self):
        self.assertIn("Priorizar", fallback_summary(context(risk="alto"))["proximo_passo"])
        self.assertIn("Manter", fallback_summary(context(risk="baixo"))["proximo_passo"])

    @patch("src.ai_summary.urlopen")
    def test_structured_response_and_header_authentication(self, network):
        network.return_value = response()
        self.assertEqual(set(generate_summary(context(), "test-secret")), set(SUMMARY_FIELDS))
        request = network.call_args.args[0]
        self.assertNotIn("test-secret", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-secret")
        body = json.loads(request.data)
        self.assertEqual(json.loads(body["contents"][0]["parts"][0]["text"]), context())
        self.assertEqual(network.call_args.kwargs["timeout"], 25)

    @patch("src.ai_summary.urlopen")
    def test_rejects_truncated_or_invalid_output(self, network):
        for value in (response(finish="MAX_TOKENS"), response({"historico": "Só um campo"}),
                      response(dict.fromkeys(SUMMARY_FIELDS, "")), io.StringIO("not json")):
            with self.subTest(value=value):
                network.return_value = value
                with self.assertRaises(SummaryError):
                    generate_summary(context(), "secret")

    @patch("src.ai_summary.urlopen")
    def test_errors_do_not_expose_credentials_or_response(self, network):
        for error in (HTTPError("https://example.com?key=secret", 429, "secret", {}, None),
                      URLError("secret"), TimeoutError("secret")):
            network.side_effect = error
            with self.assertRaises(SummaryError) as result:
                generate_summary(context(), "secret")
            self.assertNotIn("secret", str(result.exception))

    @patch("src.ai_summary.urlopen")
    def test_missing_credentials_or_bad_model_never_call_network(self, network):
        for key, model in (("", "gemini-3.5-flash-lite"), ("secret", "../malicious")):
            with self.assertRaises(SummaryError):
                generate_summary(context(), key, model)
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
