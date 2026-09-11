"""Research exports must preserve direction, activity, and provenance safely."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from connectome_lab.visualization import export_graph_html, export_report_html


def graph(count=3):
    return SimpleNamespace(
        neurons=[SimpleNamespace(id=f"n{i}", neuron_type="test", neurotransmitter="unknown",
                                 role="inter", region="A" if i % 2 else "B",
                                 x=float(i), y=float(i % 2), z=float(-i)) for i in range(count)],
        source=np.arange(count - 1, dtype=int), target=np.arange(1, count, dtype=int),
        synapse_count=np.arange(1, count, dtype=float), metadata={"kind": "synthetic"},
    )


def payload(document, element):
    match = re.search(r'<script id="' + element + r'" type="application/json">(.*?)</script>', document, re.S)
    if match is None:
        raise AssertionError("Embedded data script missing")
    return json.loads(match.group(1))


class GraphExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nested" / "graph.html"

    def test_direction_activity_and_offline_export(self):
        spikes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = export_graph_html(graph(), self.path, spikes=spikes, dt_ms=.5)
        self.assertEqual(result, self.path)
        document = result.read_text(encoding="utf-8")
        data = payload(document, "connectome-data")
        self.assertEqual(data["edges"], [[0, 1, 1.0], [1, 2, 2.0]])
        self.assertEqual(data["activity"], [[0], [1], [2]])
        self.assertEqual(data["frame_times_ms"], [0., .5, 1.])
        self.assertEqual(data["nodes"][1]["indegree"], 1)
        self.assertEqual(data["nodes"][1]["outdegree"], 1)
        self.assertEqual(data["label"], "合成資料")
        self.assertFalse(data["sampled"])
        self.assertIn("來源 → 目標", document)
        self.assertIn("上游 → 選取神經元", document)
        self.assertNotRegex(document, r'<(?:script|link|img)\b[^>]*(?:src|href)=')
        np.testing.assert_array_equal(spikes, np.eye(3, dtype=int))

    def test_untrusted_labels_cannot_break_out_of_script(self):
        attack = '</script><img src=x onerror="alert(1)">\u2028&'
        sample = graph()
        sample.neurons[0].id = attack
        sample.metadata["source"] = attack
        export_graph_html(sample, self.path)
        document = self.path.read_text(encoding="utf-8")
        self.assertNotIn(attack, document)
        self.assertNotIn('<img src=x', document)
        self.assertEqual(payload(document, "connectome-data")["nodes"][0]["id"], attack)
        self.assertIn("\\u003c/script\\u003e", document)

    def test_graph_sampling_is_disclosed_and_indices_remapped(self):
        sample = graph(9)
        sample.source = np.array([0, 0, 2, 4, 6, 8])
        sample.target = np.array([2, 8, 4, 6, 8, 0])
        sample.synapse_count = np.array([1., 9., 3., 4., 5., 6.])
        with patch("connectome_lab.visualization.MAX_GRAPH_NODES", 5), patch("connectome_lab.visualization.MAX_GRAPH_EDGES", 3):
            export_graph_html(sample, self.path)
        data = payload(self.path.read_text(encoding="utf-8"), "connectome-data")
        self.assertTrue(data["sampled"])
        self.assertEqual([node["original_index"] for node in data["nodes"]], [0, 2, 4, 6, 8])
        self.assertEqual(data["edges"], [[0, 4, 9.], [4, 0, 6.], [3, 4, 5.]])
        self.assertEqual(data["original_edges"], 6)
        self.assertEqual(data["nodes"][0]["outdegree"], 2)

    def test_temporal_binning_keeps_short_spikes(self):
        spikes = np.zeros((10, 3), dtype=int)
        spikes[1, 0] = spikes[4, 1] = spikes[9, 2] = 1
        with patch("connectome_lab.visualization.MAX_SPIKE_FRAMES", 2):
            export_graph_html(graph(), self.path, spikes=spikes, dt_ms=2)
        data = payload(self.path.read_text(encoding="utf-8"), "connectome-data")
        self.assertEqual(data["activity"], [[0, 1], [2]])
        self.assertEqual(data["frame_times_ms"], [0., 10.])
        self.assertEqual(data["frame_ends_ms"], [10., 20.])
        self.assertTrue(data["temporal_aggregation"])
        self.assertEqual(data["original_frames"], 10)

    def test_single_neuron_and_zero_activity_frames(self):
        export_graph_html(graph(1), self.path, spikes=np.zeros((0, 1)))
        data = payload(self.path.read_text(encoding="utf-8"), "connectome-data")
        self.assertEqual(data["edges"], [])
        self.assertEqual(data["activity"], [])
        self.assertEqual(data["nodes"][0]["indegree"], 0)

    def test_measured_and_control_labels(self):
        sample = graph()
        for metadata, expected in [({"kind": "measured_larval_subset"}, "實際資料匯入"),
                                   ({"kind": "control", "parent_kind": "measured_larval"}, "對照拓撲"),
                                   ({"kind": "imported"}, "資料來源待核對")]:
            sample.metadata = metadata
            export_graph_html(sample, self.path)
            self.assertEqual(payload(self.path.read_text(encoding="utf-8"), "connectome-data")["label"], expected)

    def test_invalid_inputs_rejected(self):
        for kwargs in ({"dt_ms": 0}, {"dt_ms": np.nan}, {"spikes": np.zeros((2, 2))},
                       {"spikes": [[0, 1, .5]]}, {"spikes": [[0, 1, np.nan]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                export_graph_html(graph(), self.path, **kwargs)
        for mutation in (lambda g: setattr(g, "source", np.array([0, 9])),
                         lambda g: setattr(g, "target", np.array([1., 2.])),
                         lambda g: setattr(g, "synapse_count", np.array([np.inf, 2.])),
                         lambda g: setattr(g.neurons[0], "x", np.nan),
                         lambda g: setattr(g.neurons[0], "id", "n1")):
            sample = graph()
            mutation(sample)
            with self.assertRaises(ValueError):
                export_graph_html(sample, self.path)
        with self.assertRaises(ValueError):
            export_graph_html(graph(0), self.path)


class ReportExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "report.html"
        self.report = {"metadata": {"kind": "synthetic"}, "runs": [{
            "task": "binary", "topology": "structured", "seed": 2,
            "before_success": .2, "after_success": .8,
            "training_rewards": [0., -.5, .7, 1.], "weight_change_l1": .123,
            "interactions": 120, "stability": {"finite": True},
        }]}

    def test_report_has_curves_metrics_and_exact_embedded_data(self):
        export_report_html(self.report, self.path)
        document = self.path.read_text(encoding="utf-8")
        self.assertEqual(payload(document, "report-data"), self.report)
        self.assertIn("<polyline", document)
        self.assertIn("20.0%", document)
        self.assertIn("80.0%", document)
        self.assertIn("+60.0", document)
        self.assertIn("平均成功率變化（百分點）", document)
        self.assertIn("合成資料", document)
        self.assertIn('class="numeric"', document)

    def test_report_escapes_text_and_script_data(self):
        attack = '</script><script>alert("x")</script>'
        self.report["runs"][0]["task"] = attack
        self.report["metadata"]["note"] = attack
        export_report_html(self.report, self.path)
        document = self.path.read_text(encoding="utf-8")
        self.assertNotIn(attack, document)
        self.assertIn("&lt;/script&gt;", document)
        self.assertEqual(payload(document, "report-data"), self.report)
        self.assertEqual(document.count("<script"), 1)

    def test_missing_metrics_and_constant_curve_are_valid(self):
        report = {"runs": [{"task": "one", "training_rewards": [1.]},
                           {"task": "constant", "training_rewards": [0.] * 1200}]}
        export_report_html(report, self.path)
        document = self.path.read_text(encoding="utf-8")
        self.assertIn("未提供完整", document)
        self.assertIn("<circle", document)
        self.assertNotIn("NaN", document)
        export_report_html({"runs": []}, self.path)
        self.assertIn("尚無實驗紀錄", self.path.read_text(encoding="utf-8"))

    def test_invalid_report_values_rejected(self):
        for key, value in (("before_success", 20), ("after_success", -.1),
                           ("training_rewards", [np.nan]), ("training_rewards", [[1, 2]]),
                           ("interactions", -1), ("interactions", .5), ("weight_change_l1", -.1)):
            report = {"runs": [dict(self.report["runs"][0], **{key: value})]}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                export_report_html(report, self.path)
        for report in ([], {"runs": {}}, {"runs": [3]}, {"metadata": []}):
            with self.assertRaises(ValueError):
                export_report_html(report, self.path)


if __name__ == "__main__":
    unittest.main()
