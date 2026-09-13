"""Exercise public CLI workflows with short, local, inspectable fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from connectome_lab.cli import main
from connectome_lab.data import load_sqlite


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(map(str, args)))
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        return json.loads(stdout.getvalue().splitlines()[-1])

    def reject(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            main(list(map(str, args)))
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("error:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        return stderr.getvalue()

    def test_import_and_directed_query_preserve_large_neuron_ids(self):
        a, b, c = "720575940123456789", "720575940123456790", "720575940123456791"
        neurons, synapses = self.folder / "neurons.csv", self.folder / "edges.csv"
        neurons.write_text(f"root_id,cell_type,nt_type,region\n{a},A,ACH,left\n{b},B,GABA,left\n{c},C,unknown,right\n",
                           encoding="utf-8")
        synapses.write_text(f"pre_root_id,post_root_id,syn_count,neuropil\n{a},{b},2,AL\n{a},{b},3,MB\n{b},{c},4,MB\n",
                            encoding="utf-8")
        database = self.folder / "graph.sqlite"
        result = self.invoke("import-csv", "--neurons", neurons, "--synapses", synapses, "--output", database)
        self.assertEqual(result, {"database": str(database), "neurons": 3, "edges": 2})
        graph = load_sqlite(database)
        self.assertEqual([n.id for n in graph.neurons], [a, b, c])
        np.testing.assert_array_equal(graph.synapse_count, [5, 4])
        downstream = self.invoke("query", "--graph", database, "--neurons", a, "--hops", 2)
        self.assertEqual(downstream["neurons"], [a, b, c])
        upstream = self.invoke("query", "--graph", database, "--neurons", b,
                               "--direction", "upstream", "--hops", 1)
        self.assertEqual(upstream["neurons"], [a, b])
        self.assertEqual(upstream["count"], 2)
        original = database.read_bytes()
        self.reject("import-csv", "--synapses", synapses, "--output", database)
        self.assertEqual(database.read_bytes(), original)
        replaced = self.invoke("import-csv", "--synapses", synapses, "--output", database, "--overwrite")
        self.assertEqual(replaced["edges"], 2)

    def test_graph_command_produces_offline_html_with_matching_data(self):
        output = self.folder / "nested" / "connectome.html"
        result = self.invoke("graph", "--graph", "synthetic", "--seed", 8, "--output", output)
        self.assertEqual(result["html"], str(output))
        document = output.read_text(encoding="utf-8")
        match = re.search(r'<script id="connectome-data" type="application/json">(.*?)</script>', document, re.S)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual(len(payload["nodes"]), result["neurons"])
        self.assertEqual(len(payload["edges"]), result["edges"])
        self.assertNotRegex(document, r'<(?:script|link|img)\b[^>]*(?:src|href)=')

    def test_simulate_writes_finite_spikes_voltage_and_declared_plasticity(self):
        for rule in ("none", "reward_stdp"):
            with self.subTest(rule=rule):
                output = self.folder / rule
                result = self.invoke("simulate", "--steps", 25, "--rule", rule,
                                     "--seed", 3, "--output", output)
                self.assertTrue(result["activity"]["all_finite"])
                metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["plasticity"]["rule"], rule)
                self.assertEqual(metrics["steps"], 25)
                self.assertEqual(metrics["stimulus_steps"], 13)
                self.assertTrue(metrics["weights_finite"])
                with np.load(output / "activity.npz", allow_pickle=False) as archive:
                    self.assertEqual(archive["spikes"].shape, (25, metrics["neurons"]))
                    self.assertEqual(archive["voltage"].shape, archive["spikes"].shape)
                    self.assertEqual(len(archive["source"]), metrics["edges"])
                    for field in ("voltage", "weights", "initial_weights"):
                        self.assertTrue(np.isfinite(archive[field]).all(), field)
                    if rule == "none":
                        np.testing.assert_array_equal(archive["weights"], archive["initial_weights"])
                self.assertTrue((output / "activity.html").exists())

    def test_train_command_produces_report_and_checkpoint(self):
        output = self.folder / "train"
        result = self.invoke("train", "--task", "cube2", "--episodes", 2,
                             "--eval-episodes", 2, "--frozen", "--output", output)
        self.assertEqual(result["runs"], 1)
        self.assertEqual(result["report"], str(output / "report.html"))
        report = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(report["runs"][0]["variant"], "frozen")
        self.assertTrue((output / "agent.npz").exists())
        self.assertTrue((output / "split.json").exists())

    def test_bad_arguments_and_missing_input_report_clean_errors(self):
        for args in (("simulate", "--steps", "0"), ("simulate", "--seed", "-1"),
                     ("simulate", "--reward", "nan"), ("simulate", "--reward", "inf"),
                     ("train", "--task", "cube2", "--episodes", "0"),
                     ("query", "--graph", "synthetic", "--neurons", "missing")):
            with self.subTest(args=args):
                self.reject(*args)
        missing = self.folder / "missing.sqlite"
        self.reject("graph", "--graph", missing, "--output", self.folder / "missing.html")
        self.assertFalse(missing.exists())
        self.assertFalse((self.folder / "missing.html").exists())


if __name__ == "__main__":
    unittest.main()
