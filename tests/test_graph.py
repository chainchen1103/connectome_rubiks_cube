import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from connectome_lab.data import (export_csv, import_csv, load_bundled_larval,
                                 load_sqlite, save_sqlite)
from connectome_lab.graph import (Connectome, Neuron, connectivity,
                                  degree_preserving_shuffle, random_control,
                                  subgraph, synthetic_graph)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = Connectome([Neuron("a", region="left"), Neuron("b", region="left"),
                                 Neuron("c", region="right"), Neuron("d", region="right")],
                                np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([2, 3, 5]))

    def test_direction_and_hops(self):
        self.assertEqual(connectivity(self.graph, ["b"], direction="upstream", hops=2), ["a", "b"])
        self.assertEqual(connectivity(self.graph, ["b"], hops=2), ["b", "c", "d"])
        self.assertEqual(connectivity(self.graph, ["b"], direction="both", hops=1), ["a", "b", "c"])
        self.assertEqual(connectivity(self.graph, ["b"], hops=0), ["b"])
        with self.assertRaises(ValueError):
            connectivity(self.graph, ["missing"])
        with self.assertRaises(ValueError):
            connectivity(self.graph, ["a"], hops=-1)

    def test_subgraph_remaps_indices_and_preserves_counts(self):
        selected = subgraph(self.graph, ids=["b", "c", "d"])
        np.testing.assert_array_equal(selected.source, [0, 1])
        np.testing.assert_array_equal(selected.target, [1, 2])
        np.testing.assert_array_equal(selected.synapse_count, [3, 5])
        self.assertEqual(subgraph(self.graph, region="left").n_edges, 1)
        self.assertEqual(subgraph(self.graph, region="absent").n_neurons, 0)

    def test_rejects_invalid_graphs(self):
        for source, target, weights in (([0.5], [1], [2]), ([0], [9], [1]),
                                         ([0], [1], [0]), ([0], [1], [float("nan")]),
                                         ([0], [1], [float("inf")]), ([0, 0], [1, 1], [1, 1])):
            with self.subTest(source=source, target=target, weights=weights), self.assertRaises(ValueError):
                Connectome(self.graph.neurons, np.array(source), np.array(target), np.array(weights))
        with self.assertRaises(ValueError):
            Connectome([Neuron("same"), Neuron("same")], np.array([], dtype=int), np.array([], dtype=int), [])
        with self.assertRaises(ValueError):
            Connectome([Neuron("x", x=float("nan"))], [], [], [])
        with self.assertRaises(ValueError):
            Connectome([Neuron(1)], [], [], [])

    def test_shuffle_preserves_degrees_outgoing_strength_and_counts(self):
        original = synthetic_graph(8)
        shuffled = degree_preserving_shuffle(original, seed=7, swaps=250)
        for before, after in zip(original.degrees(), shuffled.degrees()):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(np.bincount(original.source, weights=original.synapse_count, minlength=original.n_neurons),
                                      np.bincount(shuffled.source, weights=shuffled.synapse_count, minlength=original.n_neurons))
        np.testing.assert_array_equal(original.synapse_count, shuffled.synapse_count)
        self.assertFalse(np.any(shuffled.source == shuffled.target))
        self.assertEqual(len(set(zip(shuffled.source, shuffled.target))), original.n_edges)
        self.assertEqual(shuffled.metadata["completed_swaps"], 250)
        self.assertFalse(np.array_equal(original.target, shuffled.target))
        np.testing.assert_array_equal(shuffled.target, degree_preserving_shuffle(original, seed=7, swaps=250).target)
        self.assertFalse(shuffled.metadata["mixing_guaranteed"])

    def test_rigid_shuffle_reports_no_swaps(self):
        graph = Connectome([Neuron("a"), Neuron("b")], np.array([0, 1]), np.array([1, 0]), [1, 1])
        shuffled = degree_preserving_shuffle(graph, swaps=10)
        self.assertEqual(shuffled.metadata["completed_swaps"], 0)
        self.assertEqual(shuffled.metadata["requested_swaps"], 10)

    def test_random_control_matches_size_and_counts(self):
        original = synthetic_graph(2)
        random = random_control(original, seed=4)
        self.assertEqual((original.n_neurons, original.n_edges), (random.n_neurons, random.n_edges))
        np.testing.assert_array_equal(np.sort(original.synapse_count), np.sort(random.synapse_count))
        self.assertFalse(np.any(random.source == random.target))
        self.assertEqual(len(set(zip(random.source, random.target))), random.n_edges)
        self.assertEqual(random.metadata["kind"], "control")

    def test_controls_reject_self_loops_and_accept_empty_graph(self):
        loop = Connectome([Neuron("a")], np.array([0]), np.array([0]), [1])
        for control in (random_control, degree_preserving_shuffle):
            with self.assertRaises(ValueError):
                control(loop)
            empty = control(Connectome([], [], [], []))
            self.assertEqual(empty.n_edges, 0)

    def test_synthetic_is_labeled_and_has_symmetric_sensory_motor_paths(self):
        graph = synthetic_graph()
        self.assertEqual(graph.metadata["kind"], "synthetic")
        self.assertEqual(sum(n.role == "motor" for n in graph.neurons), 12)
        self.assertEqual(sum(n.role == "sensory" for n in graph.neurons), 24)
        edges = {(int(s), int(t)): c for s, t, c in zip(graph.source, graph.target, graph.synapse_count)}
        for source in range(24):
            self.assertEqual([edges[source, target] for target in range(56, 68)], [4.0] * 12)


class DataTests(unittest.TestCase):
    def test_flywire_aliases_preserve_large_root_ids_and_aggregate_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            a, b = "720575940123456789", "720575940123456790"
            (folder / "neurons.csv").write_text(f"root_id,cell_type,nt_type,neuropil\n{a},sensory,ACH,AL\n{b},motor,GABA,MB\n", encoding="utf-8")
            (folder / "edges.csv").write_text(f"pre_root_id,post_root_id,syn_count,neuropil\n{a},{b},2,AL\n{a},{b},3,MB\n", encoding="utf-8")
            graph = import_csv(folder / "neurons.csv", folder / "edges.csv")
            self.assertEqual([n.id for n in graph.neurons], [a, b])
            self.assertEqual(graph.neurons[0].region, "AL")
            np.testing.assert_array_equal(graph.source, [0])
            np.testing.assert_array_equal(graph.target, [1])
            np.testing.assert_array_equal(graph.synapse_count, [5])
            self.assertEqual(graph.metadata["edge_neuropil_total_counts_in_source"], {"AL": 2, "MB": 3})

    def test_import_rejects_missing_endpoint_and_bad_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "edges.csv"
            for row in ("a,,1", "a,b,nan", "a,b,-1", "a,b,inf"):
                path.write_text("source,target,synapse_count\n" + row + "\n", encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    import_csv(None, path)

    def test_sqlite_round_trip_and_readonly_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph.sqlite"
            graph = synthetic_graph(9)
            save_sqlite(graph, path)
            result = load_sqlite(path)
            self.assertEqual(result.neurons, graph.neurons)
            self.assertEqual(result.metadata, graph.metadata)
            np.testing.assert_array_equal(result.source, graph.source)
            np.testing.assert_array_equal(result.target, graph.target)
            np.testing.assert_array_equal(result.synapse_count, graph.synapse_count)
            with self.assertRaises(FileExistsError):
                save_sqlite(graph, path)
            save_sqlite(graph, path, overwrite=True)
            self.assertEqual(load_sqlite(path, ids=["synthetic-000", "synthetic-056"]).n_edges, 1)
            missing = Path(temp) / "missing.sqlite"
            with self.assertRaises(FileNotFoundError):
                load_sqlite(missing)
            self.assertFalse(missing.exists())

    def test_csv_roundtrip_and_unknown_node_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            original = synthetic_graph(3)
            neurons, edges = export_csv(original, temp)
            result = import_csv(neurons, edges)
            self.assertEqual(result.neurons, original.neurons)
            np.testing.assert_array_equal(result.synapse_count, original.synapse_count)
            inferred = import_csv(None, edges)
            self.assertTrue(all(n.neurotransmitter == "unknown" for n in inferred.neurons))

    def test_failed_save_keeps_existing_database_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph.sqlite"
            graph = synthetic_graph(4)
            save_sqlite(graph, path)
            original_bytes = path.read_bytes()
            graph.metadata["invalid_json"] = float("nan")
            with self.assertRaises(ValueError):
                save_sqlite(graph, path, overwrite=True)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_bundled_sample_is_genuine_but_not_complete_or_annotated_for_nt(self):
        graph = load_bundled_larval()
        self.assertEqual((graph.n_neurons, graph.n_edges), (192, 3361))
        self.assertEqual(graph.metadata["kind"], "measured_larval_subset")
        self.assertTrue(graph.metadata["biological_topology"])
        self.assertFalse(graph.metadata["complete_biological_circuit"])
        self.assertTrue(all(n.neurotransmitter == "unknown" for n in graph.neurons))
        self.assertFalse(np.any(graph.source == graph.target))
        self.assertEqual(graph.neurons[0].id, "37365")


if __name__ == "__main__":
    unittest.main()
