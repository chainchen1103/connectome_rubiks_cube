"""Local compressed fixtures verify anatomical provenance and full-graph import."""

import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from connectome_lab.flywire import load_flywire_graph, load_flywire_nodes, select_circuit
from connectome_lab.graph import Connectome, Neuron


class FlyWireTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.ids = ["720575940123456789", "720575940123456790", "720575940123456791"]
        a, b, c = self.ids
        self.write_gz("neurons", ["root_id", "nt_type", "group"],
                      [[a, "ACH", "AL"], [b, "GABA", "MB"], [c, "", ""]])
        self.write_gz("classification", ["root_id", "flow", "class", "super_class"],
                      [[a, "afferent", "olfactory", "sensory"], [b, "efferent", "descending", "descending"]])
        self.write_gz("coordinates", ["root_id", "position"],
                      [[a, "[1000, 2000, -3000]"], [b, "[1250.5 2200 3300]"],
                       [a, "[99000, 98000, 97000]"]])
        self.write_gz("connections", ["pre_root_id", "post_root_id", "syn_count", "neuropil"],
                      [[a, b, 2, "AL"], [a, b, 3, "MB"], [b, a, 4, "AL"],
                       [a, a, 6, "AL"], [a, a, 7, "MB"], [b, c, 1, "MB"], [c, b, 5, "MB"]])

    def write_gz(self, name, fields, rows):
        with gzip.open(self.folder / (name + ".csv.gz"), "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_nodes_preserve_ids_first_measured_anchor_and_disclose_missing_data(self):
        nodes, classes, metadata = load_flywire_nodes(self.folder, verify=False)
        self.assertEqual([node.id for node in nodes], self.ids)
        self.assertEqual([(node.x, node.y, node.z) for node in nodes],
                         [(1000., 2000., -3000.), (1250.5, 2200., 3300.), (0., 0., 0.)])
        self.assertEqual([node.role for node in nodes], ["sensory", "motor-related", "inter"])
        self.assertEqual([node.neurotransmitter for node in nodes], ["ACH", "GABA", "unknown"])
        self.assertEqual([node.region for node in nodes], ["AL", "MB", "unknown"])
        self.assertEqual(classes, ["sensory", "descending", "unknown"])
        self.assertEqual(metadata["missing_coordinate_ids"], [self.ids[2]])
        self.assertEqual(metadata["coordinate_rows"], 3)
        self.assertIn("nanometres", metadata["coordinates"])
        self.assertIn("not soma or skeleton", metadata["coordinates"])
        self.assertIn("excludes ventral nerve cord", metadata["scope"])

    def test_graph_aggregates_neuropils_before_threshold_and_counts_removed_loops(self):
        graph = load_flywire_graph(self.folder, min_synapses=5, verify=False)
        self.assertEqual([node.id for node in graph.neurons], self.ids)
        np.testing.assert_array_equal(graph.source, [0, 2])
        np.testing.assert_array_equal(graph.target, [1, 1])
        np.testing.assert_array_equal(graph.synapse_count, [5, 5])
        self.assertEqual(graph.metadata["source_connection_rows"], 7)
        self.assertEqual(graph.metadata["removed_self_loop_rows"], 2)
        self.assertEqual(graph.metadata["min_aggregated_synapses"], 5)
        self.assertEqual(graph.metadata["total_contacts_retained"], 10)
        self.assertEqual(graph.metadata["total_edges"], 2)
        self.assertTrue(graph.metadata["released_node_set_complete"])
        self.assertFalse(np.any(graph.source == graph.target))
        unfiltered = load_flywire_graph(self.folder, verify=False)
        self.assertEqual(unfiltered.n_edges, 4)
        self.assertEqual(unfiltered.metadata["total_contacts_retained"], 15)

    def test_threshold_can_remove_every_edge_without_removing_nodes(self):
        graph = load_flywire_graph(self.folder, min_synapses=100, verify=False)
        self.assertEqual(graph.n_neurons, 3)
        self.assertEqual(graph.n_edges, 0)
        self.assertEqual(graph.metadata["total_contacts_retained"], 0)

    def test_malformed_coordinates_are_rejected_instead_of_partially_parsed(self):
        for position in ("[1 2]", "[1 2 3 4]", "[1 nan 3]", "[inf 2 3]", "[1 2 3 junk]"):
            with self.subTest(position=position):
                self.write_gz("coordinates", ["root_id", "position"], [[self.ids[0], position]])
                with self.assertRaises(ValueError):
                    load_flywire_nodes(self.folder, verify=False)

    def test_duplicate_blank_and_empty_neuron_tables_are_rejected(self):
        for rows in ([[self.ids[0]], [self.ids[0]]], [[""]], []):
            with self.subTest(rows=rows):
                self.write_gz("neurons", ["root_id"], rows)
                with self.assertRaises(ValueError):
                    load_flywire_nodes(self.folder, verify=False)

    def test_invalid_edge_endpoints_and_counts_are_rejected(self):
        a, b, _ = self.ids
        for row in ([a, "missing", 2], [a, b, 0], [a, b, -1], [a, b, "1.5"], [a, b, "nan"]):
            with self.subTest(row=row):
                self.write_gz("connections", ["pre_root_id", "post_root_id", "syn_count"], [row])
                with self.assertRaises(ValueError):
                    load_flywire_graph(self.folder, verify=False)
        for threshold in (0, -1, 1.5):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                load_flywire_graph(self.folder, min_synapses=threshold, verify=False)

    def test_circuit_is_deterministic_induced_graph_with_original_anatomy(self):
        nodes = [Neuron(str(720575940123456789 + i), role="sensory" if i == 0 else "motor-related" if i == 23 else "inter",
                        x=1000. + i * 20, y=2000. + i * 7, z=-3000. + i * 9) for i in range(24)]
        source = np.r_[np.arange(23), [0, 3, 15, 22]]
        target = np.r_[np.arange(1, 24), [5, 7, 2, 0]]
        graph = Connectome(nodes, source, target, np.arange(1, len(source) + 1),
                          {"kind": "measured_flywire_adult", "released_node_set_complete": True})
        circuit = select_circuit(graph, max_neurons=16)
        self.assertEqual(circuit.n_neurons, 16)
        self.assertEqual(circuit.neurons, select_circuit(graph, max_neurons=16).neurons)
        selected = {node.id for node in circuit.neurons}
        self.assertTrue(all(node == nodes[int(node.id) - 720575940123456789] for node in circuit.neurons))
        expected = {(nodes[s].id, nodes[t].id): float(w)
                    for s, t, w in zip(graph.source, graph.target, graph.synapse_count)
                    if nodes[s].id in selected and nodes[t].id in selected}
        actual = {(circuit.neurons[s].id, circuit.neurons[t].id): float(w)
                  for s, t, w in zip(circuit.source, circuit.target, circuit.synapse_count)}
        self.assertEqual(actual, expected)
        self.assertFalse(circuit.metadata["released_node_set_complete"])
        self.assertFalse(circuit.metadata["sample_representative"])
        self.assertTrue(graph.metadata["released_node_set_complete"])
        self.assertIs(select_circuit(graph, max_neurons=24), graph)
        with self.assertRaises(ValueError):
            select_circuit(graph, max_neurons=15)


if __name__ == "__main__":
    unittest.main()
