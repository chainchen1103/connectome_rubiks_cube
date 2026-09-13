"""MaleCNS source selection, unit conversion and full weighted graph semantics."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json

import numpy as np

from connectome_lab.malecns import (
    ANNOTATIONS, TRANSMITTERS, CONNECTIONS, BUNDLED,
    load_malecns_nodes, load_malecns_graph,
    load_bundled_malecns, load_malecns_anatomy,
    select_malecns_circuit, save_malecns_archive, load_malecns_archive,
)
from connectome_lab.graph import Connectome, Neuron


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'Raw Feather tests require optional pyarrow')
class MaleCNSRawTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.annotations = [
            {'bodyId': 10001, 'superclass': 'cb_sensory', 'type': 's1', 'somaLocation': [1, 2, 3], 'tosomaLocation': None},
            {'bodyId': 10002, 'superclass': 'vnc_motor', 'type': 'm1', 'somaLocation': None, 'tosomaLocation': [4, 5, 6]},
            {'bodyId': 10003, 'superclass': 'descending_neuron', 'type': None, 'somaLocation': None, 'tosomaLocation': None},
            {'bodyId': 20001, 'superclass': None, 'type': None, 'somaLocation': [7, 8, 9], 'tosomaLocation': None},
        ]
        self.write_annotations()
        self.write(TRANSMITTERS, [{'body': 10001, 'consensus_nt': 'acetylcholine'},
                                 {'body': 10002, 'consensus_nt': 'gaba'},
                                 {'body': 10003, 'consensus_nt': None}])
        self.write_edges([(10001, 10002, 2), (10001, 10002, 3), (10002, 10001, 1),
                          (10002, 10003, 4), (10001, 10001, 9),
                          (20001, 10001, 7), (10001, 99999, 6)])

    def write(self, name, rows):
        import pyarrow as pa
        import pyarrow.feather as feather
        feather.write_feather(pa.Table.from_pylist(rows), self.folder / name, chunksize=2)

    def write_annotations(self):
        self.write(ANNOTATIONS, self.annotations)

    def write_edges(self, edges):
        self.write(CONNECTIONS, [dict(zip(['body_pre', 'body_post', 'weight'], edge)) for edge in edges])

    def test_nodes_keep_classified_set_and_convert_real_voxel_positions(self):
        nodes, classes, metadata = load_malecns_nodes(self.folder, verify=False)
        self.assertEqual([node.id for node in nodes], ['10001', '10002', '10003'])
        self.assertEqual([(node.x, node.y, node.z) for node in nodes], [(8., 16., 24.), (32., 40., 48.), (0., 0., 0.)])
        self.assertEqual([node.role for node in nodes], ['sensory', 'motor-related', 'motor-related'])
        self.assertEqual([node.neurotransmitter for node in nodes], ['ACH', 'GABA', 'unknown'])
        self.assertEqual(classes, ['cb_sensory', 'vnc_motor', 'descending_neuron'])
        self.assertEqual(metadata['missing_coordinate_ids'], ['10003'])
        self.assertEqual(metadata['coordinate_kind_counts'], {'soma': 1, 'tosoma': 1, 'missing': 1})
        self.assertEqual(metadata['dataset_key'], 'malecns')
        self.assertIn('ventral nerve cord', metadata['scope'])

    def test_batches_aggregate_pairs_then_threshold_without_dropping_nodes(self):
        graph = load_malecns_graph(self.folder, min_synapses=5, verify=False)
        self.assertEqual(graph.n_neurons, 3)
        np.testing.assert_array_equal(graph.source, [0])
        np.testing.assert_array_equal(graph.target, [1])
        np.testing.assert_array_equal(graph.synapse_count, [5])
        self.assertEqual(graph.metadata['source_connection_rows'], 7)
        self.assertEqual(graph.metadata['excluded_unclassified_segment_rows'], 2)
        self.assertEqual(graph.metadata['removed_self_loop_rows'], 1)
        self.assertEqual(graph.metadata['total_contacts_retained'], 5)
        self.assertTrue(graph.metadata['released_node_set_complete'])
        unfiltered = load_malecns_graph(self.folder, verify=False)
        self.assertEqual(unfiltered.n_edges, 3)
        self.assertEqual(unfiltered.metadata['total_contacts_retained'], 10)

    def test_zero_edges_retains_all_classified_nodes(self):
        graph = load_malecns_graph(self.folder, min_synapses=999, verify=False)
        self.assertEqual((graph.n_neurons, graph.n_edges), (3, 0))

    def test_bad_positions_are_rejected(self):
        for position in ([1, 2], [1, 2, 3, 4], [1., float('nan'), 3.]):
            with self.subTest(position=position):
                self.annotations[0]['somaLocation'] = position
                self.write_annotations()
                with self.assertRaises(ValueError):
                    load_malecns_nodes(self.folder, verify=False)

    def test_duplicates_and_bad_weights_are_rejected(self):
        self.annotations[1]['bodyId'] = 10001
        self.write_annotations()
        with self.assertRaises(ValueError):
            load_malecns_nodes(self.folder, verify=False)
        self.annotations[1]['bodyId'] = 10002
        self.write_annotations()
        for weight in (0, -1, 1.5):
            self.write_edges([(10001, 10002, weight)])
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                load_malecns_graph(self.folder, verify=False)
        for threshold in (0, -1, 1.5, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                load_malecns_graph(self.folder, min_synapses=threshold, verify=False)


class MaleCNSBundledTests(unittest.TestCase):
    def test_bundled_circuit_has_real_matching_anatomy_and_honest_scope(self):
        graph = load_bundled_malecns()
        anatomy = load_malecns_anatomy()
        self.assertEqual(graph.n_neurons, 512)
        self.assertEqual(len(anatomy['ids']), 140638)
        self.assertEqual(anatomy['metadata']['total_neurons'], 166700)
        self.assertEqual(len(anatomy['metadata']['missing_coordinate_ids']), 26062)
        self.assertFalse(graph.metadata['released_node_set_complete'])
        self.assertFalse(graph.metadata['sample_representative'])
        coords = dict(zip(anatomy['ids'], anatomy['positions']))
        for neuron in graph.neurons:
            self.assertEqual([neuron.x, neuron.y, neuron.z], coords[neuron.id])
        self.assertEqual(graph.metadata['dataset_key'], 'malecns')
        self.assertTrue(all(value > 0 for value in graph.synapse_count))

    def test_complete_archive_roundtrip_and_explicit_overwrite(self):
        graph = load_bundled_malecns()
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'graph.npz'
            save_malecns_archive(graph, path)
            actual = load_malecns_archive(path)
            self.assertEqual(graph.neurons, actual.neurons)
            self.assertEqual(graph.metadata, actual.metadata)
            np.testing.assert_array_equal(graph.source, actual.source)
            np.testing.assert_array_equal(graph.target, actual.target)
            np.testing.assert_array_equal(graph.synapse_count, actual.synapse_count)
            with self.assertRaises(FileExistsError):
                save_malecns_archive(graph, path)
            save_malecns_archive(graph, path, overwrite=True)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_incomplete_archive_and_tampered_bundled_assets_are_rejected(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / 'incomplete.npz'
            np.savez(path, source=np.array([0]), target=np.array([1]), synapse_count=np.array([1]))
            with self.assertRaisesRegex(ValueError, 'Incomplete MaleCNS archive'):
                load_malecns_archive(path)
            (folder / 'assets.sha256.json').write_text(json.dumps({'neurons.csv': '0'*64}), encoding='utf-8')
            (folder / 'neurons.csv').write_text('tampered asset', encoding='utf-8')
            with patch('connectome_lab.malecns.BUNDLED', folder):
                with self.assertRaisesRegex(ValueError, 'SHA-256 verification failed'):
                    load_bundled_malecns()

    def test_display_selection_excludes_missing_positions_but_retains_parent_scope(self):
        nodes = [Neuron(str(i), role='sensory' if i < 4 else 'motor-related', x=i) for i in range(24)]
        graph = Connectome(nodes, np.arange(23), np.arange(1, 24), np.ones(23),
                           {'dataset_key': 'malecns', 'total_neurons': 24,
                            'missing_coordinate_ids': ['0', '10', '15']})
        selected = select_malecns_circuit(graph, 16)
        self.assertEqual(selected.n_neurons, 16)
        self.assertEqual(selected.metadata['total_neurons'], 24)
        self.assertEqual(selected.metadata['selection']['parent_neurons'], 24)
        self.assertFalse({'0', '10', '15'} & {neuron.id for neuron in selected.neurons})
        for source, target in zip(selected.source, selected.target):
            self.assertEqual(int(selected.neurons[target].id), int(selected.neurons[source].id) + 1)
        for limit in (15, 16.5, True):
            with self.assertRaises(ValueError):
                select_malecns_circuit(graph, limit)


if __name__ == '__main__':
    unittest.main()
