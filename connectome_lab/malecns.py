"""MaleCNS v1.0: released classified CNS neurons and measured anatomy.

Bulk Arrow input is streamed in record batches. Bundled anatomy and circuit
need NumPy only. Missing somata are disclosed, never invented for display.
"""
from pathlib import Path
from urllib.request import urlopen
from collections import deque
from dataclasses import asdict
import json
import os
import tempfile

import numpy as np

from .data import import_csv, sha256_file
from .graph import Connectome, Neuron, subgraph

BASE_URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
ANNOTATIONS = 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
TRANSMITTERS = 'body-neurotransmitters-male-cns-v1.0.feather'
CONNECTIONS = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
FILES = {
    ANNOTATIONS: '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2',
    TRANSMITTERS: '95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621',
    CONNECTIONS: 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1',
}
BUNDLED = Path(__file__).resolve().parent.parent / 'data' / 'malecns'
NT_NAMES = {'acetylcholine': 'ACH', 'gaba': 'GABA', 'glutamate': 'GLUT',
            'dopamine': 'DA', 'serotonin': 'SER', 'octopamine': 'OCT',
            'histamine': 'HIST'}


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError('Raw MaleCNS Feather import requires pyarrow; install the malecns extra or pip install pyarrow. Bundled data needs only NumPy.') from exc
    return pa, feather


def fetch_malecns(directory='data/downloads/malecns', connections=True):
    """Download pinned official v1.0 assets (about 1.1 GB with connectivity)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, digest in FILES.items():
        if filename == CONNECTIONS and not connections:
            continue
        path = directory / filename
        if path.exists() and sha256_file(path) == digest:
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(BASE_URL + filename, timeout=90) as response, temporary.open('wb') as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
            if sha256_file(temporary) != digest:
                raise ValueError(f'Pinned MaleCNS source changed: {filename}; review before updating the manifest')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def _verify(directory, connections=False):
    for filename, digest in FILES.items():
        if filename != CONNECTIONS or connections:
            if sha256_file(Path(directory) / filename) != digest:
                raise ValueError(f'MaleCNS SHA-256 verification failed: {filename}')


def _role(superclass):
    if 'sensory' in superclass:
        return 'sensory'
    if 'motor' in superclass or 'efferent' in superclass or superclass.startswith('descending_neuron'):
        return 'motor-related'
    return 'inter'


def load_malecns_nodes(directory='data/downloads/malecns', verify=True):
    """Keep every annotated nonempty superclass: the Codex 166,700-node set.

Coordinates use somaLocation, or tosomaLocation when a soma is unavailable,
converted from source 8 nm voxels to nm. Missing XYZ is zero internally only;
anatomy exports omit those IDs and disclose them in metadata.
"""
    directory = Path(directory)
    if verify:
        _verify(directory)
    _, feather = _arrow()
    annotations = feather.read_table(directory / ANNOTATIONS).to_pylist()
    neurotransmitters = feather.read_table(directory / TRANSMITTERS,
                                          columns=['body', 'consensus_nt']).to_pydict()
    nt = {int(i): value for i, value in zip(neurotransmitters['body'], neurotransmitters['consensus_nt'])}
    neurons, classes, missing, kinds, seen = [], [], [], {}, set()
    for row in annotations:
        cls = row.get('superclass')
        if not cls:
            continue
        rid = str(row.get('bodyId', ''))
        if not rid or rid in seen or not rid.isdecimal() or int(rid) < 1:
            raise ValueError('MaleCNS body IDs must be unique nonempty positive integers')
        seen.add(rid)
        position = row.get('somaLocation')
        kind = 'soma'
        if position is None:
            position, kind = row.get('tosomaLocation'), 'tosoma'
        if position is None:
            xyz, kind = (0., 0., 0.), 'missing'
            missing.append(rid)
        else:
            try:
                position = np.asarray(position, dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid MaleCNS source coordinates for {rid}') from exc
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f'Invalid MaleCNS source coordinates for {rid}')
            xyz = tuple((position * 8.).tolist())
        kinds[rid] = kind
        transmitter = nt.get(int(rid)) or 'unknown'
        transmitter = NT_NAMES.get(transmitter.lower(), transmitter)
        region = row.get('somaNeuromere') or ('VNC' if cls.startswith('vnc_') else 'brain' if cls.startswith(('cb_', 'ol_', 'visual_')) else 'brain/VNC')
        neurons.append(Neuron(rid, row.get('type') or row.get('class') or cls,
                              transmitter, _role(cls), region, *xyz))
        classes.append(cls)
    if not neurons:
        raise ValueError('MaleCNS classified neuron table is empty')
    metadata = {
        'kind': 'measured_malecns_adult', 'dataset': 'MaleCNS v1.0', 'dataset_key': 'malecns',
        'biological_topology': True, 'total_neurons': len(neurons),
        'released_node_set_complete': True,
        'source_annotation_rows': len(annotations),
        'node_selection': 'All source annotation rows with a nonempty superclass; equals the Codex MCNS v1.0 released 166700 neuron set',
        'coordinates': 'Source somaLocation or tosomaLocation attachment point, converted from 8 nm voxels to nanometres; missing positions omitted from anatomy; not full skeletons',
        'coordinate_units': 'nm', 'source_coordinate_units': '8 nm voxels',
        'coordinate_kind_counts': {kind: sum(value == kind for value in kinds.values()) for kind in ('soma', 'tosoma', 'missing')},
        'missing_coordinate_ids': missing,
        'source_url': 'https://male-cns.janelia.org/download/',
        'source_files': {name: {'url': BASE_URL + name, 'sha256': digest} for name, digest in FILES.items()},
        'license_note': 'MaleCNS dataset CC BY 4.0; https://male-cns.janelia.org/download/',
        'citations': ['Male CNS Connectome Project, MaleCNS v1.0 released 2026-06-08; https://male-cns.janelia.org/'],
        'scope': 'Adult male central nervous system: brain and ventral nerve cord; classified released neurons, excluding glia and unclassified fragments; not the entire peripheral nervous system',
        'role_mapping': 'Source superclass containing sensory -> sensory; motor/efferent/descending_neuron -> motor-related; these are interface choices, not evidence of cube action tuning',
        'neurotransmitters': 'Official consensus_nt combines prediction and available cell-type/ground-truth information; remaining unknowns are preserved',
    }
    return neurons, classes, metadata


def load_malecns_graph(directory='data/downloads/malecns', min_synapses=1, verify=True):
    """All classified neurons, all released weighted pairs between them.

Source synapses have confidence >=0.5. Unclassified segments and self-loops
are explicitly excluded. Pair weights are aggregated before min_synapses.
"""
    if isinstance(min_synapses, bool) or not isinstance(min_synapses, int) or min_synapses < 1:
        raise ValueError('min_synapses must be a positive integer')
    directory = Path(directory)
    if verify:
        _verify(directory, connections=True)
    neurons, _, metadata = load_malecns_nodes(directory, verify=False)
    pa, _ = _arrow()
    ids = np.array([int(neuron.id) for neuron in neurons], dtype=np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    n = len(ids)
    pair_chunks, weight_chunks = [], []
    source_rows = excluded = loops = 0
    with pa.memory_map(str(directory / CONNECTIONS), 'r') as mapped:
        reader = pa.ipc.open_file(mapped)
        if set(reader.schema.names) != {'body_pre', 'body_post', 'weight'}:
            raise ValueError('Unexpected MaleCNS weighted edge schema')
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            source = batch.column('body_pre').to_numpy()
            target = batch.column('body_post').to_numpy()
            weights = batch.column('weight').to_numpy()
            if (source.dtype.kind not in 'iu' or target.dtype.kind not in 'iu' or
                    weights.dtype.kind not in 'iu' or np.any(source < 1) or
                    np.any(target < 1) or np.any(weights <= 0)):
                raise ValueError('MaleCNS edge endpoints and weights must be positive integers')
            source_rows += len(source)
            si, ti = np.searchsorted(sorted_ids, source), np.searchsorted(sorted_ids, target)
            si, ti = np.minimum(si, n - 1), np.minimum(ti, n - 1)
            known = (sorted_ids[si] == source) & (sorted_ids[ti] == target)
            excluded += int(np.count_nonzero(~known))
            loops += int(np.count_nonzero(known & (source == target)))
            keep = known & (source != target)
            pair_chunks.append(order[si[keep]] * n + order[ti[keep]])
            weight_chunks.append(weights[keep])
    pairs = np.concatenate(pair_chunks) if pair_chunks else np.array([], dtype=np.int64)
    weights = np.concatenate(weight_chunks) if weight_chunks else np.array([], dtype=np.int64)
    del pair_chunks, weight_chunks
    sort = np.argsort(pairs)
    pairs, weights = pairs[sort], weights[sort]
    if len(pairs):
        starts = np.r_[0, np.flatnonzero(np.diff(pairs)) + 1]
        weights = np.add.reduceat(weights, starts)
        pairs = pairs[starts]
    keep = weights >= min_synapses
    pairs, weights = pairs[keep], weights[keep]
    metadata.update({'source_connection_rows': source_rows, 'excluded_unclassified_segment_rows': excluded,
                     'removed_self_loop_rows': loops, 'min_aggregated_synapses': min_synapses,
                     'source_synapse_confidence_threshold': 0.5, 'total_edges': len(pairs),
                     'total_contacts_retained': int(weights.sum()),
                     'connectivity_scope': 'All released confidence>=0.5 weighted pairs between the classified node set; self-loops removed; no minimum-5 pair filter unless requested'})
    return Connectome(neurons, pairs // n, pairs % n, weights, metadata)


def save_malecns_anatomy(directory, output):
    neurons, classes, metadata = load_malecns_nodes(directory)
    missing = set(metadata['missing_coordinate_ids'])
    valid = [i for i, neuron in enumerate(neurons) if neuron.id not in missing]
    metadata['rendered_anatomical_points'] = len(valid)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream,
                            positions=np.array([[neurons[i].x, neurons[i].y, neurons[i].z] for i in valid], dtype=np.float32),
                            ids=np.array([neurons[i].id for i in valid]),
                            classes=np.array([classes[i] for i in valid]),
                            metadata=json.dumps(metadata))
    return output


def select_malecns_circuit(graph, max_neurons=512):
    """Induced engineering sample, restricted to measured display positions.

    This restriction is a sampling bias: most peripheral sensory neurons have
    no soma in this volume and are therefore absent from this UI circuit.
    The full graph and its neuron count are never changed by this selection.
    """
    if isinstance(max_neurons, bool) or not isinstance(max_neurons, int) or max_neurons < 16:
        raise ValueError('Circuit requires an integer of at least 16 neurons')
    missing = set(graph.metadata.get('missing_coordinate_ids', []))
    valid = np.array([neuron.id not in missing for neuron in graph.neurons])
    if not np.any(valid):
        raise ValueError('No measured positions are available for a display circuit')
    limit = min(max_neurons, int(valid.sum()))
    sensory = [i for i, neuron in enumerate(graph.neurons) if valid[i] and neuron.role == 'sensory']
    motor = [i for i, neuron in enumerate(graph.neurons) if valid[i] and neuron.role == 'motor-related']
    def spread(values, count):
        return [values[i] for i in np.linspace(0, len(values)-1, min(len(values), count), dtype=int)] if values else []
    selected = list(dict.fromkeys(spread(sensory, 24) + spread(motor, 12)))[:limit]
    if not selected:
        selected = [int(np.flatnonzero(valid)[0])]
    # Sorted sources from the full importer need no additional 25-million-edge
    # sort. For another valid graph, stable sorting keeps deterministic BFS.
    order = None if np.all(graph.source[:-1] <= graph.source[1:]) else np.argsort(graph.source, kind='stable')
    targets = graph.target if order is None else graph.target[order]
    offsets = np.r_[0, np.cumsum(np.bincount(graph.source, minlength=graph.n_neurons))]
    seen, queue = set(selected), deque(selected)
    while queue and len(selected) < limit:
        node = queue.popleft()
        for target in targets[offsets[node]:offsets[node+1]]:
            target = int(target)
            if valid[target] and target not in seen:
                seen.add(target)
                selected.append(target)
                queue.append(target)
                if len(selected) == limit:
                    break
    if len(selected) < limit:
        selected.extend(int(i) for i in np.flatnonzero(valid) if int(i) not in seen)
        selected = selected[:limit]
    circuit = subgraph(graph, ids=[graph.neurons[i].id for i in selected])
    circuit.metadata.update({
        'kind': 'measured_malecns_subset', 'released_node_set_complete': False,
        'sample_method': '24 dispersed measured-position sensory and 12 motor-related seeds; directed BFS through measured-position nodes; induced measured edges',
        'sample_representative': False, 'sample_neurons': circuit.n_neurons,
        'sample_edges': circuit.n_edges,
        'sample_bias': 'Only neurons with source soma/tosoma coordinates are eligible; most peripheral sensory neurons lack these positions. This display circuit is not a representative full-CNS sample.',
    })
    return circuit


def save_malecns_archive(graph, path, *, overwrite=False):
    """Atomic NumPy-only complete graph archive, including exact string IDs."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, source=graph.source, target=graph.target,
                                synapse_count=graph.synapse_count,
                                neurons=json.dumps([asdict(neuron) for neuron in graph.neurons]),
                                metadata=json.dumps(graph.metadata), schema_version=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_malecns_archive(path):
    """Load a complete archive without PyArrow or access to source downloads."""
    with np.load(path, allow_pickle=False) as data:
        required = {'schema_version', 'neurons', 'metadata', 'source', 'target', 'synapse_count'}
        if not required.issubset(data.files):
            raise ValueError('Incomplete MaleCNS archive; requires node records, metadata, schema version and all edge arrays')
        version = data['schema_version']
        if version.ndim != 0 or version.dtype.kind not in 'iu' or int(version) != 1:
            raise ValueError('Unsupported MaleCNS archive version')
        try:
            neurons = [Neuron(**record) for record in json.loads(str(data['neurons']))]
            metadata = json.loads(str(data['metadata']))
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError('Invalid MaleCNS archive node records or metadata') from exc
        if not isinstance(metadata, dict):
            raise ValueError('MaleCNS archive metadata must be an object')
        if metadata.get('dataset_key') != 'malecns':
            raise ValueError('Archive is not a MaleCNS graph')
        return Connectome(neurons, data['source'], data['target'], data['synapse_count'], metadata)


def _verify_bundled_assets(folder, names):
    manifest = json.loads((folder / 'assets.sha256.json').read_text(encoding='utf-8'))
    for name in names:
        if name not in manifest or sha256_file(folder / name) != manifest[name]:
            raise ValueError(f'Bundled MaleCNS asset SHA-256 verification failed: {name}')


def load_malecns_anatomy(path=None):
    from .flywire import load_anatomy
    path = Path(path or BUNDLED / 'anatomy.npz')
    if path.parent == BUNDLED:
        _verify_bundled_assets(BUNDLED, ['anatomy.npz', 'skeletons.json'])
    payload = load_anatomy(path)
    skeletons = path.parent / 'skeletons.json'
    if skeletons.is_file():
        payload['skeletons'] = json.loads(skeletons.read_text(encoding='utf-8'))
    return payload


def load_bundled_malecns():
    folder = BUNDLED
    _verify_bundled_assets(folder, ['neurons.csv', 'synapses.csv', 'provenance.json'])
    metadata = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    graph = import_csv(folder / 'neurons.csv', folder / 'synapses.csv', metadata=metadata)
    graph.metadata['kind'] = 'measured_malecns_subset'
    return graph
