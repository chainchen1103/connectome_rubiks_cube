"""Adult female FlyWire v783: measured anatomical anchors and full raw graph.

Coordinates are community anchor locations in nm, NOT complete skeletons,
synapse endpoints or necessarily soma centers. No brain-shaped fake layout.
"""
from pathlib import Path
from dataclasses import asdict
from urllib.request import urlopen
from collections import deque
import csv
import gzip
import hashlib
import json
import os
import numpy as np
from .graph import Connectome, Neuron, subgraph
from .data import sha256_file

BASE_URL = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/'
FILES = {
    'neurons.csv.gz': '6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719',
    'coordinates.csv.gz': '14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05',
    'classification.csv.gz': 'e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663',
    'connections.csv.gz': 'd49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9',
}


def fetch_flywire(directory='data/downloads/flywire783', connections=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = dict(FILES)
    if not connections:
        files.pop('connections.csv.gz', None)
    for filename, digest in files.items():
        path = directory / filename
        if path.exists() and sha256_file(path) == digest:
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(BASE_URL + filename, timeout=60) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha256_file(temporary) != digest:
                raise ValueError(f"Pinned FlyWire source changed: {filename}; review before updating the manifest")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def _rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def _verify(directory, connections=False):
    for filename, digest in FILES.items():
        if filename == 'connections.csv.gz' and not connections:
            continue
        if sha256_file(Path(directory) / filename) != digest:
            raise ValueError(f"FlyWire SHA-256 verification failed: {filename}")


def load_flywire_nodes(directory='data/downloads/flywire783', verify=True):
    directory = Path(directory)
    if verify:
        _verify(directory)
    classes = {}
    for row in _rows(directory / 'classification.csv.gz'):
        rid = row.get('root_id', '').strip()
        if not rid:
            raise ValueError('Classification requires nonempty root_id')
        classes[rid] = row
    positions = {}
    coordinate_rows = 0
    for row in _rows(directory / 'coordinates.csv.gz'):
        coordinate_rows += 1
        rid = row.get('root_id', '').strip()
        if not rid or 'position' not in row:
            raise ValueError('Coordinate row requires root_id and position')
        if rid in positions:
            continue
        try:
            position = np.array([float(v) for v in row['position'].strip('[]').replace(',', ' ').split()])
        except ValueError as exc:
            raise ValueError('Invalid FlyWire anatomical anchor') from exc
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid FlyWire anatomical anchor")
        positions[rid] = tuple(position.tolist())
    neurons, missing, superclasses = [], [], []
    seen_ids = set()
    for row in _rows(directory / 'neurons.csv.gz'):
        rid = row.get('root_id', '').strip()
        if not rid or rid in seen_ids:
            raise ValueError('Neuron IDs must be nonempty and unique')
        seen_ids.add(rid)
        cls = classes.get(rid, {})
        flow = cls.get('flow', '')
        role = 'sensory' if flow == 'afferent' else 'motor-related' if flow == 'efferent' else 'inter'
        position = positions.get(rid)
        if position is None:
            missing.append(rid)
            position = (0., 0., 0.)
        neurons.append(Neuron(rid, cls.get('class') or 'unknown', row.get('nt_type') or 'unknown',
                              role, row.get('group') or 'unknown', *position))
        superclasses.append(cls.get('super_class') or 'unknown')
    if not neurons:
        raise ValueError('FlyWire neuron table is empty')
    metadata = {'kind': 'measured_flywire_adult', 'dataset': 'FAFB FlyWire v783',
                'biological_topology': True, 'total_neurons': len(neurons),
                'coordinates': 'First published community anchor per neuron, in nanometres; not soma or skeleton',
                'coordinate_rows': coordinate_rows, 'missing_coordinate_ids': missing,
                'source_url': 'https://codex.flywire.ai',
                'source_files': {name: {'url': BASE_URL+name, 'sha256': digest} for name, digest in FILES.items()},
                'license_note': 'FlyWire community edits/annotations CC BY-NC 4.0; see https://flywire.ai/tos',
                'citations': ['Dorkenwald et al. (2024), Nature, doi:10.1038/s41586-024-07558-y',
                              'Schlegel et al. (2024), Nature, doi:10.1038/s41586-024-07686-5'],
                'scope': 'Released adult female brain; excludes ventral nerve cord and unproofread fragments',
                'role_mapping': 'afferent -> sensory, efferent -> motor-related; efferent is not evidence of cube/fly action tuning'}
    return neurons, superclasses, metadata


def load_flywire_graph(directory='data/downloads/flywire783', min_synapses=1, verify=True):
    """All released nodes; aggregate rows per directed pair before threshold.

    The source table already applies release-specific filtering; this is not
    every chemical synapse of the biological animal. Self-loops are removed
    explicitly to make the same loopless null models applicable.
    """
    if not isinstance(min_synapses, int) or min_synapses < 1:
        raise ValueError("min_synapses must be a positive integer")
    directory = Path(directory)
    if verify:
        _verify(directory, connections=True)
    neurons, _, metadata = load_flywire_nodes(directory, verify=False)
    indices = {n.id: i for i, n in enumerate(neurons)}
    n = len(neurons)
    aggregated, rows, loops = {}, 0, 0
    for row in _rows(directory / 'connections.csv.gz'):
        rows += 1
        try:
            s, t = indices[row['pre_root_id']], indices[row['post_root_id']]
            count = int(row['syn_count'])
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid FlyWire edge row") from exc
        if count <= 0:
            raise ValueError("Nonpositive FlyWire contact count")
        if s == t:
            loops += 1
            continue
        key = s * n + t
        aggregated[key] = aggregated.get(key, 0) + count
    keys = np.fromiter(aggregated, dtype=np.int64)
    counts = np.fromiter(aggregated.values(), dtype=np.float64)
    del aggregated
    mask = counts >= min_synapses
    keys, counts = keys[mask], counts[mask]
    order = np.argsort(keys)
    keys, counts = keys[order], counts[order]
    metadata.update({'source_connection_rows': rows, 'removed_self_loop_rows': loops,
                     'min_aggregated_synapses': min_synapses, 'released_node_set_complete': True,
                     'total_edges': len(keys), 'total_contacts_retained': int(counts.sum())})
    return Connectome(neurons, keys // n, keys % n, counts, metadata)


def select_circuit(graph, max_neurons=512):
    """Deterministic induced sample around sensory inputs; preserves real XYZ."""
    if max_neurons < 16:
        raise ValueError("Circuit requires at least 16 nodes for interfaces")
    if max_neurons >= graph.n_neurons:
        return graph
    # Build CSR adjacency with NumPy, avoiding millions of Python edge objects.
    order = np.argsort(graph.source, kind='stable')
    targets = graph.target[order]
    offsets = np.r_[0, np.cumsum(np.bincount(graph.source, minlength=graph.n_neurons))]
    sensory = [i for i, neuron in enumerate(graph.neurons) if neuron.role == 'sensory']
    motor = [i for i, neuron in enumerate(graph.neurons) if neuron.role == 'motor-related']
    # Seed several dispersed real interfaces; no fabricated anatomical roles.
    def spread(values, count):
        return [values[i] for i in np.linspace(0, len(values)-1, min(len(values), count), dtype=int)] if values else []
    selected = list(dict.fromkeys(spread(sensory, 24) + spread(motor, 12)))
    if not selected:
        selected = [0]
    selected = selected[:max_neurons]
    seen, queue = set(selected), deque(selected)
    while queue and len(selected) < max_neurons:
        node = queue.popleft()
        for target in targets[offsets[node]:offsets[node+1]]:
            target = int(target)
            if target not in seen:
                seen.add(target); selected.append(target); queue.append(target)
                if len(selected) == max_neurons:
                    break
    if len(selected) < max_neurons:
        selected.extend(i for i in range(graph.n_neurons) if i not in seen)
        selected = selected[:max_neurons]
    circuit = subgraph(graph, ids=[graph.neurons[i].id for i in selected])
    circuit.metadata.update({'kind': 'measured_flywire_subset', 'released_node_set_complete': False,
                             'sample_method': '24 dispersed afferent + 12 efferent seeds, directed BFS, induced edges',
                             'sample_representative': False, 'sample_neurons': circuit.n_neurons})
    return circuit


def save_anatomy(directory, output):
    neurons, classes, metadata = load_flywire_nodes(directory)
    missing = set(metadata['missing_coordinate_ids'])
    valid = [i for i, neuron in enumerate(neurons) if neuron.id not in missing]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream, positions=np.array([[neurons[i].x, neurons[i].y, neurons[i].z] for i in valid], dtype=np.float32),
                            ids=np.array([neurons[i].id for i in valid]), classes=np.array([classes[i] for i in valid]),
                            metadata=json.dumps(metadata))
    return output


def load_anatomy(path):
    with np.load(path, allow_pickle=False) as data:
        return {'positions': data['positions'].tolist(), 'ids': data['ids'].tolist(),
                'classes': data['classes'].tolist(), 'metadata': json.loads(str(data['metadata']))}
