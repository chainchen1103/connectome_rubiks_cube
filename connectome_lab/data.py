"""CSV/SQLite interchange and a pinned, attributed public larval connectome."""
from __future__ import annotations

import csv
from contextlib import closing
from dataclasses import asdict, fields
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from .graph import Connectome, Neuron, subgraph


LARVAL_URL = "https://cdn.elifesciences.org/articles/83739/elife-83739-fig1-data1-v2.zip"
LARVAL_SHA256 = "c92177b4720d86b210ea44c10b9354fdf079c8cca88f0c0bce3b57815293c9cb"
LARVAL_CITATIONS = [
    "Winding et al. (2023), The connectome of an insect brain. Science. doi:10.1126/science.add9330",
    "Pedigo et al. (2023), Generative network modeling reveals quantitative definitions of bilateral symmetry exhibited by a whole insect brain connectome. eLife 12:e83739. doi:10.7554/eLife.83739",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(row: dict, aliases: tuple[str, ...], default: str = "") -> str:
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _csv_rows(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        yield from reader


def import_csv(neurons_path: str | Path | None, edges_path: str | Path,
               *, metadata: dict | None = None) -> Connectome:
    """Read canonical or FlyWire-style CSVs without lossy integer-ID conversion.

    Edge rows sharing a directed pair are summed, including rows split by
    neuropil. Original per-neuropil counts are retained in provenance metadata.
    Region filtering uses neuron annotations; it never interprets an edge's
    neuropil as a soma region. With no neuron CSV, annotated fields are unknown.
    """
    neurons: list[Neuron] = []
    if neurons_path is not None:
        for line, row in enumerate(_csv_rows(neurons_path), 2):
            neuron_id = _value(row, ("id", "root_id", "pt_root_id", "bodyId", "neuron_id"))
            if not neuron_id:
                raise ValueError(f"Missing neuron ID at CSV line {line}.")
            try:
                neurons.append(Neuron(
                    neuron_id, _value(row, ("neuron_type", "cell_type", "type"), "unknown"),
                    _value(row, ("neurotransmitter", "nt_type", "predicted_nt"), "unknown"),
                    _value(row, ("role",), "inter"),
                    _value(row, ("region", "neuropil"), "unknown"),
                    *[float(_value(row, (axis,), "0")) for axis in ("x", "y", "z")]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid neuron at CSV line {line}: {exc}") from exc
    id_to_i = {n.id: i for i, n in enumerate(neurons)}
    if len(id_to_i) != len(neurons):
        raise ValueError("Duplicate neuron IDs in neuron CSV.")
    edges: dict[tuple[int, int], float] = {}
    neuropil_counts: dict[str, float] = {}
    rows_read = 0
    for line, row in enumerate(_csv_rows(edges_path), 2):
        source = _value(row, ("source", "pre_root_id", "pre_pt_root_id", "bodyId_pre", "source_id"))
        target = _value(row, ("target", "post_root_id", "post_pt_root_id", "bodyId_post", "target_id"))
        if not source or not target:
            raise ValueError(f"Missing directed edge endpoint at CSV line {line}.")
        try:
            count = float(_value(row, ("synapse_count", "syn_count", "weight", "count")))
        except ValueError as exc:
            raise ValueError(f"Invalid synapse count at CSV line {line}.") from exc
        if not np.isfinite(count) or count <= 0:
            raise ValueError(f"Synapse count must be finite and positive at CSV line {line}.")
        for neuron_id in (source, target):
            if neuron_id not in id_to_i:
                if neurons_path is not None:
                    raise ValueError(f"Edge references unknown neuron {neuron_id!r} at line {line}.")
                id_to_i[neuron_id] = len(neurons)
                neurons.append(Neuron(neuron_id))
        pair = (id_to_i[source], id_to_i[target])
        edges[pair] = edges.get(pair, 0.0) + count
        neuropil = _value(row, ("neuropil", "region"))
        if neuropil:
            neuropil_counts[neuropil] = neuropil_counts.get(neuropil, 0.0) + count
        rows_read += 1
    pairs = sorted(edges)
    info = dict(metadata or {})
    info.update({"format": "canonical_csv", "source_csv_rows": rows_read,
                 "duplicate_pair_rows_aggregated": rows_read - len(pairs),
                 "edges_sha256": sha256_file(edges_path),
                 "neurons_sha256": sha256_file(neurons_path) if neurons_path else None,
                 "edge_neuropil_total_counts_in_source": neuropil_counts,
                 "region_filter_semantics": "neuron_region_induced_subgraph"})
    info.setdefault("kind", "imported")
    return Connectome(neurons, np.array([p[0] for p in pairs], dtype=np.int64),
                      np.array([p[1] for p in pairs], dtype=np.int64),
                      np.array([edges[p] for p in pairs]), info)


def export_csv(graph: Connectome, folder: str | Path) -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    neuron_path, edge_path = folder / "neurons.csv", folder / "synapses.csv"
    with neuron_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Neuron)])
        writer.writeheader()
        writer.writerows(asdict(n) for n in graph.neurons)
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "synapse_count"])
        writer.writerows((graph.neurons[s].id, graph.neurons[t].id, float(c))
                        for s, t, c in zip(graph.source, graph.target, graph.synapse_count))
    (folder / "provenance.json").write_text(json.dumps(graph.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return neuron_path, edge_path


def save_sqlite(graph: Connectome, path: str | Path, *, overwrite: bool = False) -> None:
    """Atomically save a portable graph database; existing paths need overwrite=True."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing database: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            with conn:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript("""
                CREATE TABLE neurons (idx INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                    neuron_type TEXT, neurotransmitter TEXT, role TEXT, region TEXT,
                    x REAL, y REAL, z REAL);
                CREATE TABLE synapses (source INTEGER NOT NULL REFERENCES neurons(idx),
                    target INTEGER NOT NULL REFERENCES neurons(idx),
                    synapse_count REAL NOT NULL CHECK(synapse_count>0), PRIMARY KEY(source,target));
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
                conn.executemany("INSERT INTO neurons VALUES (?,?,?,?,?,?,?,?,?)",
                                 ((i, n.id, n.neuron_type, n.neurotransmitter, n.role, n.region, n.x, n.y, n.z)
                                  for i, n in enumerate(graph.neurons)))
                conn.executemany("INSERT INTO synapses VALUES (?,?,?)",
                                 ((int(s), int(t), float(c)) for s, t, c in zip(graph.source, graph.target, graph.synapse_count)))
                conn.executemany("INSERT INTO metadata VALUES (?,?)",
                                 [(str(k), json.dumps(v, ensure_ascii=False, allow_nan=False)) for k, v in graph.metadata.items()])
                # Bulk-load before building secondary indexes. Streaming tuples
                # avoids materializing tens of millions of Python edge objects.
                conn.execute("CREATE INDEX synapses_target ON synapses(target)")
                conn.execute("CREATE INDEX neurons_region ON neurons(region)")
                conn.execute("PRAGMA user_version=1")
        # Close before replace so this also works on Windows.
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_sqlite(path: str | Path, *, region: str | None = None,
                ids: list[str] | None = None) -> Connectome:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported connectome database schema version.")
        rows = conn.execute("SELECT idx,id,neuron_type,neurotransmitter,role,region,x,y,z FROM neurons ORDER BY idx").fetchall()
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError("Database neuron indices must be contiguous from zero.")
        neurons = [Neuron(*r[1:]) for r in rows]
        count = conn.execute("SELECT count(*) FROM synapses").fetchone()[0]
        edges = np.fromiter(
            conn.execute("SELECT source,target,synapse_count FROM synapses ORDER BY source,target"),
            dtype=[('source', np.int64), ('target', np.int64), ('count', np.float64)], count=count)
        metadata = {k: json.loads(v) for k, v in conn.execute("SELECT key,value FROM metadata")}
    graph = Connectome(neurons, edges['source'], edges['target'], edges['count'], metadata)
    return subgraph(graph, region=region, ids=ids) if region is not None or ids is not None else graph


def fetch_larval(cache_dir: str | Path) -> Path:
    """Download only the public pinned source ZIP, checking SHA-256 before use."""
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "elife-83739-fig1-data1-v2.zip"
    if path.exists():
        if sha256_file(path) != LARVAL_SHA256:
            raise ValueError(f"Cached source failed SHA-256 verification: {path}")
        return path
    request = Request(LARVAL_URL, headers={"User-Agent": "connectome-lab/0.1 (public research data)"})
    with urlopen(request, timeout=60) as response:
        payload = response.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("Source download exceeds the 20 MiB safety limit.")
    if hashlib.sha256(payload).hexdigest() != LARVAL_SHA256:
        raise ValueError("Source archive changed: SHA-256 mismatch; inspect the publisher before updating the pin.")
    path.write_bytes(payload)
    return path


def load_larval_archive(path: str | Path, *, max_neurons: int | None = None,
                        remove_self_loops: bool = True) -> Connectome:
    """Load measured larval connections from the publisher's verified archive.

    A bounded sample uses deterministic breadth-first traversal on the undirected
    projection, starting at the first sensory neuron by numeric skeleton ID.
    Directed edges and counts are unchanged inside that induced sample. This is
    a convenience sample, not a representative brain or a complete circuit.
    """
    if max_neurons is not None and (isinstance(max_neurons, bool) or not isinstance(max_neurons, int) or max_neurons < 1):
        raise ValueError("max_neurons must be a positive integer or None.")
    if sha256_file(path) != LARVAL_SHA256:
        raise ValueError("Larval source archive failed SHA-256 verification.")
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("elife/meta_data.csv").decode("utf-8-sig"))))
        raw_edges = [line.split() for line in archive.read("elife/G_edgelist.txt").decode("utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r[""]))
    neurons = []
    for r in rows:
        role = "sensory" if r.get("sensory") == "True" or r.get("io") == "input" else "motor" if r.get("motor") == "True" else "inter"
        neurons.append(Neuron(r[""], r.get("merge_class") or "unknown", "unknown", role,
                              {"L": "left", "R": "right", "C": "center"}.get(r.get("hemisphere"), "unknown")))
    id_to_i = {n.id: i for i, n in enumerate(neurons)}
    counts: dict[tuple[int, int], float] = {}
    dropped = 0
    for source, target, count in raw_edges:
        if source == target and remove_self_loops:
            dropped += 1
            continue
        key = (id_to_i[source], id_to_i[target])
        counts[key] = counts.get(key, 0.0) + float(count)
    pairs = sorted(counts)
    metadata = {"kind": "measured_larval", "biological_topology": True,
                "species": "Drosophila melanogaster", "life_stage": "larva",
                "source_url": LARVAL_URL, "source_sha256": LARVAL_SHA256,
                "source_archive": "eLife 83739 Figure 1 source data 1, version 2",
                "citations": LARVAL_CITATIONS,
                "license": "CC BY 4.0 (publisher article and associated source data)",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "coordinates": "not supplied; zero placeholders, not anatomy",
                "neurotransmitters": "not supplied; unknown for every neuron",
                "roles": "sensory/input and motor flags from metadata; other nodes inter",
                "region": "hemisphere annotation, not neuropil",
                "self_loop_rows_removed": dropped,
                "source_metadata_neurons": len(neurons), "source_edge_rows": len(raw_edges),
                "source_scope": "publisher archive includes brain neurons and annotated inputs/accessory neurons; not a count of brain-only neurons"}
    graph = Connectome(neurons, np.array([p[0] for p in pairs], dtype=np.int64),
                       np.array([p[1] for p in pairs], dtype=np.int64),
                       np.array([counts[p] for p in pairs]), metadata)
    if max_neurons is None or max_neurons >= graph.n_neurons:
        return graph
    adjacency: list[set[int]] = [set() for _ in neurons]
    for s, t in pairs:
        adjacency[s].add(t)
        adjacency[t].add(s)
    start = next((i for i, n in enumerate(neurons) if n.role == "sensory" and adjacency[i]), 0)
    selected, seen = [start], {start}
    cursor = 0
    while len(selected) < max_neurons and cursor < len(selected):
        node = selected[cursor]
        cursor += 1
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                selected.append(neighbor)
                if len(selected) == max_neurons:
                    break
    sample = subgraph(graph, ids=[neurons[i].id for i in selected])
    sample.metadata.update({"kind": "measured_larval_subset", "sample_method": "undirected BFS induced subgraph, neighbors in numeric skeleton-ID order",
                            "sample_seed_neuron": neurons[start].id,
                            "sample_requested_neurons": max_neurons,
                            "sample_representative": False,
                            "complete_biological_circuit": False})
    return sample


def load_bundled_larval(folder: str | Path | None = None) -> Connectome:
    folder = Path(folder) if folder else Path(__file__).resolve().parent.parent / "data" / "larval_subset"
    metadata = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in metadata.get("derived_files_sha256", {}).items():
        if sha256_file(folder / filename) != expected:
            raise ValueError(f"Bundled data failed SHA-256 verification: {filename}")
    return import_csv(folder / "neurons.csv", folder / "synapses.csv", metadata=metadata)
