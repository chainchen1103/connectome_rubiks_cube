"""Command-line workflows for importing, inspecting, simulating and learning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .data import (fetch_larval, import_csv, load_bundled_larval,
                   load_larval_archive, load_sqlite, save_sqlite)
from .graph import Connectome, connectivity, subgraph, synthetic_graph


TASKS = ("stimulus", "tmaze", "cube2", "cube3")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number")
    return number


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                               default=_json_default, indent=2) + "\n", encoding="utf-8")


def _load_graph(selection: str, seed: int = 0) -> Connectome:
    if selection == "synthetic":
        return synthetic_graph(seed)
    if selection == "larval":
        return load_bundled_larval()
    if selection == "flywire":
        folder = Path(__file__).resolve().parent.parent / 'data/flywire783/circuit'
        metadata = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
        return import_csv(folder / 'neurons.csv', folder / 'synapses.csv', metadata=metadata)
    if selection == 'malecns':
        from .malecns import load_bundled_malecns
        return load_bundled_malecns()
    if Path(selection).suffix.lower() == '.npz':
        from .malecns import load_malecns_archive
        return load_malecns_archive(selection)
    return load_sqlite(selection)


def _add_graph(parser: argparse.ArgumentParser, default: str = "synthetic") -> None:
    parser.add_argument("--graph", default=default, metavar="synthetic|larval|flywire|malecns|PATH",
                        help="bundled circuit, SQLite graph, or full MaleCNS NPZ archive")


def _simulate(graph: Connectome, output: Path, *, steps: int = 300,
              rule: str = "none", reward: float = 1.0, seed: int = 0) -> dict:
    from .dynamics import LIFNetwork, activity_metrics
    from .plasticity import LocalPlasticity, PlasticityConfig
    from .visualization import export_graph_html

    if graph.n_neurons < 1:
        raise ValueError("Cannot simulate a graph with no neurons.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    network = LIFNetwork(graph, seed=seed)
    plasticity_config = PlasticityConfig(rule=rule)
    plasticity = LocalPlasticity(network, plasticity_config) if rule != 'none' else None
    sensory = [i for i, n in enumerate(graph.neurons) if n.role == "sensory"]
    stimulated = sensory or [0]
    spikes = np.zeros((steps, graph.n_neurons), dtype=bool)
    voltage = np.zeros((steps, graph.n_neurons), dtype=float)
    current = np.zeros(graph.n_neurons, dtype=float)
    current[stimulated] = 3.5
    stimulus_steps = (steps + 1) // 2
    for step in range(steps):
        if step == stimulus_steps:
            current.fill(0)
        spikes[step] = network.step(current)
        voltage[step] = network.voltage
        if plasticity:
            plasticity.observe(spikes[step])
    # A single delayed reward is delivered after the stimulation episode.
    if plasticity:
        plasticity.reward(reward)
    metrics = {
        "metadata": graph.metadata,
        "neurons": graph.n_neurons, "edges": graph.n_edges,
        "seed": seed, "steps": steps, "dt_ms": network.config.dt_ms,
        "model": asdict(network.config), "plasticity": asdict(plasticity_config),
        "stimulus_current": 3.5, "stimulus_steps": stimulus_steps,
        "stimulated_neuron_ids": [graph.neurons[i].id for i in stimulated],
        "stimulus_mapping": "annotated sensory neurons" if sensory else "engineering fallback: neuron index zero",
        "reward": reward, "reward_delivery": "once after final step",
        "activity": activity_metrics(spikes, network.config.dt_ms),
        "weight_change_l1": float(np.abs(network.weights - network.initial_weights).sum()),
        "weights_finite": bool(np.isfinite(network.weights).all()),
        "transmitter_assumption": "GABA inhibitory; acetylcholine excitatory; unknown and other labels assigned excitatory unless explicitly overridden in Python API",
    }
    np.savez_compressed(output / "activity.npz", spikes=spikes, voltage=voltage,
                        weights=network.weights, initial_weights=network.initial_weights,
                        source=graph.source, target=graph.target,
                        neuron_ids=np.array([n.id for n in graph.neurons]),
                        dt_ms=np.array(network.config.dt_ms))
    _write_json(output / "metrics.json", metrics)
    export_graph_html(graph, output / "activity.html", spikes=spikes, dt_ms=network.config.dt_ms)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connectome-lab",
                                     description="Reproducible connectome learning experiments; research prototype.")
    commands = parser.add_subparsers(dest="command", required=True)

    server = commands.add_parser('serve', help='open local interactive timed training and checkpoint UI')
    server.add_argument('--port',type=_positive_int,default=8765)
    server.add_argument('--dataset',choices=('malecns','flywire'),default='malecns')
    server.add_argument('--output',type=Path,default=Path('outputs/live'))
    server.add_argument('--open',action='store_true',dest='open_browser')

    timed = commands.add_parser('train-timed', help='repeat cube attempts until deadline or solved, save and resume neural weights')
    _add_graph(timed,default='malecns')
    timed.add_argument('--seconds',type=_finite_float,default=60.)
    timed.add_argument('--size',type=int,choices=(2,3),default=3)
    timed.add_argument('--depth',type=_positive_int,default=3)
    timed.add_argument('--seed',type=_nonnegative_int,default=0)
    timed.add_argument('--output',type=Path,default=Path('outputs/sessions/default'))
    timed.add_argument('--fresh',action='store_true')

    dashboard = commands.add_parser('dashboard', help='export synchronized cube/fly/neural replay with anatomical brain view')
    _add_graph(dashboard, default='malecns')
    dashboard.add_argument('--output', type=Path, default=Path('outputs/dashboard.html'))
    dashboard.add_argument('--frames', type=_positive_int, default=90)
    dashboard.add_argument('--seed', type=_nonnegative_int, default=0)
    dashboard.add_argument('--size', type=int, choices=(2,3), default=3)
    dashboard.add_argument('--train-episodes', type=_nonnegative_int, default=0)
    dashboard.add_argument('--anatomy', type=Path)
    dashboard.add_argument('--cube-checkpoint', type=Path)

    adult = commands.add_parser('fetch-flywire', help='download pinned adult full brain raw tables and import all released neurons')
    adult.add_argument('--output', type=Path, default=Path('data/downloads/flywire783'))
    adult.add_argument('--database', type=Path, default=Path('outputs/full_brain/flywire783.sqlite'))
    adult.add_argument('--min-synapses', type=_positive_int, default=1)
    adult.add_argument('--overwrite', action='store_true')

    male = commands.add_parser('fetch-malecns', help='download MaleCNS v1.0 and import all 166,700 classified neurons (requires optional pyarrow)')
    male.add_argument('--output', type=Path, default=Path('data/downloads/malecns'))
    male.add_argument('--archive', type=Path, default=Path('outputs/full_brain/malecns.npz'))
    male.add_argument('--database', type=Path, help='also export SQLite; optional and slower for 25 million pairs')
    male.add_argument('--min-synapses', type=_positive_int, default=1)
    male.add_argument('--overwrite', action='store_true')

    demo = commands.add_parser("demo", help="generate graph, simulation, benchmark and cube curriculum reports")
    _add_graph(demo)
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--episodes", type=_positive_int, default=1200)
    demo.add_argument("--seeds", nargs="+", type=_nonnegative_int, default=[0, 1, 2])
    demo.add_argument("--eval-episodes", type=_positive_int, default=200)

    graph = commands.add_parser("graph", help="export an offline interactive directed graph viewer")
    _add_graph(graph, default="larval")
    graph.add_argument("--output", type=Path, default=Path("outputs/connectome.html"))
    graph.add_argument("--region", help="select an induced graph by neuron region annotation")
    graph.add_argument("--seed", type=_nonnegative_int, default=0)

    importer = commands.add_parser("import-csv", help="validate and import canonical/FlyWire CSV into SQLite")
    importer.add_argument("--neurons", type=Path, help="optional neuron annotation CSV")
    importer.add_argument("--synapses", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--overwrite", action="store_true", help="replace an existing target SQLite file")

    fetch = commands.add_parser("fetch-data", help="fetch the pinned public eLife larval data archive")
    fetch.add_argument("--output", type=Path, default=Path("data/downloads"), help="download cache directory")
    fetch.add_argument("--max-neurons", type=_positive_int, default=192)
    fetch.add_argument("--database", type=Path, default=Path("outputs/larval.sqlite"))
    fetch.add_argument("--overwrite", action="store_true", help="replace an existing target SQLite file")

    query = commands.add_parser("query", help="query directed reachability; results include seed neurons")
    _add_graph(query, default="larval")
    query.add_argument("--neurons", nargs="+", required=True, help="exact neuron IDs, preserved as strings")
    query.add_argument("--direction", choices=("upstream", "downstream", "both"), default="downstream")
    query.add_argument("--hops", type=_nonnegative_int, default=1)

    simulate = commands.add_parser("simulate", help="simulate a fixed stimulus, optionally with local plasticity")
    _add_graph(simulate)
    simulate.add_argument("--steps", type=_positive_int, default=300)
    simulate.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    simulate.add_argument("--rule", choices=("none", "hebbian", "stdp", "reward_stdp"), default="none")
    simulate.add_argument("--reward", type=_finite_float, default=1.0)
    simulate.add_argument("--seed", type=_nonnegative_int, default=0)

    train = commands.add_parser("train", help="train and evaluate one task")
    _add_graph(train)
    train.add_argument("--task", choices=TASKS, required=True)
    train.add_argument("--episodes", type=_positive_int, default=1200)
    train.add_argument("--seed", type=_nonnegative_int, default=0)
    train.add_argument("--delay", type=_positive_int, default=3)
    train.add_argument("--depth", type=_positive_int, default=1)
    train.add_argument("--output", type=Path, default=Path("outputs/train"))
    train.add_argument("--eval-episodes", type=_positive_int, default=200)
    train.add_argument("--frozen", action="store_true", help="disable weight updates for an ablation")
    train.add_argument("--no-memory", action="store_true", help="disable persistent neural state for an ablation")

    benchmark = commands.add_parser("benchmark", help="compare original, shuffled, random and ablation controls")
    _add_graph(benchmark)
    benchmark.add_argument("--episodes", type=_positive_int, default=1200)
    benchmark.add_argument("--seeds", nargs="+", type=_nonnegative_int, default=[0, 1, 2])
    benchmark.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    benchmark.add_argument("--tasks", nargs="+", choices=TASKS, default=["stimulus", "tmaze"])
    benchmark.add_argument("--eval-episodes", type=_positive_int, default=200)

    curriculum = commands.add_parser("curriculum", help="train a cube agent across increasing scramble depths")
    _add_graph(curriculum)
    curriculum.add_argument("--size", type=int, choices=(2, 3), default=2)
    curriculum.add_argument("--depths", nargs="+", type=_positive_int, default=[1, 2, 3])
    curriculum.add_argument("--episodes", type=_positive_int, default=300)
    curriculum.add_argument("--seed", type=_nonnegative_int, default=0)
    curriculum.add_argument("--output", type=Path, default=Path("outputs/curriculum"))
    curriculum.add_argument("--eval-episodes", type=_positive_int, default=100)
    return parser


def _dispatch(args: argparse.Namespace) -> dict:
    if args.command == 'fetch-malecns':
        from .malecns import fetch_malecns, load_malecns_graph, save_malecns_archive
        for destination in (args.archive, args.database):
            if destination and destination.exists() and not args.overwrite:
                raise FileExistsError(f'Refusing to replace existing graph: {destination}')
        directory = fetch_malecns(args.output)
        graph = load_malecns_graph(directory, min_synapses=args.min_synapses)
        save_malecns_archive(graph, args.archive, overwrite=args.overwrite)
        if args.database:
            save_sqlite(graph, args.database, overwrite=args.overwrite)
        return {'archive':args.archive, 'database':args.database, 'neurons':graph.n_neurons,
                'edges':graph.n_edges, 'contacts':float(graph.synapse_count.sum())}
    if args.command == 'serve':
        from .server import serve
        serve(args.port,args.dataset,args.output,args.open_browser)
        return {'state':'closed','output':args.output}
    if args.command == 'fetch-flywire':
        from .flywire import fetch_flywire, load_flywire_graph
        directory = fetch_flywire(args.output)
        graph = load_flywire_graph(directory, min_synapses=args.min_synapses)
        save_sqlite(graph, args.database, overwrite=args.overwrite)
        return {'database': args.database, 'neurons': graph.n_neurons, 'edges': graph.n_edges}
    if args.command == "import-csv":
        graph = import_csv(args.neurons, args.synapses)
        save_sqlite(graph, args.output, overwrite=args.overwrite)
        return {"database": args.output, "neurons": graph.n_neurons, "edges": graph.n_edges}
    if args.command == "fetch-data":
        archive = fetch_larval(args.output)
        graph = load_larval_archive(archive, max_neurons=args.max_neurons)
        save_sqlite(graph, args.database, overwrite=args.overwrite)
        return {"archive": archive, "database": args.database, "neurons": graph.n_neurons,
                "edges": graph.n_edges, "sha256": graph.metadata["source_sha256"]}

    # Graph topology is fixed across benchmark seeds. Other commands expose a
    # graph seed explicitly alongside their simulator/learner seed.
    graph = _load_graph(args.graph, getattr(args, "seed", 0))
    if args.command == 'train-timed':
        from .training import train_timed
        status = train_timed(graph,args.output,args.seconds,args.size,args.depth,args.seed,
                             resume=not args.fresh,dataset_key=args.graph)
        return {k:v for k,v in status.items() if k not in {'latest_frame','target_state'}}
    if args.command == 'dashboard':
        from .dashboard import export_dashboard
        output = export_dashboard(graph, args.output, args.anatomy, args.frames, args.seed,
                                  args.size, args.train_episodes, args.cube_checkpoint)
        return {'html': output, 'simulated_neurons': graph.n_neurons, 'simulated_edges': graph.n_edges}
    if args.command == "query":
        result = connectivity(graph, args.neurons, direction=args.direction, hops=args.hops)
        return {"direction": args.direction, "hops": args.hops, "count": len(result), "neurons": result}
    if args.command == "graph":
        from .visualization import export_graph_html
        if args.region is not None:
            graph = subgraph(graph, region=args.region)
        export_graph_html(graph, args.output)
        return {"html": args.output, "neurons": graph.n_neurons, "edges": graph.n_edges}
    if args.command == "simulate":
        metrics = _simulate(graph, args.output, steps=args.steps, rule=args.rule,
                            reward=args.reward, seed=args.seed)
        return {"output": args.output, "activity": metrics["activity"],
                "weight_change_l1": metrics["weight_change_l1"]}

    from .experiments import run_benchmark, run_curriculum, run_training
    if args.command == "train":
        report = run_training(graph, args.output, task=args.task, seed=args.seed,
                              episodes=args.episodes, eval_episodes=args.eval_episodes,
                              delay=args.delay, depth=args.depth, frozen=args.frozen,
                              memory=not args.no_memory)
    elif args.command == "benchmark":
        report = run_benchmark(graph, args.output, tasks=tuple(args.tasks), seeds=tuple(args.seeds),
                               episodes=args.episodes, eval_episodes=args.eval_episodes)
    elif args.command == "curriculum":
        report = run_curriculum(graph, args.output, size=args.size, depths=tuple(args.depths),
                                episodes=args.episodes, seed=args.seed, eval_episodes=args.eval_episodes)
    elif args.command == "demo":
        from .visualization import export_graph_html, export_report_html
        export_graph_html(graph, args.output / "connectome.html")
        _simulate(graph, args.output / "simulation", seed=args.seeds[0])
        benchmark_report = run_benchmark(graph, args.output / "benchmark", seeds=tuple(args.seeds),
                                         episodes=args.episodes, eval_episodes=args.eval_episodes)
        curriculum_report = run_curriculum(graph, args.output / "curriculum", size=2,
                                           depths=(1, 2, 3), episodes=min(args.episodes, 300),
                                           seed=args.seeds[0], eval_episodes=min(args.eval_episodes, 100))
        report = {"metadata": {**graph.metadata, "workflow": "demo",
                                "note": "Combined benchmark and cube curriculum. Compare within each task and evaluation protocol."},
                  "runs": benchmark_report.get("runs", []) + curriculum_report.get("runs", []),
                  "benchmark": benchmark_report, "curriculum": curriculum_report,
                  "artifacts": {"graph": "connectome.html", "simulation": "simulation/activity.html",
                                "benchmark": "benchmark/report.html", "curriculum": "curriculum/report.html"}}
        _write_json(args.output / "report.json", report)
        export_report_html(report, args.output / "report.html")
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return {"output": args.output, "report": args.output / "report.html",
            "runs": len(report.get("runs", []))}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
        # ASCII JSON keeps Windows cp950/cp1252 consoles from failing on paths
        # or metadata; files and HTML remain UTF-8.
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, default=_json_default))
    except (ValueError, OSError, sqlite3.Error, FloatingPointError) as exc:
        message = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        parser.exit(2, f"connectome-lab: error: {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
