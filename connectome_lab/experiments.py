"""Seeded experiments, null models, checkpointing and honest held-out audits."""
from dataclasses import asdict
from pathlib import Path
import copy
import csv
import json
import platform
import sys
import time
import numpy as np
from . import __version__
from .graph import degree_preserving_shuffle, random_control
from .environments import StimulusActionEnv, TMazeEnv, RubiksCubeEnv, build_cube_state_split
from .learning import NeuralAgent, AgentConfig, evaluate, run_episode, graph_fingerprint
from .dynamics import activity_metrics
from .visualization import export_graph_html, export_report_html


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return path


def environment_factory(task, delay=3, depth=1):
    if task == 'stimulus':
        return StimulusActionEnv
    if task == 'tmaze':
        return lambda: TMazeEnv(corridor_steps=delay)
    if task in {'cube2', 'cube3'}:
        return lambda: RubiksCubeEnv(size=int(task[-1]), scramble_depth=depth, max_steps=max(2, depth * 3))
    raise ValueError(f"Unknown task: {task}")


def _split(task, depth, seed):
    if not task.startswith('cube'):
        return None, None
    return build_cube_state_split(size=int(task[-1]), scramble_depth=depth,
                                  n_train=8 if depth == 1 else 64,
                                  n_test=4 if depth == 1 else 16, seed=seed)


def _random_baseline(factory, episodes, seed, states=None):
    rng = np.random.default_rng(seed)
    success, total_steps = 0, 0
    for i in range(episodes):
        env = factory()
        kwargs = {"seed": seed + i}
        if states is not None:
            kwargs['options'] = {'state': states[i % len(states)]}
        env.reset(**kwargs)
        while True:
            _, reward, terminated, truncated, info = env.step(int(rng.integers(env.n_actions)))
            total_steps += 1
            if terminated or truncated:
                success += bool(info.get('success', terminated and reward > 0))
                break
    return {"success_rate": success / episodes, "episodes": episodes,
            "mean_steps": total_steps / episodes, "policy": "uniform independent actions"}


def _representation(agent):
    records = []
    for cue in range(min(agent.observation_size, 2)):
        probe = copy.deepcopy(agent)
        probe.reset()
        obs = np.zeros(agent.observation_size)
        obs[cue] = 1
        probe.act(obs, learn=False)
        records.append({"input_index": cue, "neuron_activity": probe.activity.tolist(),
                        "action_probabilities": probe.last_probabilities.tolist(),
                        "stability": activity_metrics(probe.last_spikes, probe.network.config.dt_ms)})
    return records


def _metadata(graph):
    return {"schema_version": 1, "package_version": __version__, "python": sys.version,
            "numpy": np.__version__, "platform": platform.platform(),
            "graph_sha256": graph_fingerprint(graph), "graph": graph.metadata,
            "n_neurons": graph.n_neurons, "n_edges": graph.n_edges,
            "learning_rule": "motor_edge_three_factor_semi_gradient",
            "evaluation_policy": "stochastic softmax; frozen copy; fixed evaluation seeds",
            "scope": "Research prototype; no claim of biological intelligence or general cube solving"}


def run_training(graph, output_dir, task='stimulus', seed=0, episodes=1200,
                 eval_episodes=200, delay=3, depth=1, frozen=False, memory=True):
    if episodes < 1 or eval_episodes < 1:
        raise ValueError("Episode counts must be positive")
    start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = environment_factory(task, delay, depth)
    env = factory()
    config = AgentConfig(memory=memory)
    agent = NeuralAgent(graph, env.observation_size, env.n_actions, seed, config)
    train_states, test_states = _split(task, depth, seed)
    eval_seed = 100000 + seed * 1000
    before = evaluate(agent, factory, eval_episodes, eval_seed, test_states)
    representation_before = _representation(agent)
    initial_weights = agent.network.weights.copy()
    rewards, rows, curve = [], [], []
    visited = set() if train_states is not None else None
    rng = np.random.default_rng(seed + 50000)
    interactions = 0
    for episode in range(episodes):
        state = None if train_states is None else train_states[int(rng.integers(len(train_states)))]
        result = run_episode(agent, factory(), seed=seed * 100000 + episode, learn=not frozen,
                             state=state, visited_states=visited)
        interactions += result['interactions']
        rewards.append(result['reward'])
        rows.append({"episode": episode + 1, "interactions": interactions, **result})
        # Keep cumulative interactions distinct from per-episode interactions.
        rows[-1]['cumulative_interactions'] = interactions
        if (episode + 1) % max(1, episodes // 5) == 0 or episode + 1 == episodes:
            score = evaluate(agent, factory, min(100, eval_episodes), eval_seed, test_states)
            curve.append({"episode": episode + 1, "interactions": interactions,
                          "success_rate": score['success_rate']})
    after = evaluate(agent, factory, eval_episodes, eval_seed, test_states)
    change = np.abs(agent.network.weights - initial_weights)
    result = {"task": task, "seed": seed, "topology": graph.metadata.get('control', graph.metadata.get('kind', 'imported')),
              "variant": 'frozen' if frozen else 'plastic' if memory else 'memory_reset',
              "before_success": before['success_rate'], "after_success": after['success_rate'],
              "before": before, "after": after, "training_rewards": rewards,
              "evaluation_curve": curve, "interactions": interactions,
              "weight_change_l1": float(change.sum()), "changed_edges": int(np.sum(change > 1e-12)),
              "plastic_edges": int(agent.plastic_mask.sum()), "agent_config": asdict(config),
              "lif_config": asdict(agent.network.config), "interface_notes": agent.interface_notes,
              "sensory_ids": [graph.neurons[i].id for i in agent.sensory],
              "motor_ids": [graph.neurons[i].id for i in agent.motor],
              "representation_before": representation_before, "representation_after": _representation(agent),
              "stability": activity_metrics(agent.last_spikes, agent.network.config.dt_ms),
              "stability_scope": "Last training decision window, not full-run stationarity",
              "random_baseline": _random_baseline(factory, eval_episodes, eval_seed, test_states),
              "environment": {"delay": delay if task == 'tmaze' else None,
                              "scramble_depth": depth if task.startswith('cube') else None},
              "elapsed_seconds": time.perf_counter() - start}
    if train_states is not None:
        unseen = [s for s in test_states if s not in visited]
        result['split'] = {"train_initial_states": len(train_states), "test_initial_states": len(test_states),
                           "initial_overlap": len(set(train_states) & set(test_states)),
                           "test_states_visited_during_training": len(test_states) - len(unseen),
                           "strict_unseen_count": len(unseen),
                           "strict_unseen_evaluation": evaluate(agent, factory, eval_episodes, eval_seed, unseen) if unseen else None,
                           "symmetry_disjoint": False}
        write_json(output_dir / 'split.json', {"train": train_states, "test": test_states,
                                               "strict_unseen": unseen})
    if task == 'tmaze':
        result['delay_generalization'] = {str(d): evaluate(agent, environment_factory(task, d),
                                                eval_episodes, eval_seed) for d in sorted({delay, delay * 2, delay * 4})}
        ablated = copy.deepcopy(agent)
        ablated.config = AgentConfig(**{**asdict(agent.config), "memory": False})
        result['post_training_memory_ablation'] = evaluate(ablated, factory, eval_episodes, eval_seed)
    agent.save(output_dir / 'agent.npz')
    with (output_dir / 'episodes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"metadata": _metadata(graph), "runs": [result]}
    write_json(output_dir / 'results.json', report)
    export_report_html(report, output_dir / 'report.html')
    return report


def run_benchmark(graph, output_dir, tasks=('stimulus', 'tmaze'), seeds=(0, 1, 2),
                  episodes=1200, eval_episodes=200):
    if not seeds or not tasks:
        raise ValueError("At least one seed and task required")
    output_dir = Path(output_dir)
    runs = []
    for seed in seeds:
        controls = [('original', graph), ('degree_shuffled', degree_preserving_shuffle(graph, seed)),
                    ('random', random_control(graph, seed))]
        for name, control in controls:
            for task in tasks:
                variants = [(False, True), (True, True)]
                if task == 'tmaze':
                    variants.append((False, False))
                for frozen, memory in variants:
                    variant = 'frozen' if frozen else 'plastic' if memory else 'memory_reset'
                    print(f"[{task}/{name}/{variant}] seed={seed}, episodes={episodes}", flush=True)
                    result = run_training(control, output_dir / f'{task}_{name}_{variant}_seed{seed}',
                                          task, seed, episodes, eval_episodes, frozen=frozen, memory=memory)['runs'][0]
                    result['topology'] = name
                    result['control_metadata'] = control.metadata
                    runs.append(result)
    summary = []
    for task in tasks:
        for name in ['original', 'degree_shuffled', 'random']:
            for variant in ['plastic', 'frozen', 'memory_reset']:
                selected = [r for r in runs if r['task'] == task and r['topology'] == name and r['variant'] == variant]
                if selected:
                    summary.append({"task": task, "topology": name, "variant": variant, "seeds": len(selected),
                                    "before_mean": float(np.mean([r['before_success'] for r in selected])),
                                    "after_mean": float(np.mean([r['after_success'] for r in selected])),
                                    "after_std": float(np.std([r['after_success'] for r in selected])),
                                    "weight_change_l1_mean": float(np.mean([r['weight_change_l1'] for r in selected]))})
    report = {"metadata": _metadata(graph), "runs": runs, "summary": summary}
    write_json(output_dir / 'results.json', report)
    export_report_html(report, output_dir / 'report.html')
    export_graph_html(graph, output_dir / 'connectome.html')
    return report


def run_curriculum(graph, output_dir, size=2, depths=(1, 2, 3), episodes=300, seed=0, eval_episodes=100):
    if not depths or episodes < 1 or eval_episodes < 1 or list(depths) != sorted(set(depths)):
        raise ValueError("Use increasing unique depths and positive episode counts")
    output_dir = Path(output_dir)
    final_depth = max(depths)
    _, heldout = build_cube_state_split(size, final_depth, 8 if final_depth == 1 else 64,
                                       4 if final_depth == 1 else 16, seed)
    heldout_set = set(heldout)
    factory = environment_factory(f'cube{size}', depth=final_depth)
    env = factory()
    agent = NeuralAgent(graph, env.observation_size, env.n_actions, seed)
    before = evaluate(agent, factory, eval_episodes, 100000 + seed, heldout)
    initial = agent.network.weights.copy()
    visited, runs, splits = set(), [], {}
    rng = np.random.default_rng(seed)
    interactions = 0
    for depth in depths:
        candidates, _ = build_cube_state_split(size, depth, 8 if depth == 1 else 64,
                                              4 if depth == 1 else 16, seed)
        train = [s for s in candidates if s not in heldout_set]
        if not train:
            raise ValueError(f"No training states left after global holdout exclusion at depth {depth}")
        splits[str(depth)] = train
        rewards = []
        stage_factory = environment_factory(f'cube{size}', depth=depth)
        stage_before = evaluate(agent, stage_factory, eval_episodes, 200000 + seed, train)
        start_interactions = interactions
        start_weights = agent.network.weights.copy()
        for i in range(episodes):
            result = run_episode(agent, stage_factory(), seed=seed * 100000 + i,
                                 learn=True, state=train[int(rng.integers(len(train)))], visited_states=visited)
            rewards.append(result['reward'])
            interactions += result['interactions']
        score = evaluate(agent, stage_factory, eval_episodes, 200000 + seed, train)
        runs.append({"task": f'cube{size}/depth{depth}', "topology": 'original', "seed": seed,
                     "variant": 'curriculum', "before_success": stage_before['success_rate'],
                     "after_success": score['success_rate'], "evaluation_scope": "training start states; not generalization",
                     "training_rewards": rewards, "interactions": interactions-start_interactions,
                     "weight_change_l1": float(np.abs(agent.network.weights-start_weights).sum()),
                     "stability": activity_metrics(agent.last_spikes), "train_start_count": len(train)})
    unseen = [s for s in heldout if s not in visited]
    report = {"metadata": _metadata(graph), "runs": runs,
              "curriculum": {"size": size, "depths": list(depths), "episodes_per_depth": episodes,
                             "interactions": interactions, "before_final_holdout": before,
                             "after_final_holdout": evaluate(agent, factory, eval_episodes, 100000 + seed, heldout),
                             "final_holdout_count": len(heldout), "all_training_initial_overlap": 0,
                             "heldout_visited_in_training_trajectories": len(heldout)-len(unseen),
                             "strict_unseen_count": len(unseen), "symmetry_disjoint": False,
                             "strict_unseen_evaluation": evaluate(agent, factory, eval_episodes, 100000+seed, unseen) if unseen else None,
                             "random_baseline": _random_baseline(factory, eval_episodes, 100000+seed, heldout),
                             "weight_change_l1": float(np.abs(agent.network.weights-initial).sum()),
                             "agent_config": asdict(agent.config), "interface_notes": agent.interface_notes}}
    write_json(output_dir / 'splits.json', {"training_by_depth": splits, "heldout": heldout, "strict_unseen": unseen})
    write_json(output_dir / 'results.json', report)
    agent.save(output_dir / 'agent.npz')
    export_report_html(report, output_dir / 'report.html')
    return report
