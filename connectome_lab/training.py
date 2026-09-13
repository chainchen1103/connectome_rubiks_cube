"""Wall-time-limited repeated cube attempts with atomic resumable checkpoints."""
from pathlib import Path
from dataclasses import asdict
import copy
import json
import math
import os
import threading
import time
import uuid
import numpy as np
from .learning import NeuralAgent, graph_fingerprint
from .environments import RubiksCubeEnv
from .dashboard import _neural_frame


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def _save(path, agent, progress):
    """One atomic file contains policy, neural state, RNGs and progress.

    Compatible with NeuralAgent.load for inference. Neural state is retained
    for inspection; a new run starts a fresh attempt with the saved weights.
    """
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    meta = {'schema_version': 1, 'graph_sha256': graph_fingerprint(agent.network.graph),
            'config': asdict(agent.config), 'lif_config': asdict(agent.network.config),
            'observation_size': agent.observation_size, 'n_actions': agent.n_actions,
            'interface_notes': agent.interface_notes}
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, weights=agent.network.weights, initial_weights=agent.network.initial_weights,
                            projection=agent.projection, sensory=agent.sensory, motor=agent.motor,
                            metadata=json.dumps(meta), progress=json.dumps(progress),
                            rng=json.dumps(agent.rng.bit_generator.state),
                            network_rng=json.dumps(agent.network.rng.bit_generator.state),
                            voltage=agent.network.voltage, synaptic_current=agent.network.synaptic_current,
                            refractory=agent.network.refractory, spikes=agent.network.spikes,
                            neural_time_ms=agent.network.time_ms, activity=agent.activity,
                            eligibility=agent.eligibility)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _restore(path, graph):
    agent = NeuralAgent.load(path, graph)
    with np.load(path, allow_pickle=False) as data:
        progress = json.loads(str(data['progress']))
        agent.rng.bit_generator.state = json.loads(str(data['rng']))
        agent.network.rng.bit_generator.state = json.loads(str(data['network_rng']))
        if 'initial_weights' in data:
            agent.network.initial_weights[:] = data['initial_weights']
    return agent, progress


def train_timed(graph, output_dir, seconds=60, size=3, depth=3, seed=0, resume=True,
                stop_event=None, callback=None, dataset_key='custom', checkpoint_interval=10.0):
    """Retry one scrambled initial state until solved, stopped, or deadline.

    No episode budget. Each failed attempt has a finite move horizon, then the
    same cube is reset. Timeout-resume keeps that target. After a solved run,
    the next invocation samples a new target while retaining synaptic weights.
    Time is checked between neural decisions; checkpoint I/O follows stopping.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int,float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and greater than zero')
    if not math.isfinite(checkpoint_interval) or checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be positive')
    # Validate task before touching existing output.
    env = RubiksCubeEnv(size=size, scramble_depth=depth, max_steps=max(6, depth*4))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'checkpoint.npz'
    stop_event = stop_event or threading.Event()
    task = {'size': size, 'depth': depth, 'max_steps': env.max_steps, 'failure_penalty': 0.1}
    progress = {'task': task,
                'total_episodes': 0, 'total_interactions': 0, 'total_successes': 0,
                'run_count': 0, 'seed': seed, 'reason': 'new', 'dataset_key': dataset_key}
    if resume and checkpoint.exists():
        agent, progress = _restore(checkpoint, graph)
        if progress['task'] != task:
            raise ValueError('Checkpoint cube task differs; select another session directory or use --fresh')
        seed = int(progress['seed'])
    else:
        agent = NeuralAgent(graph, env.observation_size, env.n_actions, seed)
        if checkpoint.exists():
            # Preserve the preceding trained state when explicitly starting fresh.
            backup = output_dir / f'checkpoint-before-fresh-{uuid.uuid4().hex[:10]}.npz'
            __import__('shutil').copy2(checkpoint, backup)
    run_id = uuid.uuid4().hex
    progress['run_count'] += 1
    if 'starting_state' not in progress or progress.get('reason') == 'solved':
        env.reset(seed=seed + progress['run_count'] * 100003)
        progress['starting_state'] = list(env.state_key())
    target = tuple(progress['starting_state'])
    before_weights = agent.network.weights.copy()
    start = time.monotonic()
    deadline = start + seconds
    last_save = start
    episodes = interactions = successes = 0
    reason, latest_frame = 'time_limit', None
    graph_hash = graph_fingerprint(graph)
    base_episodes, base_interactions, base_successes = (progress['total_episodes'], progress['total_interactions'], progress['total_successes'])
    status = {'state': 'running', 'run_id': run_id, 'dataset_key': dataset_key,
              'graph_sha256': graph_hash, 'graph_name': graph.metadata.get('dataset', graph.metadata.get('kind', 'graph')),
              'cube_size': size, 'depth': depth, 'limit_seconds': seconds,
              'checkpoint_path': str(checkpoint), 'resumed': bool(resume and checkpoint.exists()),
              'total_neurons': graph.n_neurons, 'total_edges': graph.n_edges,
              'target_state': list(target), 'run_number': progress['run_count']}

    def notify(state='running'):
        status.update({'state': state, 'reason': reason if state != 'running' else None,
                       'elapsed_seconds': time.monotonic()-start, 'episodes': episodes,
                       'interactions': interactions, 'successes': successes,
                       'total_episodes': base_episodes+episodes, 'total_interactions': base_interactions+interactions,
                       'total_successes': base_successes+successes,
                       'weight_change_l1': float(np.abs(agent.network.weights-before_weights).sum()),
                       'latest_frame': latest_frame})
        if callback:
            callback(copy.deepcopy(status))

    def persist(reason_value):
        progress.update({'reason': reason_value, 'total_episodes': base_episodes+episodes,
                         'total_interactions': base_interactions+interactions, 'total_successes': base_successes+successes,
                         'last_run_id': run_id, 'last_limit_seconds': seconds,
                         'last_elapsed_seconds': time.monotonic()-start,
                         'last_weight_change_l1': float(np.abs(agent.network.weights-before_weights).sum()),
                         'saved_neural_state': 'Decision-boundary state retained for inspection; next run resets attempt dynamics'})
        _save(checkpoint, agent, progress)
        _atomic_json(output_dir / 'progress.json', progress)

    last_notify = start
    try:
        notify()
        while time.monotonic() < deadline and not stop_event.is_set():
            observation, _ = env.reset(options={'state': target})
            agent.reset()
            current_episode = base_episodes + episodes + 1
            for step in range(1, env.max_steps+1):
                if stop_event.is_set() or time.monotonic() >= deadline:
                    break
                action = agent.act(observation, learn=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                # Timed-attempt protocol supplies explicit scalar failure feedback.
                # No target action, solver or distance-to-solution is revealed.
                if truncated:
                    reward -= task['failure_penalty']
                agent.reward(reward, learn=True)
                interactions += 1
                latest_frame = {'state': list(env.state_key()), 'action': action, 'reward': float(reward),
                                'episode': current_episode, 'step': step, 'terminated': bool(terminated),
                                'truncated': bool(truncated), **_neural_frame(agent)}
                now = time.monotonic()
                if now-last_notify >= .2:
                    notify(); last_notify = now
                if terminated or truncated:
                    episodes += 1
                    successes += int(terminated)
                    break
            if successes:
                reason = 'solved'
                break
            if time.monotonic()-last_save >= checkpoint_interval:
                persist('running')
                last_save = time.monotonic()
        if stop_event.is_set() and not successes:
            reason = 'stopped'
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception:
        reason = 'error'
        notify('saving')
        persist(reason)
        raise
    notify('saving')
    persist(reason)
    # Keep immutable per-round checkpoint and exact delta for scientific audit.
    history_dir = output_dir / 'history'
    history_dir.mkdir(exist_ok=True)
    __import__('shutil').copy2(checkpoint, history_dir / f'{run_id}.npz')
    np.savez_compressed(history_dir / f'{run_id}-changes.npz',
                        source=graph.source, target=graph.target,
                        before=before_weights, after=agent.network.weights,
                        delta=agent.network.weights-before_weights,
                        neuron_ids=np.array([n.id for n in graph.neurons]))
    notify('completed')
    _atomic_json(history_dir / f'{run_id}.json', {k:v for k,v in status.items() if k != 'latest_frame'})
    return status
