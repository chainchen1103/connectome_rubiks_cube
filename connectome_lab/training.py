"""One continuous cube episode, dense rewards, and exact continuation."""
from pathlib import Path
from dataclasses import asdict
import copy
import json
import math
import os
import shutil
import threading
import time
import uuid
import numpy as np
from .learning import NeuralAgent, graph_fingerprint
from .environments import RubiksCubeEnv
from .dashboard import _neural_frame
from .rewards import CubeRewardConfig, CubeRewardTracker


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def _save(path, agent, progress):
    """Atomically save policy, current cube, reward history, dynamics and RNGs."""
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    # Policy format remains compatible with NeuralAgent.load; progress has its
    # own protocol version and contains the cube/reward continuation state.
    meta = {'schema_version': 1, 'graph_sha256': graph_fingerprint(agent.network.graph),
            'config': asdict(agent.config), 'lif_config': asdict(agent.network.config),
            'observation_size': agent.observation_size, 'n_actions': agent.n_actions,
            'interface_notes': agent.interface_notes}
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, weights=agent.network.weights, initial_weights=agent.network.initial_weights,
                            projection=agent.projection, sensory=agent.sensory, motor=agent.motor,
                            metadata=json.dumps(meta), progress=json.dumps(progress, allow_nan=False),
                            rng=json.dumps(agent.rng.bit_generator.state),
                            network_rng=json.dumps(agent.network.rng.bit_generator.state),
                            voltage=agent.network.voltage, synaptic_current=agent.network.synaptic_current,
                            refractory=agent.network.refractory, spikes=agent.network.spikes,
                            neural_time_ms=agent.network.time_ms, activity=agent.activity,
                            eligibility=agent.eligibility,
                            last_spikes=agent.last_spikes if agent.last_spikes is not None else
                                np.empty((0, agent.network.graph.n_neurons), dtype=bool),
                            last_probabilities=agent.last_probabilities)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _restore(path, graph):
    agent = NeuralAgent.load(path, graph)
    with np.load(path, allow_pickle=False) as data:
        if not {'progress', 'rng', 'network_rng'}.issubset(data.files):
            raise ValueError('Checkpoint is not a resumable timed training session')
        progress = json.loads(str(data['progress']))
        if (not isinstance(progress, dict) or isinstance(progress.get('schema_version', 1), bool)
                or progress.get('schema_version', 1) not in (1, 2)):
            raise ValueError('Unsupported timed training progress version')
        agent.rng.bit_generator.state = json.loads(str(data['rng']))
        agent.network.rng.bit_generator.state = json.loads(str(data['network_rng']))
        if 'initial_weights' in data:
            initial = data['initial_weights']
            if initial.shape != agent.network.weights.shape or not np.isfinite(initial).all() or (initial < 0).any():
                raise ValueError('Invalid initial checkpoint weights')
            agent.network.initial_weights[:] = initial
        if progress.get('schema_version') == 2:
            for name, owner in [('voltage', agent.network), ('synaptic_current', agent.network),
                                ('refractory', agent.network), ('spikes', agent.network),
                                ('activity', agent), ('eligibility', agent), ('last_probabilities', agent)]:
                if name not in data:
                    raise ValueError(f'Checkpoint is missing neural state: {name}')
                value, expected = data[name], getattr(owner, name)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid checkpoint neural state: {name}')
                if name == 'refractory' and (value.dtype.kind not in 'iu' or (value < 0).any()):
                    raise ValueError('Invalid checkpoint refractory counters')
                if name == 'spikes' and not np.isin(value, [0, 1]).all():
                    raise ValueError('Invalid checkpoint spikes')
                if name in {'activity', 'last_probabilities'} and (value < 0).any():
                    raise ValueError(f'Invalid checkpoint {name}')
                if name == 'last_probabilities' and not np.isclose(value.sum(), 1):
                    raise ValueError('Invalid checkpoint action probabilities')
                setattr(owner, name, value.astype(expected.dtype, copy=True))
            if not {'neural_time_ms', 'last_spikes'}.issubset(data.files):
                raise ValueError('Missing decision-window checkpoint state')
            time_value = data['neural_time_ms']
            if time_value.ndim != 0 or not np.isfinite(time_value) or time_value < 0:
                raise ValueError('Invalid checkpoint neural time')
            agent.network.time_ms = float(time_value)
            spikes = data['last_spikes']
            if (spikes.shape not in ((0, graph.n_neurons), (agent.config.steps_per_action, graph.n_neurons))
                    or not np.isin(spikes, [0, 1]).all()):
                raise ValueError('Invalid checkpoint decision spike window')
            agent.last_spikes = spikes.astype(bool) if len(spikes) else None
    return agent, progress


def _counter(progress, key):
    value = progress.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'Invalid checkpoint counter: {key}')
    return value


def train_timed(graph, output_dir, seconds=60, size=3, depth=3, seed=0, resume=True,
                stop_event=None, callback=None, dataset_key='custom', checkpoint_interval=10.0,
                notify_interval=0.2, reward_config=None):
    """Continue one cube until solved, stopped, or its wall-time allowance ends.

    No move horizon or restart inside a run. An unsolved checkpoint resumes its
    stickers, step counter, reward history, neural state and RNGs. A solved one
    starts a new puzzle with the learned weights. Time is checked between
    decisions; final checkpoint I/O follows stopping.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and greater than zero')
    if not math.isfinite(checkpoint_interval) or checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be positive')
    if not math.isfinite(notify_interval) or notify_interval < 0:
        raise ValueError('notify_interval must be nonnegative')
    config = CubeRewardConfig() if reward_config is None else reward_config
    if not isinstance(config, CubeRewardConfig):
        raise ValueError('reward_config must be CubeRewardConfig')
    env = RubiksCubeEnv(size=size, scramble_depth=depth, max_steps=None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'checkpoint.npz'
    stop_event = stop_event or threading.Event()
    task = {'size': size, 'depth': depth, 'max_steps': None,
            'protocol': 'continuous_cube_v2', 'reward': asdict(config)}
    progress = {'schema_version': 2, 'task': task, 'total_episodes': 0,
                'total_interactions': 0, 'total_successes': 0, 'run_count': 0,
                'seed': seed, 'reason': 'new', 'dataset_key': dataset_key}
    resumed = bool(resume and checkpoint.exists())
    legacy = False
    resume_notice = ''
    if resumed:
        agent, progress = _restore(checkpoint, graph)
        saved_task = progress.get('task', {})
        if saved_task.get('size') != size or saved_task.get('depth') != depth:
            raise ValueError('Checkpoint cube task differs; select another session directory or use --fresh')
        legacy = progress.get('schema_version', 1) == 1
        if not legacy and saved_task != task:
            raise ValueError('Checkpoint reward protocol differs; select another session directory or use --fresh')
        if not legacy and not {'starting_state', 'current_state', 'puzzle_id', 'puzzle_number',
                               'episode_steps', 'reward_tracker', 'cumulative_reward'}.issubset(progress):
            raise ValueError('Incomplete continuous checkpoint progress')
        seed = int(progress['seed'])
    else:
        agent = NeuralAgent(graph, env.observation_size, env.n_actions, seed)
    for key in ('total_episodes', 'total_interactions', 'total_successes', 'run_count'):
        _counter(progress, key)
    if legacy:
        # V1 saved only the initial cube. Preserve the old file and weights;
        # never claim to recover an interrupted sticker state it did not save.
        progress['legacy_totals'] = {key: progress[key] for key in
                                    ('total_episodes', 'total_interactions', 'total_successes')}
        progress.update(schema_version=2, task=task,
                        total_episodes=progress['total_successes'], episode_steps=0, cumulative_reward=0.)
        progress['migration'] = 'v1 retries -> v2 continuous: weights retained; original scramble and reset dynamics'
        resume_notice = '舊版紀錄已備份並保留權重；因舊版未保存當下魔方，這次從原始打亂狀態開始連續回合。'
        if progress.get('reason') == 'solved':
            resume_notice = '舊版紀錄已備份並保留權重；上一顆魔方已解出，這次開始新的連續回合。'

    run_id = uuid.uuid4().hex
    progress['run_count'] += 1
    new_puzzle = 'starting_state' not in progress or progress.get('reason') == 'solved'
    if new_puzzle:
        puzzle_number = progress['total_episodes'] + 1
        observation, _ = env.reset(seed=seed + puzzle_number * 100003)
        progress.update(starting_state=list(env.state_key()), current_state=list(env.state_key()),
                        puzzle_id=uuid.uuid4().hex, puzzle_number=puzzle_number,
                        episode_steps=0, cumulative_reward=0.)
        agent.reset()
        tracker = CubeRewardTracker(size, config, env.state_key())
    elif legacy:
        observation, _ = env.reset(options={'state': progress['starting_state']})
        progress.update(current_state=list(env.state_key()), puzzle_id=uuid.uuid4().hex,
                        puzzle_number=progress['total_episodes'] + 1)
        agent.reset()
        tracker = CubeRewardTracker(size, config, env.state_key())
    else:
        required = {'current_state', 'puzzle_id', 'puzzle_number', 'episode_steps', 'reward_tracker', 'cumulative_reward'}
        if not required.issubset(progress):
            raise ValueError('Incomplete continuous checkpoint progress')
        if not isinstance(progress['puzzle_id'], str) or not progress['puzzle_id']:
            raise ValueError('Invalid checkpoint puzzle ID')
        _counter(progress, 'puzzle_number')
        observation, _ = env.reset(options={'state': progress['current_state'],
                                           'steps': _counter(progress, 'episode_steps')})
        tracker = CubeRewardTracker.from_snapshot(progress['reward_tracker'], size=size, config=config)
        recent = tracker.snapshot()['recent_states']
        if not recent or recent[-1] != list(env.state_key()):
            raise ValueError('Checkpoint reward history differs from current cube')
    if not isinstance(progress['cumulative_reward'], (int, float)) or not math.isfinite(progress['cumulative_reward']):
        raise ValueError('Invalid checkpoint cumulative reward')

    # Do not touch existing files until state and task validation has succeeded.
    if legacy:
        shutil.copy2(checkpoint, output_dir / f'checkpoint-v1-{uuid.uuid4().hex[:10]}.npz')
    elif not resume and checkpoint.exists():
        shutil.copy2(checkpoint, output_dir / f'checkpoint-before-fresh-{uuid.uuid4().hex[:10]}.npz')
    target = tuple(progress['starting_state'])
    before_weights = agent.network.weights.copy()
    graph_hash = graph_fingerprint(graph)
    episodes = interactions = successes = 0
    run_reward = 0.
    reason = 'time_limit'
    base_episodes, base_interactions, base_successes = (progress['total_episodes'], progress['total_interactions'], progress['total_successes'])
    latest_frame = {'state': list(env.state_key()), 'action': None, 'reward': 0.,
                    'episode': progress['puzzle_number'], 'puzzle_id': progress['puzzle_id'],
                    'step': env.steps, 'terminated': False, 'truncated': False, **_neural_frame(agent)}
    status = {'state': 'running', 'run_id': run_id, 'dataset_key': dataset_key,
              'graph_sha256': graph_hash, 'graph_name': graph.metadata.get('dataset', graph.metadata.get('kind', 'graph')),
              'cube_size': size, 'depth': depth, 'limit_seconds': seconds,
              'checkpoint_path': str(checkpoint), 'resumed': resumed, 'resume_notice': resume_notice,
              'total_neurons': graph.n_neurons, 'total_edges': graph.n_edges,
              'target_state': list(target), 'run_number': progress['run_count'],
              'puzzle_id': progress['puzzle_id'], 'puzzle_number': progress['puzzle_number'],
              'protocol': task['protocol'], 'reward_config': asdict(config)}
    start = time.monotonic()
    deadline = start + seconds
    last_save = last_notify = start
    training_elapsed = None

    def notify(state='running'):
        elapsed = time.monotonic() - start
        status.update({'state': state, 'reason': reason if state != 'running' else None,
                       'elapsed_seconds': elapsed if training_elapsed is None else training_elapsed,
                       'total_elapsed_seconds': elapsed,
                       'saving_seconds': 0. if training_elapsed is None else max(0., elapsed-training_elapsed),
                       'episodes': episodes, 'interactions': interactions, 'successes': successes,
                       'total_episodes': base_episodes+episodes, 'total_interactions': base_interactions+interactions,
                       'total_successes': base_successes+successes, 'episode_steps': env.steps,
                       'cumulative_reward': progress['cumulative_reward'], 'run_reward': run_reward,
                       'current_state': list(env.state_key()),
                       'weight_change_l1': float(np.abs(agent.network.weights-before_weights).sum()),
                       'latest_frame': latest_frame})
        if callback:
            callback(copy.deepcopy(status))

    def persist(reason_value):
        progress.update({'reason': reason_value, 'total_episodes': base_episodes+episodes,
                         'total_interactions': base_interactions+interactions, 'total_successes': base_successes+successes,
                         'current_state': list(env.state_key()), 'episode_steps': env.steps,
                         'reward_tracker': tracker.snapshot(), 'last_run_id': run_id,
                         'last_limit_seconds': seconds, 'last_elapsed_seconds': time.monotonic()-start,
                         'last_weight_change_l1': float(np.abs(agent.network.weights-before_weights).sum()),
                         'saved_neural_state': 'Exact decision-boundary cube, reward history, neural dynamics and RNG continuation'})
        _save(checkpoint, agent, progress)
        _atomic_json(output_dir / 'progress.json', progress)

    try:
        notify()
        while time.monotonic() < deadline and not stop_event.is_set():
            before = env.state_key()
            action = agent.act(observation, learn=True)
            observation, _, terminated, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError('Continuous cube environment unexpectedly truncated')
            terms = tracker.step(before, action, env.state_key(), terminated)
            reward = terms['reward']
            agent.reward(reward, learn=True)
            interactions += 1
            run_reward += reward
            progress['cumulative_reward'] += reward
            if terminated:
                episodes = successes = 1
                reason = 'solved'
            latest_frame = {'state': list(env.state_key()), 'action': action, 'reward': reward, 'reward_terms': terms,
                            'episode': progress['puzzle_number'], 'puzzle_id': progress['puzzle_id'],
                            'step': env.steps, 'terminated': bool(terminated), 'truncated': False, **_neural_frame(agent)}
            now = time.monotonic()
            if now-last_notify >= notify_interval:
                notify(); last_notify = now
            if terminated:
                break
            # Periodic saves do not end or reset this continuous episode.
            if now-last_save >= checkpoint_interval:
                persist('running')
                last_save = time.monotonic()
        if stop_event.is_set() and not successes:
            reason = 'stopped'
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception:
        reason = 'error'
        training_elapsed = time.monotonic()-start
        notify('saving')
        persist(reason)
        raise
    training_elapsed = time.monotonic()-start
    notify('saving')
    persist(reason)
    history_dir = output_dir / 'history'
    history_dir.mkdir(exist_ok=True)
    shutil.copy2(checkpoint, history_dir / f'{run_id}.npz')
    np.savez_compressed(history_dir / f'{run_id}-changes.npz', source=graph.source, target=graph.target,
                        before=before_weights, after=agent.network.weights, delta=agent.network.weights-before_weights,
                        neuron_ids=np.array([n.id for n in graph.neurons]))
    notify('completed')
    _atomic_json(history_dir / f'{run_id}.json', {k:v for k,v in status.items() if k != 'latest_frame'})
    return status
