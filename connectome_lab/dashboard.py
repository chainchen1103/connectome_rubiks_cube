"""Export synchronized recorded cube/fly behavior and measured-space neurons."""
from pathlib import Path
import json
import math
import numpy as np
from .environments import RubiksCubeEnv, _cube_geometry
from .embodiment import FlyArenaEnv
from .learning import NeuralAgent, AgentConfig, run_episode, graph_fingerprint
from .flywire import load_anatomy


def _neural_frame(agent):
    spikes = agent.last_spikes
    if spikes is None:
        return {'voltage': [0.] * agent.network.graph.n_neurons,
                'rates': [0.] * agent.network.graph.n_neurons, 'spikes': [],
                'probabilities': agent.last_probabilities.tolist()}
    return {'voltage': np.round(agent.network.voltage, 5).tolist(),
            'rates': np.round(spikes.mean(axis=0)*1000/agent.network.config.dt_ms, 2).tolist(),
            'spikes': np.flatnonzero(spikes.any(axis=0)).tolist(),
            'probabilities': np.round(agent.last_probabilities, 6).tolist()}


def record_track(graph, kind='cube', frames=90, seed=0, size=3, train_episodes=0, checkpoint=None):
    if frames < 2 or train_episodes < 0:
        raise ValueError('frames >= 2 and nonnegative training episodes required')
    factory = (lambda: RubiksCubeEnv(size=size, scramble_depth=2, max_steps=18)) if kind == 'cube' else FlyArenaEnv
    if kind not in {'cube', 'fly'}:
        raise ValueError('kind must be cube or fly')
    env = factory()
    agent = NeuralAgent.load(checkpoint, graph, seed) if checkpoint else NeuralAgent(
        graph, env.observation_size, env.n_actions, seed, AgentConfig(temperature=.15))
    if agent.observation_size != env.observation_size or agent.n_actions != env.n_actions:
        raise ValueError('Checkpoint environment does not match replay task')
    for i in range(train_episodes):
        run_episode(agent, factory(), seed=seed+i, learn=True)
    # Record frozen post-training behavior with separate rollout seeds. Neural
    # values precede the action whose resulting state/pose is in the frame.
    rows, episode = [], 0
    while len(rows) < frames:
        episode += 1
        agent.reset()
        observation, _ = env.reset(seed=seed+10000+episode)
        initial = {'action': None, 'reward': 0., 'episode': episode, 'step': 0, **_neural_frame(agent)}
        initial.update({'state': list(env.state_key())} if kind == 'cube' else env.pose())
        rows.append(initial)
        for step in range(1, env.max_steps+1):
            if len(rows) >= frames:
                break
            action = agent.act(observation, learn=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            row = {'action': action, 'reward': float(reward), 'episode': episode,
                   'step': step, 'terminated': bool(terminated), 'truncated': bool(truncated), **_neural_frame(agent)}
            row.update({'state': list(env.state_key())} if kind == 'cube' else env.pose())
            rows.append(row)
            if terminated or truncated:
                break
    track = {'frames': rows, 'training_episodes': train_episodes,
             'policy': 'reward trained from checkpoint' if checkpoint and train_episodes else 'frozen checkpoint' if checkpoint else 'reward trained' if train_episodes else 'untrained stochastic neural policy',
             'dt_ms': agent.network.config.dt_ms, 'integration_steps': agent.config.steps_per_action,
             'interface_notes': agent.interface_notes, 'graph_sha256': graph_fingerprint(graph)}
    if kind == 'cube':
        track.update({'size': size, 'actions': list(env.action_names),
                      'geometry': [{'position': list(p), 'normal': list(n)} for p,n in _cube_geometry(size)]})
    else:
        track.update({'actions': ['前進', '左轉', '右轉'],
                      'model_note': '工程動作介面；無肌肉、VNC、生物力學或已驗證的天然行為模型',
                      'reward_note': '食物距離改善 × 0.01 + 接近食物成功獎勵 1'})
    return track


def export_dashboard(graph, output='outputs/dashboard.html', anatomy_path=None, frames=90, seed=0,
                     size=3, train_episodes=0, cube_checkpoint=None):
    output = Path(output)
    dataset_key = graph.metadata.get('dataset_key') or ('malecns' if 'malecns' in str(graph.metadata.get('dataset','')).lower()
                    else 'flywire' if graph.metadata.get('dataset') == 'FAFB FlyWire v783' else 'custom')
    if anatomy_path is None:
        default = Path(__file__).resolve().parent.parent / ('data/malecns/anatomy.npz' if dataset_key == 'malecns' else 'data/flywire783/anatomy.npz')
        anatomy_path = default if default.exists() and dataset_key != 'custom' else None
    if anatomy_path and dataset_key == 'malecns':
        from .malecns import load_malecns_anatomy
        anatomy = load_malecns_anatomy(anatomy_path)
    else:
        anatomy = load_anatomy(anatomy_path) if anatomy_path else {
        'positions': [[n.x,n.y,n.z] for n in graph.neurons], 'ids': [n.id for n in graph.neurons],
        'classes': [n.role for n in graph.neurons], 'metadata': graph.metadata}
    anatomy_ids = set(anatomy['ids'])
    matching = sum(n.id in anatomy_ids for n in graph.neurons)
    if matching != graph.n_neurons and anatomy_path:
        raise ValueError('Simulation neuron IDs do not match the anatomical dataset; do not overlay different flies')
    anatomical_metadata = anatomy.get('metadata', {})
    coordinate_note = anatomical_metadata.get('coordinates', 'No anatomical coordinate provenance')
    payload = {'metadata': {'brain_name': anatomical_metadata.get('dataset', graph.metadata.get('kind', 'Graph')),
                            'dataset_key': dataset_key, 'graph_sha256': graph_fingerprint(graph),
                            'total_neurons': anatomical_metadata.get('total_neurons', len(anatomy['ids'])),
                            'total_edges': graph.metadata.get('total_edges', graph.n_edges),
                            'simulated_neurons': graph.n_neurons, 'simulated_edges': graph.n_edges,
                            'coordinate_note': coordinate_note,
                            'coordinate_kind': 'soma / attachment coordinates; sampled SWC neurites' if dataset_key == 'malecns' else 'community anchor; not full skeleton',
                            'source_url': anatomical_metadata.get('source_url', ''),
                            'voltage_units': 'dimensionless',
                            'rate_units': 'Hz per decision window',
                            'recording': 'Python LIF traces; independent cube and arena policies on the same structural graph',
                            'activity_timing': 'Neural window precedes the action producing this frame; reset frames have no spikes',
                            'training_episodes': train_episodes, 'source': anatomical_metadata},
               'anatomy': anatomy,
               'circuit': {'ids': [n.id for n in graph.neurons],
                           'positions': [[n.x,n.y,n.z] for n in graph.neurons],
                           'roles': [n.role for n in graph.neurons], 'nt': [n.neurotransmitter for n in graph.neurons],
                           'edges': [[int(graph.source[i]),int(graph.target[i]),float(graph.synapse_count[i])]
                                     for i in np.argsort(-graph.synapse_count)[:25000]],
                           'displayed_edges': min(graph.n_edges,25000), 'total_edges': graph.n_edges},
               'cube': record_track(graph, 'cube', frames, seed, size, train_episodes, cube_checkpoint),
               'fly': record_track(graph, 'fly', frames, seed+1, size, train_episodes)}
    payload['metadata']['track_timing'] = {track: {key: payload[track][key] for key in ('dt_ms','integration_steps')} for track in ('cube','fly')}
    payload['metadata'].update(payload['metadata']['track_timing']['cube'])
    template = Path(__file__).with_name('assets') / 'dashboard.html'
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = template.read_text(encoding='utf-8')
    if html.count('__PAYLOAD__') != 1:
        raise ValueError('Dashboard template must contain exactly one payload placeholder')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace('__PAYLOAD__', encoded), encoding='utf-8')
    # A compact manifest makes provenance review possible without parsing HTML.
    output.with_suffix('.manifest.json').write_text(json.dumps({
        'metadata': payload['metadata'], 'cube_policy': payload['cube']['policy'],
        'fly_policy': payload['fly']['policy'], 'cube_frames': len(payload['cube']['frames']),
        'fly_frames': len(payload['fly']['frames']), 'html_sha256': __import__('hashlib').sha256(output.read_bytes()).hexdigest()
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return output
