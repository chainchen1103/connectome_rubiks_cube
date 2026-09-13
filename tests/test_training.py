import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from connectome_lab.graph import synthetic_graph
from connectome_lab.training import train_timed
from connectome_lab.learning import NeuralAgent


class TimedTrainingTests(unittest.TestCase):
    def test_time_stop_checkpoint_resume_preserves_weights_target_and_progress(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as folder:
            # Stop with deterministic interaction count instead of a flaky speed assertion.
            stop = threading.Event()
            def callback(status):
                if status.get('interactions',0)>=4:
                    stop.set()
            first = train_timed(graph,folder,seconds=10,size=3,depth=4,stop_event=stop,callback=callback)
            path = Path(folder)/'checkpoint.npz'
            self.assertTrue(path.exists())
            self.assertEqual(first['reason'],'stopped')
            with np.load(path,allow_pickle=False) as file:
                saved = file['weights'].copy()
                progress = json.loads(str(file['progress']))
                for field in ['voltage','spikes','synaptic_current','eligibility','rng','projection']:
                    self.assertIn(field,file.files)
            frozen_stop = threading.Event();frozen_stop.set()
            second = train_timed(graph,folder,seconds=1,size=3,depth=4,stop_event=frozen_stop)
            np.testing.assert_array_equal(NeuralAgent.load(path,graph).network.weights,saved)
            self.assertEqual(second['target_state'],first['target_state'])
            self.assertEqual(second['total_interactions'],first['total_interactions'])
            self.assertEqual(second['run_number'],2)
            self.assertTrue(second['resumed'])
            self.assertGreaterEqual(len(list((Path(folder)/'history').glob('*-changes.npz'))),2)

    def test_deadline_stops_and_bad_config_does_not_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = synthetic_graph()
            result = train_timed(graph,folder,seconds=.001,size=3,depth=4)
            self.assertEqual(result['reason'],'time_limit')
            original = (Path(folder)/'checkpoint.npz').read_bytes()
            for kwargs in [{'seconds':0},{'seconds':float('inf')},{'size':2,'depth':4}]:
                with self.assertRaises(ValueError):
                    train_timed(graph,folder,**kwargs)
                self.assertEqual((Path(folder)/'checkpoint.npz').read_bytes(),original)

    def test_first_solution_stops_and_next_round_changes_target_but_keeps_weights(self):
        from connectome_lab.environments import RubiksCubeEnv
        original_reset = RubiksCubeEnv.reset
        def reset(env,seed=None,options=None):
            if options is not None:
                return original_reset(env,seed,options)
            original_reset(env,seed)
            env._stickers = env._solved.copy();env.apply_move('U')
            return env._observe(),{}
        with tempfile.TemporaryDirectory() as folder, patch.object(RubiksCubeEnv,'reset',reset), \
                patch.object(NeuralAgent,'act',return_value=1), \
                patch('connectome_lab.training._neural_frame',return_value={'voltage':[],'rates':[],'spikes':[],'probabilities':[]}):
            result = train_timed(synthetic_graph(),folder,seconds=5,size=3,depth=1)
            self.assertEqual(result['reason'],'solved')
            self.assertEqual(result['interactions'],1)
            self.assertEqual(result['successes'],1)
            second = train_timed(synthetic_graph(),folder,seconds=5,size=3,depth=1)
            self.assertEqual(second['total_successes'],2)


if __name__=='__main__':
    unittest.main()
