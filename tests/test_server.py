import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from connectome_lab.graph import synthetic_graph
from connectome_lab.server import ExperimentService,make_server
from connectome_lab.training import train_timed
from connectome_lab.environments import RubiksCubeEnv


class ServerTests(unittest.TestCase):
    def test_loopback_json_contract_and_host_origin_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            service=ExperimentService(folder,'flywire')
            server=make_server(service,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                connection=HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                connection.request('GET','/api/status')
                response=connection.getresponse();self.assertEqual(response.status,200)
                self.assertEqual(json.loads(response.read())['state'],'idle')
                for headers in [{'Host':'evil.example'}, {'Origin':'https://evil.example'}]:
                    connection.request('GET','/api/status',headers=headers)
                    response=connection.getresponse();self.assertEqual(response.status,403);response.read()
                connection.request('POST','/api/start',body='{}',headers={'Content-Type':'text/plain'})
                response=connection.getresponse();self.assertEqual(response.status,415);response.read()
                connection.request('POST','/api/start',body='{"seconds":-1}',headers={'Content-Type':'application/json'})
                response=connection.getresponse();self.assertEqual(response.status,400);response.read()
                connection.close()
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_job_saves_checkpoint_and_rejects_concurrent_job(self):
        with tempfile.TemporaryDirectory() as folder:
            service=ExperimentService(folder,'flywire')
            service.graphs['flywire']=synthetic_graph()
            replay=Path(folder)/'replay.html';replay.write_text('ok')
            with patch('connectome_lab.server.export_dashboard',return_value=replay):
                service.start({'seconds':1,'dataset':'flywire','size':3,'depth':5})
                with self.assertRaises(RuntimeError):
                    service.start({'seconds':1,'dataset':'flywire','size':3,'depth':5})
                service.stop();service.close()
            status=service.get_status()
            self.assertEqual(status['state'],'completed')
            self.assertTrue(Path(status['checkpoint_path']).exists())
            self.assertEqual(len(status['history']),1)
            restarted = ExperimentService(folder, 'flywire')
            self.assertEqual(restarted.get_status()['history'][0]['run_id'], status['run_id'])
            restarted.graphs['flywire'] = synthetic_graph()
            with patch('connectome_lab.server.export_dashboard', return_value=replay):
                restarted.start({'seconds':1,'dataset':'flywire','size':3,'depth':5})
                restarted.stop(); restarted.close()
            resumed = restarted.get_status()
            self.assertEqual(resumed['state'], 'completed')
            self.assertTrue(resumed['resumed'])
            self.assertEqual(resumed['run_number'], 2)
            self.assertGreaterEqual(resumed['total_interactions'], status['total_interactions'])
            self.assertEqual(len(resumed['history']), 2)

    def test_http_continuous_puzzle_survives_server_restart(self):
        """The browser contract resumes the current cube, not just its weights."""
        entered, release = threading.Event(), threading.Event()
        traces = []

        def bounded_training(*args, **kwargs):
            run_index = len(traces)
            trace = []
            traces.append(trace)
            publish = kwargs['callback']

            def notify(status):
                publish(status)
                if (status['state'] != 'running' or not status.get('latest_frame') or
                        status['latest_frame'].get('action') is None):
                    return
                trace.append(status['latest_frame'])
                if run_index == 0 and status['interactions'] == 1:
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError('Test did not release the initial HTTP job')
                if status['interactions'] >= (5 if run_index == 0 else 3):
                    kwargs['stop_event'].set()

            kwargs.update(callback=notify, notify_interval=0)
            return train_timed(*args, **kwargs)

        def request(connection, method, path, data=None):
            body = None if data is None else json.dumps(data)
            connection.request(method, path, body=body,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())

        with tempfile.TemporaryDirectory() as folder:
            replay = Path(folder) / 'replay.html'
            replay.write_text('ok')
            results = []
            with patch('connectome_lab.server.export_dashboard', return_value=replay), \
                    patch('connectome_lab.server.train_timed', bounded_training):
                for attempt in range(2):
                    service = ExperimentService(folder, 'flywire')
                    service.graphs['flywire'] = synthetic_graph()
                    server = make_server(service, 0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    connection = HTTPConnection('127.0.0.1', server.server_port, timeout=10)
                    settings = {'seconds': 30, 'dataset': 'flywire', 'size': 3, 'depth': 5}
                    try:
                        code, _ = request(connection, 'POST', '/api/start', settings)
                        self.assertEqual(code, 202)
                        if attempt == 0:
                            self.assertTrue(entered.wait(10))
                            code, _ = request(connection, 'POST', '/api/start', settings)
                            self.assertEqual(code, 409)
                            release.set()
                        service.worker.join(timeout=10)
                        self.assertFalse(service.worker.is_alive())
                        code, result = request(connection, 'GET', '/api/status')
                        self.assertEqual(code, 200)
                        self.assertEqual(result['state'], 'completed')
                        self.assertEqual(result['reason'], 'stopped')
                        self.assertEqual(result['episodes'], 0)
                        self.assertEqual(result['total_episodes'], 0)
                        self.assertTrue(Path(result['checkpoint_path']).exists())
                        results.append(result)
                    finally:
                        release.set()
                        service.close()
                        connection.close()
                        server.shutdown()
                        server.server_close()
                        thread.join()
            first, resumed = results
            self.assertEqual(first['episode_steps'], 5)
            self.assertEqual(resumed['episode_steps'], 8)
            self.assertEqual(resumed['total_interactions'], 8)
            self.assertEqual(resumed['puzzle_id'], first['puzzle_id'])
            self.assertEqual(resumed['target_state'], first['target_state'])
            self.assertTrue(resumed['resumed'])
            self.assertEqual(resumed['run_number'], 2)
            self.assertEqual(len(resumed['history']), 2)
            cube = RubiksCubeEnv(size=3, max_steps=None)
            cube.reset(options={'state': first['current_state'], 'steps': 5})
            for frame in traces[1]:
                cube.step(frame['action'])
                self.assertEqual(frame['state'], list(cube.state_key()))
                self.assertEqual(frame['step'], cube.steps)
                self.assertIn('reward_terms', frame)
            self.assertEqual(resumed['current_state'], list(cube.state_key()))


if __name__=='__main__':
    unittest.main()
