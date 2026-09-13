import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from connectome_lab.graph import synthetic_graph
from connectome_lab.server import ExperimentService,make_server


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


if __name__=='__main__':
    unittest.main()
