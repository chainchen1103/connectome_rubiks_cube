"""Loopback-only experiment UI; one training job with atomic round checkpoints."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import copy
import json
import threading
import webbrowser
from .dashboard import export_dashboard
from .training import train_timed


class ExperimentService:
    def __init__(self, output_dir='outputs/live', dataset='malecns'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        # Round summaries survive process restarts alongside the checkpoints.
        history = []
        paths = sorted(self.output_dir.glob('sessions/*/history/*.json'), key=lambda p: p.stat().st_mtime)
        for path in paths[-50:]:
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(item,dict) and item.get('run_id'):
                    history.append({k:v for k,v in item.items() if k not in {'latest_frame','target_state'}})
            except (OSError,ValueError):
                continue
        self.status = {'state': 'idle', 'dataset_key': dataset, 'history': history}
        self.graphs = {}
        self.pages = {key: self.output_dir / f'{key}-latest.html' for key in ('malecns','flywire')
                      if (self.output_dir / f'{key}-latest.html').is_file()}

    def graph(self, dataset):
        if dataset not in {'flywire','malecns'}:
            raise ValueError('dataset must be flywire or malecns')
        with self.lock:
            if dataset not in self.graphs:
                from .cli import _load_graph
                self.graphs[dataset] = _load_graph(dataset)
            return self.graphs[dataset]

    def page(self, dataset):
        graph = self.graph(dataset)
        with self.lock:
            if dataset not in self.pages:
                self.pages[dataset] = export_dashboard(graph, self.output_dir / f'{dataset}.html', frames=45)
            return self.pages[dataset]

    def get_status(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def start(self, request):
        dataset = request.get('dataset', self.dataset)
        seconds = request.get('seconds',60)
        size, depth = request.get('size',3), request.get('depth',3)
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not __import__('math').isfinite(seconds) or seconds<=0:
            raise ValueError('seconds must be finite and positive')
        if isinstance(size,bool) or not isinstance(size,int) or size not in (2,3):
            raise ValueError('size must be 2 or 3')
        if isinstance(depth,bool) or not isinstance(depth,int) or not 1 <= depth <= 100:
            raise ValueError('depth must be an integer in 1..100')
        if request.get('resume',True) is not True:
            raise ValueError('Web sessions always preserve and resume checkpoints; use a separate CLI directory to start fresh')
        graph = self.graph(dataset)
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError('A training job is already running; stop and save it first')
            self.stop_event = threading.Event()
            history = self.status.get('history',[])
            self.status = {'state':'running', 'dataset_key':dataset, 'limit_seconds':seconds,
                           'episodes':0, 'interactions':0,'successes':0,'elapsed_seconds':0.,'history':history}
            session_dir = self.output_dir / 'sessions' / f'{dataset}_cube{size}_depth{depth}'

            def update(status):
                with self.lock:
                    self.status = {**status,'state':'saving' if status['state']=='completed' else status['state'],'history': history}

            def run():
                try:
                    result = train_timed(graph, session_dir, seconds, size, depth, resume=True,
                                         stop_event=self.stop_event, callback=update, dataset_key=dataset)
                    # Publish a recorded policy replay after weights are safely saved.
                    with self.lock:
                        self.status['state'] = 'saving'
                    checkpoint = session_dir / 'checkpoint.npz'
                    replay = export_dashboard(graph, self.output_dir / f'{dataset}-latest.html',
                                              frames=60,size=size,cube_checkpoint=checkpoint)
                    with self.lock:
                        self.pages[dataset] = replay
                        history.append({k:v for k,v in result.items() if k not in {'latest_frame','target_state'}})
                        self.status = {**result,'state':'completed','history':history[-50:], 'replay_url':f'/api/replay?dataset={dataset}'}
                except Exception as exc:
                    with self.lock:
                        self.status.update({'state':'error','error':str(exc)})
            self.worker = threading.Thread(target=run,daemon=False,name='connectome-training')
            self.worker.start()
        return self.get_status()

    def stop(self):
        self.stop_event.set()
        return self.get_status()

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join()


def make_server(service, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def respond(self, code, data, content_type='application/json'):
            raw = json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8') if content_type == 'application/json' else data
            self.send_response(code)
            self.send_header('Content-Type',content_type+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(raw)

        def local_request(self):
            allowed = {f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:
                self.respond(403,{'error':'Loopback host required'});return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://'+host for host in allowed}:
                self.respond(403,{'error':'Same-origin request required'});return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            url = urlsplit(self.path)
            try:
                if url.path == '/api/status':
                    return self.respond(200,service.get_status())
                if url.path in {'/','/api/replay','/review'}:
                    dataset = parse_qs(url.query).get('dataset',[service.dataset])[0]
                    return self.respond(200,service.page(dataset).read_bytes(),'text/html')
                if url.path == '/favicon.ico':
                    return self.respond(204,b'','image/x-icon')
                self.respond(404,{'error':'Not found'})
            except (ValueError,OSError) as exc:
                self.respond(400,{'error':str(exc)})

        def do_POST(self):
            if not self.local_request():
                return
            try:
                if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                    return self.respond(415,{'error':'application/json required'})
                length = int(self.headers.get('Content-Length','0'))
                if length < 1 or length > 8192:
                    return self.respond(413,{'error':'Invalid request size'})
                data = json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('JSON object required')
                if self.path == '/api/start':
                    self.respond(202,service.start(data))
                elif self.path == '/api/stop':
                    self.respond(200,service.stop())
                else:
                    self.respond(404,{'error':'Not found'})
            except RuntimeError as exc:
                self.respond(409,{'error':str(exc)})
            except (ValueError,OSError) as exc:
                self.respond(400,{'error':str(exc)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def serve(port=8765,dataset='malecns',output_dir='outputs/live',open_browser=False):
    service = ExperimentService(output_dir,dataset)
    service.page(dataset)
    server = make_server(service,port)
    address = f'http://127.0.0.1:{server.server_port}/?dataset={dataset}'
    print(f'Connectome training UI: {address}',flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping training and saving checkpoint...',flush=True)
    finally:
        service.close();server.server_close()
