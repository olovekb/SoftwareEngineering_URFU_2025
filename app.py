import os
import uuid
import time
import json
import threading
import matplotlib
from flask import (
    Flask, request, render_template, send_from_directory,
    Response, stream_with_context, jsonify
)
from werkzeug.utils import secure_filename
from detection import Detection
from config import Config

matplotlib.use('Agg')


class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = Config.SECRET_KEY
        self.detection = Detection()
        self._mount_routes()

    def run(self):
        self.app.run(host=Config.HOST, port=Config.PORT)

    def _mount_routes(self):
        self.app.add_url_rule('/', 'index', self._index)
        self.app.add_url_rule(
            '/upload_json', 'upload_json', self._upload_json, methods=['POST']
        )
        self.app.add_url_rule('/stream/<sid>', 'stream', self._stream)
        self.app.add_url_rule('/result/<sid>', 'show_result', self._show_result)
        self.app.add_url_rule('/download/<filename>', 'download_file',
                              self._download_file)

    def _index(self):
        return render_template('index.html')

    def _upload_json(self):
        file = request.files.get('demfile')
        if not file or not self._allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        path = os.path.join(
            Config.UPLOAD_FOLDER, secure_filename(file.filename)
        )
        file.save(path)

        sid = str(uuid.uuid4())
        self.detection.progress[sid] = 0
        threading.Thread(
            target=self.detection.process_dem,
            args=(path, sid, Config.STATIC_FOLDER),
            daemon=True
        ).start()

        return jsonify({'sid': sid})

    def _stream(self, sid):
        if sid not in self.detection.progress:
            return 'Session not found', 404

        def event_stream():
            while True:
                p = self.detection.progress.get(sid, 0)
                message = {'percent': p}

                if 98 <= p < 100:
                    message['note'] = (
                        'Подождите пожалуйста, идет финальная обработка'
                    )

                yield f"data: {json.dumps(message)}\n\n"

                if p >= 100:
                    break

                time.sleep(0.3)

        return Response(
            stream_with_context(event_stream()),
            mimetype='text/event-stream'
        )

    def _show_result(self, sid):
        res = self.detection.results.get(sid)
        if not res:
            return 'Result not ready', 404

        return render_template(
            'result.html',
            image_file=res['png'],
            csv_file=res['csv'],
            anim_file=res['anim']
        )

    def _download_file(self, filename):
        return send_from_directory(
            Config.STATIC_FOLDER, filename, as_attachment=True
        )

    def _allowed_file(self, filename):
        return (
            '.' in filename and
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
        )
