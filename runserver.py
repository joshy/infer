import os
import logging
import daiquiri

LOG_DIR = 'logs'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

daiquiri.setup(level=logging.DEBUG,
    outputs=(
        daiquiri.output.File('logs/infer-errors.log', level=logging.ERROR),
        daiquiri.output.RotatingFile(
            'logs/infer-debug.log',
            level=logging.DEBUG,
            # 10 MB
            max_size_bytes=10000000)
    ))

from infer.app import app
app.run(host='0.0.0.0', port=9555)
