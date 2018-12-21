#!/bin/sh
nodemon -e py,html --ignore notebooks --ignore logs --exec "python" runserver.py
