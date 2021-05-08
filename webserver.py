from flask import Flask, request, jsonify, render_template, cli, Response
pipe = None # The multiprocessing pipe, sent by the main.py process starter
app = Flask(__name__, static_url_path='/')

import threading
mainlock = threading.Lock()

#cli.show_server_banner = lambda *_: None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/gen', methods=['POST'])
def gen():
    #print("/gen receives:", type(request.json), request.json)

    mainlock.acquire()
    pipe.send(request.json)
    r = pipe.recv()
    mainlock.release()
    return r
