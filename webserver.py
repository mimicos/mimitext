from flask import Flask, request, jsonify, render_template, cli, Response
pipe = None # The multiprocessing pipe, sent by the main.py process starter
app = Flask(__name__, static_url_path='/')

#cli.show_server_banner = lambda *_: None

@app.route('/')
def index():
    global pipe
    print(pipe)
    return render_template("index.html")

@app.route('/gen', methods=['POST'])
def gen():
    print("/gen receives:", type(request.json), request.json)
    pipe.send(request.json)
    return pipe.recv()