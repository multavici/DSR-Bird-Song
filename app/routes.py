from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


app.run(host='0.0.0.0', port=50000)

@app.route("/classify", methods=['POST'])
def classify():
    #audio = request.form['audio']
    pred = "Plegadis falcinellus"
    return jsonify({'species': pred})