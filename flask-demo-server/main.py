import os
import re
from flask import Flask, send_from_directory, request, redirect, url_for, render_template

app = Flask(__name__, template_folder='template')

# default route
@app.route('/')
def main():
    return send_from_directory("static", "index.html")

# text receive post endpoint
@app.route("/text", methods=["POST"])
def textPost():
    if request.method == 'POST':
        _t = request.form['text']
        sanitized_text = "".join(re.findall("[a-zA-Z]", _t)) # regex only keeps alphabet characters, removes all others including spaces
        return redirect(url_for('show_result', text=sanitized_text)) # redirects to result page

# result page
@app.route("/show-result/<text>")
def show_result(text):
    print("client requested:", text)
    # inference function call here, maybe we do a process queue, or we can let the page hang until it completes?
    return render_template("result.html", data=text) # send result page

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
