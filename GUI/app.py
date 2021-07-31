from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from inference import *

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, "images")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("Has a valid file, starting ")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            acc,result_path=infer_image(filename)
            result_path=os.path.join("static",result_path)
            print("app.py : upload_file() : returns a output image path",result_path)
            return render_template("index.html", filename=filename, acc=acc,result=result_path)
    return render_template("index.html")