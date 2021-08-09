from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from inference import *

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, "images")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_label_file(filename):
    return "." in filename and \
        filename.rsplit('.', 1)[1].lower() in {"mat"}
imageUpload = ""
labelUpload = ""

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("possst")
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     # flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        uploaded_files = request.files.getlist("file")

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        print(uploaded_files)

        file_image = uploaded_files[0]
        file_label = uploaded_files[1]

        if file_image.filename == '' or file_label.filename == "":
            # flash('No selected file')
            return redirect(request.url)

        if file_image and allowed_file(file_image.filename) and file_label and allowed_label_file(file_label.filename):
            print("Has a valid file, starting ")
            filename_image = secure_filename(file_image.filename)
            file_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_image))

            filename_label = secure_filename(file_label.filename)
            file_label.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_label))           
  
            pq_score, dice_score,result_path, actual_path=infer_image(filename_image, filename_label)
            result_path=os.path.join("static",result_path)
            actual_path=os.path.join("static",actual_path)
            print("app.py : upload_file() : returns a output image path",result_path)
            return render_template("index.html", filename=filename_image, acc=pq_score , dice_score=dice_score,  result=result_path, actual=actual_path)
    

        
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)