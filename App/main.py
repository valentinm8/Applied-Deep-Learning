from flask import *
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

import run_inference

ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files/uploads'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])

    submit = SubmitField("Run Inference")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        if allowed_file(file.filename):
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file


            return redirect(url_for('result_form', filename=file.filename))

    return render_template('index.html', form=form)

@app.route('/result_form/<filename>', methods=['GET',"POST"])
def result_form(filename):

    if request.method == 'POST':
        return redirect(url_for('home'))
    result = run_inference.run_inference_fuc(filename)

    return render_template('result.html', imagedict = result)


if __name__ == '__main__':
    app.run(debug=True)