from flask import *
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

import run_inference

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    #submit = SubmitField("Upload File")
    submit = SubmitField("Run Inference")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file

        #return "File has been uploaded."
        return redirect(url_for('result_form'))
        #return run_inference.run_inference_fuc()
    return render_template('index.html', form=form)

@app.route('/result_form', methods=['GET',"POST"])
def result_form():

    #print(run_inference.run_inference_fuc())
    if request.method == 'POST':
        return redirect(url_for('home'))
    result = run_inference.run_inference_fuc()
    print(result.keys())
    print(result.values())
    #return jsonify(keys = list(result.keys()), values = list(result.values()))
    return render_template('result.html', imagedict = result)


if __name__ == '__main__':
    app.run(debug=True)