import os
from app import app
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from predict import load_model, predict


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ROOT'] = 'C:/Users/dinhc/Downloads/DoAn_CS231/demo'

os.chdir(app.config['ROOT'])

result_predicts = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    if request.method == 'POST':
        try:
            model_selected = request.form['model_selected']
            type_predict_selected = request.form['type_predict_selected']
        except:
            # Default
            model_selected = "VGG16"
            type_predict_selected = "base_2_type"

        path_model = os.path.join(
            'models', "species" if type_predict_selected == "base_11_type" else "general")
        if model_selected == "VGG16":
            path_model = os.path.join(path_model, 'vgg16.h5')
        elif model_selected == "ResNet50":
            path_model = os.path.join(path_model, 'resnet50.h5')
        else:
            path_model = os.path.join(path_model, 'efficientnet_b0.h5')

        model = load_model(os.path.join(app.config['ROOT'], path_model))
        os.chdir(app.config['ROOT'])

        files = request.files.getlist('files[]')
        file_names = []
        result_predicts = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_names.append(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                class_img, class_name, prob_img = predict(model, os.path.join(app.config['UPLOAD_FOLDER'], filename),
                                                          classification=type_predict_selected)

                item_pred = {'filename': filename,
                             'class': class_img,
                             'name': class_name,
                             'prob': prob_img}

                result_predicts.append(item_pred)
            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url)

        return render_template('upload.html',
                               filenames=file_names,
                               result_predicts=result_predicts,
                               type_class=type_predict_selected,
                               model_selected=model_selected)

    return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
