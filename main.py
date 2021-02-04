from dotenv import load_dotenv

load_dotenv()

from bottle import Bottle, run, static_file, template, request, install, response, redirect
from helpers import get_algorithms
from train import get_model_file_name, get_vectorizer_file_name
from tika import parser
from tika.tika import checkTikaServer
from bottle_utils.flash import message_plugin
import joblib
import os

checkTikaServer()
install(message_plugin)
app = Bottle()


@app.route("/")
def main():
    message = request.get_cookie("message")

    if message != "":
        response.set_cookie("message", "")

    return template(
        "main",
        algorithms=get_algorithms(),
        message=message
    )


@app.route("/process", method="POST")
def process():
    uploaded_doc = request.files.get('document')

    if uploaded_doc is None:
        return 'Kolom dokumen harus diisi.'

    name, ext = os.path.splitext(uploaded_doc.filename)
    if ext not in ('.pdf'):
        return 'Berkas dokumen wajib bertipe PDF.'

    input_text = parser.from_buffer(
        uploaded_doc.file
    )["content"]

    algorithm_identifiers = request.forms.get("algorithm").split("-")
    algorithm_id = algorithm_identifiers[0]
    algorithm_fold = algorithm_identifiers[1]

    model = joblib.load(
        get_model_file_name(
            algorithm_id,
            algorithm_fold
        )
    )

    vectorizer = joblib.load(
        get_vectorizer_file_name(
            algorithm_fold
        )
    )

    answer = model.predict(
        vectorizer.transform(
            [input_text]
        ).toarray()
    )

    response.set_cookie(
        "message",
        "Dokumen {} termasuk klasifikasi {}".format(
            uploaded_doc.filename,
            answer[0],
        )
    )

    return redirect("/")


@app.route("/assets/<path:path>")
def assets(path):
    return static_file(path, "./assets")


if __name__ == "__main__":
    run(
        app,
        debug=True,
        host='localhost',
        port=8080,
        reloader=True
    )
