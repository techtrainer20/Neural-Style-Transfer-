
from flask import Flask, request, send_file
import os
from werkzeug.utils import secure_filename
from style_transfer import stylize_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_PATH = "static/output.jpg"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/stylize", methods=["POST"])
def stylize():
    image = request.files["image"]
    style = request.form["style"]
    input_path = os.path.join(UPLOAD_FOLDER, secure_filename(image.filename))
    image.save(input_path)
    stylize_image(input_path, OUTPUT_PATH, style)
    return send_file(OUTPUT_PATH, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
