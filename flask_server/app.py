from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import traceback
import cv2
import pytesseract

from ocr import process_image, process_image2, process_image3

app = Flask(__name__)
_VERSION = 1  # API version


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/json')
def json():
    return jsonify({"message": "This is a sample route"})

@app.route('/version')
def version():
    # return jsonify({"opencv_version": cv2.__version__, "pytesseract": pytesseract.get_tesseract_version() })
    return jsonify({"opencv_version": cv2.__version__ })
    
@app.route('/v{}/ocr'.format(_VERSION), methods=["POST"])
def ocr():
    print('--call ocr processing --')
    try:
        if request.files.get("image"):
            print('--read image --')
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            print('RECV:', image.format, image.size, image.mode)

            output = process_image3(image)
            return jsonify({"output": output})
        else:
            return jsonify({"error": "only .jpg files, please"})
    except Exception as e:
        print('ocr processing exception:', e)
        print(traceback.format_exc())
        return jsonify(
            {"error": str(e)}
        )


if __name__ == '__main__':
    # cannot use host='0.0.0.0' on MacOs without a port since default port 5000 is already taken
    # app.run(debug=True, host='0.0.0.0', threaded=True)

    # use self signed adhoc ssl in order to use getUserMedia() which longer works on insecure origins
    app.run(debug=True, host='0.0.0.0', threaded=True, ssl_context="adhoc", port=8050)

    print("--end--")
