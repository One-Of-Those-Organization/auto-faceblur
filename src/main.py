from flask import Flask, render_template, session, redirect, request, jsonify, url_for
from flask_sock import Sock
from werkzeug.utils import secure_filename
from connection import Database
import utils
import os
import re
import base64
import time
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = "INI KUNCI RAHASIA YANG TIDAK RAHASIA C4F3B4BE600DF00D"
app.config.update({
    "SESSION_COOKIE_SAMESITE": "Lax",
    "SESSION_COOKIE_SECURE": False,
    "SESSION_COOKIE_HTTPONLY": True,
    "SESSION_COOKIE_DOMAIN": None,
    "SESSION_COOKIE_PATH": "/",
})

sock = Sock(app)

db = Database("db.sqlite")
db.create_table_if_not_exist()
"""
db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", ("admin", "admin@admin.com", utils.hash_password("1234")))
"""
db.close()

# -----------------------
# -- WebSocket & CV Setup --
# -----------------------

MAX_FPS = 30
MIN_FRAME_INTERVAL = 1.0 / MAX_FPS  # ~0.033 seconds between frames


def process_frame(img):
    """Process incoming camera frames with face blur."""
    # Example CV processing (grayscale for now - replace with actual face blur logic)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(gs, cv2.COLOR_GRAY2BGR)
    return processed


def get_user_whitelist_dir(user_id: int) -> str:
    """Return absolute path for a user's whitelist folder inside static."""
    return os.path.join(app.static_folder, "whitelist", str(user_id))


def ensure_user_whitelist_dir(user_id: int) -> str:
    path = get_user_whitelist_dir(user_id)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------
# -- Frontend route --
# --------------------
# Home Route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Camera Route
@app.route('/camera', methods=['GET'])
def camera():
    logged_in = session.get("logged_in")
    if not logged_in:
        return redirect("/login")

    return render_template('camera.html')

# About Route
"""
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')
"""

# Login Route
@app.route('/login', methods=['GET'])
def login():
    logged_in = session.get("logged_in")
    if logged_in:
        return redirect("/camera")
    return render_template('login.html')

# NOTE: Will be disabled when deployed so no rando can
#       register and use it as they pleased.
# Register Route
"""
@app.route('/register')
def register():
    return render_template('register.html')
"""

# -------------------
# -- Backend Route --
# -------------------
@app.route('/be/login', methods=['POST'])
def be_login():
    if session.get("logged_in"):
        return jsonify({
            "status": 0,
            "message": "Already logged-in"
        })
    email = request.form["email"]
    password = request.form["password"]

    email = request.form["email"]
    password = request.form["password"]

    if len(email) == 0:
        return jsonify({
            "status": 0,
            "message": "Empty email is invalid"
        })

    if len(password) == 0:
        return jsonify({
            "status": 0,
            "message": "Empty email is invalid"
        })

    if re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        db = Database("db.sqlite")
        out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
        if len(out) == 0:
            return jsonify({
                "status": 0,
                "message": "User is not registered"
            })

        row = out[0]
        db.close()
        if utils.verify_password(row["password"], password):
            session["logged_in"] = True
            session["user"] = row["name"]
            session["id"] = row["id"]
            return jsonify({
                "status": 1,
                "message": {
                    "user": row["name"],
                    "email": row["email"]
                }
            })
        else:
            return jsonify({
                "status": 0,
                "message": "Wrong password for that email"
            })

    else:
        return jsonify({
            "status": 0,
            "message": "Wrong email format"
        })


# WebSocket endpoint for camera stream
@sock.route('/ws/camera')
def ws_camera(ws):
    """WebSocket endpoint for processing camera frames."""
    last_frame_time = 0

    while True:
        try:
            # Receive base64 encoded frame from client
            data = ws.receive()
            if data is None:
                break

            # Rate limiting - enforce max 30 FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < MIN_FRAME_INTERVAL:
                # Skip this frame to maintain max FPS
                continue

            last_frame_time = current_time

            # Decode base64 image
            try:
                # Remove data URL prefix if present
                if ',' in data:
                    data = data.split(',')[1]

                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Process the frame (apply face blur)
                processed = process_frame(frame)

                # Encode processed frame back to base64 JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success, buffer = cv2.imencode(".jpg", processed, encode_param)

                if not success:
                    print("Failed to encode frame")
                    continue

                processed_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

                # Send processed frame back to client
                ws.send(f"data:image/jpeg;base64,{processed_b64}")

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

        except Exception as e:
            print(f"WebSocket error: {e}")
            break


@app.route('/be/register', methods=["POST"])
def register():
    if session.get("logged_in"):
        return jsonify({
            "status": 0,
            "message": "Already logged-in"
        })

    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    db = Database("db.sqlite")
    out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
    db.close()

    if len(out) > 0:
        return jsonify({
            "status": 0,
            "message": "The user already registered"
        })

    newpass = utils.hash_password(password)
    db = Database("db.sqlite")
    db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (username, email, newpass))
    db.close()
    return jsonify({
        "status": 1,
        "message": "Registered"
    })

@app.route('/be/logout', methods=["POST"])
def be_logout():
    logged_in = session.get("logged_in")
    if not logged_in:
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })
    session.clear()
    return jsonify({
        "status": 1,
        "message": "Logged out"
    })

@app.route('/be/add-whitelist', methods=["POST"])
def be_add_whitelist():
    logged_in = session.get("logged_in")
    if not logged_in:
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })

    user_id = session.get("id")
    targetdir = ensure_user_whitelist_dir(user_id)

    if not request.is_json:
        return jsonify({
            "status": 0,
            "message": "Body need to be at JSON format."
        })

    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({
            "status": 0,
            "message": f"error parsing the body because: {e}."
        })

    imgdata = data.get("image", "")
    if len(imgdata) == 0:
        return jsonify({
            "status": 0,
            "message": "Empty data is not a valid data."
        })

    data_pattern = re.compile(r'data:image/.*?;base64,(.*)')
    match = data_pattern.match(imgdata)
    if match:
        encoded_data = match.group(1)
        extension_match = re.search(r'data:image/(\w+);', imgdata)
        extension = extension_match.group(1) if extension_match else 'png'
    else:
        encoded_data = imgdata
        extension = 'png'  # Default extension if header is missing

    try:
        image_bytes = base64.b64decode(encoded_data)
    except base64.binascii.Error as e:
        print(f"Error decoding Base64 data: {e}")
        return jsonify({
            "status": 0,
            "message": "Failed to decode the base64 string."
        })

    current = int(time.time())
    filename = f"{user_id}_{current}.{extension}"
    filepath = os.path.join(targetdir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    return jsonify({
        "status": 1,
        "message": "File saved! ",
        "file": {
            "name": filename,
            "url": url_for('static', filename=f"whitelist/{user_id}/{filename}", _external=False)
        }
    })

@app.route('/be/del-whitelist', methods=["POST"])
def be_del_whitelist():
    logged_in = session.get("logged_in")
    if not logged_in:
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })

    user_id = session.get("id")
    targetdir = ensure_user_whitelist_dir(user_id)

    if not request.is_json:
        return jsonify({
            "status": 0,
            "message": "Body need to be at JSON format."
        })

    data = request.get_json()
    filename = secure_filename(data.get("filename", ""))

    if not filename:
        return jsonify({
            "status": 0,
            "message": "Filename is required."
        })

    filepath = os.path.join(targetdir, filename)
    # Prevent path traversal
    if os.path.commonpath([os.path.abspath(filepath), os.path.abspath(targetdir)]) != os.path.abspath(targetdir):
        return jsonify({
            "status": 0,
            "message": "Invalid filename."
        })

    if not os.path.exists(filepath):
        return jsonify({
            "status": 0,
            "message": "File not found."
        })

    try:
        os.remove(filepath)
    except Exception as e:
        return jsonify({
            "status": 0,
            "message": f"Failed to delete file: {e}"
        })

    return jsonify({
        "status": 1,
        "message": "File deleted."
    })

@app.route('/be/list-whitelist', methods=["GET"])
def be_list_whitelist():
    logged_in = session.get("logged_in")
    if not logged_in:
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })

    user_id = session.get("id")
    targetdir = ensure_user_whitelist_dir(user_id)

    files = []
    for fname in sorted(os.listdir(targetdir)):
        fpath = os.path.join(targetdir, fname)
        if os.path.isfile(fpath):
            files.append({
                "name": fname,
                "url": url_for('static', filename=f"whitelist/{user_id}/{fname}", _external=False),
                "modified": int(os.path.getmtime(fpath))
            })

    return jsonify({
        "status": 1,
        "files": files
    })

if __name__ == '__main__':
    app.run(debug=True)
