from flask import Flask, render_template, session, redirect, request, jsonify, url_for
from flask_sock import Sock
from werkzeug.utils import secure_filename
from src.connection import Database
from src import utils
import os
import re
import base64
import time
import secrets
import cv2
import numpy as np
import traceback
from src.model import ActiveModel, ActiveModelConfig

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
db.close()

# -----------------------
# -- WebSocket & CV Setup --
# -----------------------

MAX_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / MAX_FPS  # ~0.033 seconds between frames

# Token storage for WebSocket authentication
ws_tokens = {}
WS_TOKEN_EXPIRY = 30  # Token expires after 30 seconds

def cleanup_expired_tokens():
    current = time.time()
    expired = [t for t, data in ws_tokens.items() if current - data["created"] > WS_TOKEN_EXPIRY]
    for t in expired:
        del ws_tokens[t]

def process_frame(model, img):
    out_frame, detections = model.predict_frame(img)
    return out_frame

def get_user_whitelist_dir(user_id: int) -> str:
    return os.path.join(app.static_folder, "whitelist", str(user_id))

def ensure_user_whitelist_dir(user_id: int) -> str:
    path = get_user_whitelist_dir(user_id)
    os.makedirs(path, exist_ok=True)
    return path

# --------------------
# -- Frontend route --
# --------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/camera', methods=['GET'])
def camera():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template('camera.html')

@app.route('/login', methods=['GET'])
def login():
    if session.get("logged_in"):
        return redirect("/camera")
    return render_template('login.html')

# -------------------
# -- Backend Route --
# -------------------
@app.route('/be/login', methods=['POST'])
def be_login():
    if session.get("logged_in"):
        return jsonify({"status": 0, "message": "Already logged-in"})
    email = request.form.get("email", "")
    password = request.form.get("password", "")

    if len(email) == 0:
        return jsonify({"status": 0, "message": "Empty email is invalid"})
    if len(password) == 0:
        return jsonify({"status": 0, "message": "Empty password is invalid"})

    if re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        db = Database("db.sqlite")
        out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
        db.close()
        if len(out) == 0:
            return jsonify({"status": 0, "message": "User is not registered"})
        row = out[0]
        if utils.verify_password(row["password"], password):
            session["logged_in"] = True
            session["user"] = row["name"]
            session["id"] = row["id"]
            return jsonify({"status": 1, "message": {"user": row["name"], "email": row["email"]}})
        else:
            return jsonify({"status": 0, "message": "Wrong password for that email"})
    else:
        return jsonify({"status": 0, "message": "Wrong email format"})

@app.route('/be/ws-token', methods=['POST'])
def get_ws_token():
    if not session.get("logged_in"):
        return jsonify({"status": 0, "message": "Unauthorized"}), 401
    cleanup_expired_tokens()
    token = secrets.token_urlsafe(32)
    user_id = session.get("id")
    ws_tokens[token] = {"user_id": user_id, "created": time.time()}
    return jsonify({"status": 1, "token": token})

@sock.route('/ws/camera')
def ws_camera(ws):
    try:
        auth_msg = ws.receive(timeout=5)
        if auth_msg is None:
            ws.send("error:no_token")
            ws.close()
            return

        if auth_msg not in ws_tokens:
            ws.send("error:unauthorized")
            ws.close()
            return

        token_data = ws_tokens.pop(auth_msg)
        user_id = token_data["user_id"]
        ws.send("auth:ok")
    except Exception as e:
        print(f"WebSocket auth error: {e}")
        traceback.print_exc()
        try:
            ws.send("error:auth_failed")
            ws.close()
        except Exception:
            pass
        return

    # Setup model with user's whitelist
    user_whitelist_dir = os.path.join(app.static_folder, "whitelist", str(user_id))
    os.makedirs(user_whitelist_dir, exist_ok=True)

    cfg = ActiveModelConfig(selected_model="facenet", whitelist_dir=user_whitelist_dir)
    model = ActiveModel(cfg)

    print(f"WebSocket connected for user_id: {user_id}, whitelist: {user_whitelist_dir}")

    last_frame_time = 0.0
    FRAME_MAX_AGE = 0.25  # seconds; drop frames older than this

    while True:
        try:
            data = ws.receive()
            # If connection closed, receive returns None
            if data is None:
                print("WebSocket receive returned None (client disconnected)")
                try:
                    ws.close()
                except Exception:
                    pass
                break

            # ----- UDP-style: parse timestamp and drop stale frames -----
            # Expected format from client:
            #   "ts:<unix_ts>,<data:image/jpeg;base64,...>" OR "ts:<unix_ts>,<raw_base64>"
            frame_time = time.time()
            if isinstance(data, str) and data.startswith("ts:"):
                try:
                    header, rest = data.split(",", 1)
                    _, ts_str = header.split(":", 1)
                    frame_time = float(ts_str)
                    data = rest
                except Exception as e:
                    # If parsing timestamp fails, fall back to now
                    print(f"Timestamp parse failed: {e}")
                    frame_time = time.time()

            now = time.time()
            if now - frame_time > FRAME_MAX_AGE:
                # Frame too old, drop it completely (no processing, no send)
                continue
            # ------------------------------------------------------------

            # FPS limiting, but by dropping frames instead of sleeping
            elapsed = now - last_frame_time
            if elapsed < MIN_FRAME_INTERVAL:
                # We're still within the FPS window: drop this frame
                continue

            last_frame_time = now

            try:
                if isinstance(data, str) and ',' in data:
                    # handle data:url;base64,... or raw base64
                    data = data.split(',', 1)[1]

                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Decoded frame is None, skipping")
                    continue

                processed = process_frame(model, frame)
                if processed is None:
                    print("Processing returned None, sending original frame")
                    processed = frame

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success, buffer = cv2.imencode(".jpg", processed, encode_param)

                if not success:
                    print("Failed to encode frame")
                    continue

                processed_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                ws.send(f"data:image/jpeg;base64,{processed_b64}")

            except Exception as e:
                # On processing exception: break loop and close socket to avoid infinite loop
                print(f"Error processing frame: {e}")
                traceback.print_exc()
                try:
                    ws.send("error:processing_failed")
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass
                break

        except Exception as e:
            # Any other socket-level exception: log and break
            print(f"WebSocket error: {e}")
            traceback.print_exc()
            try:
                ws.close()
            except Exception:
                pass
            break

    print(f"WebSocket disconnected for user_id: {user_id}")

@app.route('/be/register', methods=["POST"])
def register():
    if session.get("logged_in"):
        return jsonify({"status": 0, "message": "Already logged-in"})

    username = request.form.get("username", "")
    email = request.form.get("email", "")
    password = request.form.get("password", "")

    db = Database("db.sqlite")
    out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
    db.close()

    if len(out) > 0:
        return jsonify({"status": 0, "message": "The user already registered"})

    newpass = utils.hash_password(password)
    db = Database("db.sqlite")
    db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (username, email, newpass))
    db.close()
    return jsonify({"status": 1, "message": "Registered"})

@app.route('/be/logout', methods=["POST"])
def be_logout():
    if not session.get("logged_in"):
        return jsonify({"status": 0, "message": "Need to be logged-in first."})
    session.clear()
    return jsonify({"status": 1, "message": "Logged out"})

@app.route('/be/add-whitelist', methods=["POST"])
def be_add_whitelist():
    if not session.get("logged_in"):
        return jsonify({"status": 0, "message": "Need to be logged-in first."})

    user_id = session.get("id")
    targetdir = ensure_user_whitelist_dir(user_id)

    if not request.is_json:
        return jsonify({"status": 0, "message": "Body need to be at JSON format."})

    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"status": 0, "message": f"error parsing the body because: {e}."})

    imgdata = data.get("image", "")
    if len(imgdata) == 0:
        return jsonify({"status": 0, "message": "Empty data is not a valid data."})

    # corrected pattern: capture after 'base64,'
    data_pattern = re.compile(r'data:image/.*;base64,(.*)', re.DOTALL)
    match = data_pattern.match(imgdata)
    if match:
        encoded_data = match.group(1)
        extension_match = re.search(r'data:image/(\w+);', imgdata)
        extension = extension_match.group(1) if extension_match else 'png'
    else:
        encoded_data = imgdata
        extension = 'png'

    try:
        image_bytes = base64.b64decode(encoded_data)
    except base64.binascii.Error as e:
        print(f"Error decoding Base64 data: {e}")
        return jsonify({"status": 0, "message": "Failed to decode the base64 string."})

    current = int(time.time())
    filename = f"{user_id}_{current}.{extension}"
    filepath = os.path.join(targetdir, secure_filename(filename))
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    return jsonify({
        "status": 1,
        "message": "File saved!",
        "file": {
            "name": filename,
            "url": url_for('static', filename=f"whitelist/{user_id}/{filename}", _external=False)
        }
    })

@app.route('/be/del-whitelist', methods=["POST"])
def be_del_whitelist():
    if not session.get("logged_in"):
        return jsonify({"status": 0, "message": "Need to be logged-in first."})

    user_id = session.get("id")
    targetdir = ensure_user_whitelist_dir(user_id)

    if not request.is_json:
        return jsonify({"status": 0, "message": "Body need to be at JSON format."})

    data = request.get_json()
    filename = secure_filename(data.get("filename", ""))

    if not filename:
        return jsonify({"status": 0, "message": "Filename is required."})

    filepath = os.path.join(targetdir, filename)
    if os.path.commonpath([os.path.abspath(filepath), os.path.abspath(targetdir)]) != os.path.abspath(targetdir):
        return jsonify({"status": 0, "message": "Invalid filename."})

    if not os.path.exists(filepath):
        return jsonify({"status": 0, "message": "File not found."})

    try:
        os.remove(filepath)
    except Exception as e:
        return jsonify({"status": 0, "message": f"Failed to delete file: {e}"})

    return jsonify({"status": 1, "message": "File deleted."})

@app.route('/be/list-whitelist', methods=["GET"])
def be_list_whitelist():
    if not session.get("logged_in"):
        return jsonify({"status": 0, "message": "Need to be logged-in first."})

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

    return jsonify({"status": 1, "files": files})

if __name__ == '__main__':
    app.run(debug=True)
