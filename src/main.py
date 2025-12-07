from flask import Flask, render_template, session, redirect, request, jsonify
from connection import Database
import utils
import os
import re
import base64
import time

app = Flask(__name__)
app.secret_key = "INI KUNCI RAHASIA YANG TIDAK RAHASIA C4F3B4BE600DF00D"

db = Database("db.sqlite")
db.create_table_if_not_exist()

db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", ("admin", "admin@admin.com", utils.hash_password("1234")))
db.close()

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

@app.rout('/be/add-whitelist', methods=["POST"])
def be_add_whitelist():
    logged_in = session.get("logged_in")
    if not logged_in:
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })
    userid = session.get("id")
    targetdir = f"../{id}_whitelist"
    os.mkdirs(targetdir, exist_ok=True)
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

    imgdata = data.get("image")
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
        extension = 'png' # Default extension if header is missing

    try:
        image_bytes = base64.b64decode(encoded_data)
    except base64.binascii.Error as e:
        print(f"Error decoding Base64 data: {e}")
        return jsonify({
            "status": 0,
            "message": "Failed to decode the base64 string."
        })

    current = time.time()
    filename = f"{id}_{int(time)}.{extension}"
    filepath = os.path.join(targetdir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    return jsonify({
        "status": 1,
        "message": "File saved!"
    })

if __name__ == '__main__':
    app.run(debug=True)
