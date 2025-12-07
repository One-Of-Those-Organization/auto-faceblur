from flask import Flask, render_template, session, redirect, request, jsonify
from connection import Database
import utils
import re

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
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

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
    if no session.get("logged_in"):
        return jsonify({
            "status": 0,
            "message": "Need to be logged-in first."
        })
    session.clear()
    return jsonify({
        "status": 1,
        "message": "Logged out"
    })

if __name__ == '__main__':
    app.run(debug=True)
