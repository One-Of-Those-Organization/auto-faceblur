from flask import Flask, render_template, session, redirect, request
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
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template('camera.html')

# About Route
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

# Login Route
@app.route('/login', methods=['GET'])
def login():
    if session.get("logged_in"):
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
        return redirect("/camera")

    email = request.form["email"]
    password = request.form["password"]

    if len(email) == 0:
        return "Empty email is invalid"

    if len(password) == 0:
        return "Empty password is invalid"

    if re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        db = Database("db.sqlite")
        out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
        if len(out) == 0:
            return "User is not registered"

        row = out[0]
        db.close()
        if utils.verify_password(row["password"], password):
            session["logged_in"] = True
            session["user"] = row["name"]
            return redirect("/camera")
        else:
            return "Wrong password for that email"

    else:
        return "Wrong email format"

@app.route('/be/register', methods=["POST"])
def register():
    if session.get("logged_in"):
        return redirect("/camera")

    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    db = Database("db.sqlite")
    out = db.query("SELECT * from users where email = ?", (email,), get_output=True)
    db.close()

    if len(out) > 0:
        return "The user already registered"

    newpass = utils.hash_password(password)
    db = Database("db.sqlite")
    db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (username, email, newpass))
    db.close()
    return redirect("/login")

if __name__ == '__main__':
    app.run(debug=True)
