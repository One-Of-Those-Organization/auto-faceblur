from flask import Flask, render_template

app = Flask(__name__)


# Home Route
@app.route('/')
def index():
    return render_template('index.html')


# Camera Route
@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')


# About Route
@app.route('/about')
def about():
    return render_template('about.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


# Register Route
@app.route('/register')
def register():
    return render_template('register.html')


if __name__ == '__main__':
    app.run(debug=True)
