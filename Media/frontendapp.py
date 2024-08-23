from flask import Flask, redirect, url_for, render_template

# Create an instance of a flask application
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/<name>") # name inside angled brackets will be grabbed and passed to user function 
def user(name):
    return f"Hello {name}"

if __name__ == "__main__":
    app.run()