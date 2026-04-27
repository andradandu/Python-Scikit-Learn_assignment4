from flask import Flask, request, jsonify, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


df = pd.read_csv("data/obesity_classification.csv")
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Must match the columns used during training AFTER one-hot encoding
MODEL_COLUMNS = ["Age", "Height", "Weight", "BMI", "Gender_Male"]

def is_logged_in():
    return "user" in session


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if not user:
            return render_template("login.html", login_error="Username does not exist")

        if not check_password_hash(user.password, password):
            return render_template("login.html", login_error="Password is wrong")

        session["user"] = user.username
        return redirect("/dashboard")

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form["username"]
    password = request.form["password"]

    existing_user = User.query.filter_by(username=username).first()

    if existing_user:
        return render_template("login.html", register_error="Username already exists")

    new_user = User(
        username=username,
        password=generate_password_hash(password)
    )

    db.session.add(new_user)
    db.session.commit()

    session["user"] = new_user.username
    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect("/")

    return render_template("dashboard.html", username=session["user"])


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/")


@app.route("/api/shape")
def shape():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({
        "rows": df.shape[0],
        "columns": df.shape[1]
    })


@app.route("/api/columns")
def columns():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify([
        {
            "column": column,
            "dtype": str(dtype)
        }
        for column, dtype in df.dtypes.items()
    ])


@app.route("/api/head/<int:n>")
def head(n):
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify(df.head(n).to_dict(orient="records"))


@app.route("/api/tail/<int:n>")
def tail(n):
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify(df.tail(n).to_dict(orient="records"))


@app.route("/api/describe")
def describe():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify(df.describe().to_dict())


@app.route("/api/predict", methods=["POST"])
def predict():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "Age": float(data["Age"]),
            "Gender": data["Gender"],
            "Height": float(data["Height"]),
            "Weight": float(data["Weight"]),
            "BMI": float(data["BMI"])
        }])

        input_df = pd.get_dummies(input_df, columns=["Gender"], drop_first=True)

        input_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)