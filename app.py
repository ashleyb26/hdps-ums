import numpy as np
import pickle
from flask import Flask, url_for, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView
from flask_login import UserMixin, LoginManager, current_user, login_user, logout_user
from flask_admin.contrib.sqla import ModelView
from flask_admin.menu import MenuLink


# Load ML model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.db"
app.config["SECRET_KEY"] = "secretkey"

db = SQLAlchemy(app)
login = LoginManager(app)


@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class User(db.Model, UserMixin):
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    usertype = db.Column(db.String(20), default="user")

    def __init__(self, email, password, name, usertype):
        self.email = email
        self.password = password
        self.name = name
        self.usertype = usertype


class MyModelView(ModelView):
    def is_accessible(self):
        user = User.query.filter_by(usertype=current_user.usertype).first()
        role = user.usertype == "admin"
        return role

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for("login"))


class MyAdminIndexView(AdminIndexView):
    def is_accessible(self):
        user = User.query.filter_by(usertype=current_user.usertype).first()
        role = user.usertype == "admin"
        return role


class LogoutMenuLink(MenuLink):
    def is_accessible(self):
        user = User.query.filter_by(usertype=current_user.usertype).first()
        role = user.usertype == "admin"
        return role


admin = Admin(app, index_view=MyAdminIndexView())
admin.add_view(MyModelView(User, db.session))
admin.add_link(LogoutMenuLink(name="Logout", category="", url="/logout"))


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template(
            "Heart Disease Prediction Test.html", name=current_user.name
        )
    else:
        return render_template("log in.html")


@app.route("/login", methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect("hdps")
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email, password=password).first()
        if user is not None:
            login_user(user)
            role = current_user.usertype
            if role == "admin":
                return redirect("/admin/user")
            else:
                return redirect("hdps")
        else:
            return redirect("/")


@app.route("/hdps")
def hdps():
    if current_user.is_authenticated:
        return render_template(
            "Heart Disease Prediction Test.html", name=current_user.name
        )
    else:
        return redirect("/")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    name = current_user.name
    features = [float(i) for i in request.form.values()]
    print(features)
    array_features = [np.array(features)]
    print(array_features)
    array_features = scaler.transform(array_features)
    prediction = model.predict(array_features)
    output = prediction
    if output == 0:
        return render_template(
            "Heart Disease Prediction Test.html",
            result_no="Result: Not Affected with Heart Disease",
            name=current_user.name,
        )
    else:
        return render_template(
            "Heart Disease Prediction Test.html",
            result_yes="Result: Affected with Heart Disease",
            name=current_user.name,
        )


@app.route("/logout")
def logout():
    logout_user()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
