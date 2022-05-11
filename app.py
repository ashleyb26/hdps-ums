import numpy as np
import pandas as pd
import pickle
import sqlalchemy
from flask import Flask, url_for, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView
from flask_login import UserMixin, LoginManager, current_user, login_user, logout_user
from flask_admin.contrib.sqla import ModelView
from flask_admin.menu import MenuLink
from datetime import datetime
import plotly
import plotly.express as px
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

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
    name = db.Column(db.String(1000), unique=True)
    usertype = db.Column(db.String(20), default="user")

    def __init__(self, email, password, name, usertype):
        self.email = email
        self.password = password
        self.name = name
        self.usertype = usertype


class History(db.Model):
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    name = db.Column(db.String(1000))
    age = db.Column(db.String(1000))
    sex = db.Column(db.String(1000))
    cp = db.Column(db.String(1000))
    trestbps = db.Column(db.String(1000))
    chol = db.Column(db.String(1000))
    fbs = db.Column(db.String(1000))
    restecg = db.Column(db.String(1000))
    thalach = db.Column(db.String(1000))
    exang = db.Column(db.String(1000))
    oldpeak = db.Column(db.String(1000))
    slope = db.Column(db.String(1000))
    ca = db.Column(db.String(1000))
    thal = db.Column(db.String(1000))
    result = db.Column(db.String(1000))
    currentdate = db.Column(db.String(1000))

    def __init__(self, name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, currentdate):
        self.name = name
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.thalach = thalach
        self.exang = exang
        self.slope = slope
        self.oldpeak = oldpeak
        self.ca = ca
        self.thal = thal
        self.result = result
        self.currentdate = currentdate


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
admin.add_view(MyModelView(History, db.session))
# admin.add_view(MyModelView(Analysis, db.session))
admin.add_link(LogoutMenuLink(name="Logout", category="", url="/logout"))


def addHistory(featuresHistory):

    name = current_user.name
    if featuresHistory[1] == "0":
        featuresHistory[1] = "Male"
    elif featuresHistory[1] == "1":
        featuresHistory[1] = "Female"
    # CP
    if featuresHistory[2] == "1":
        featuresHistory[2] = "Typical Angina"
    elif featuresHistory[2] == "2":
        featuresHistory[2] = "Atypical Angina"
    elif featuresHistory[2] == "3":
        featuresHistory[2] = "Non Anginal Pain"
    elif featuresHistory[2] == "4":
        featuresHistory[2] = "Asymptomatic"

    if featuresHistory[5] == "1":
        featuresHistory[5] = "True"
    elif featuresHistory[5] == "0":
        featuresHistory[5] = "False"

    if featuresHistory[6] == "0":
        featuresHistory[6] = "Normal"
    elif featuresHistory[6] == "1":
        featuresHistory[6] = "Abnormal"
    elif featuresHistory[6] == "2":
        featuresHistory[6] = "Probable"

    if featuresHistory[8] == "1":
        featuresHistory[8] = "Yes"
    elif featuresHistory[8] == "0":
        featuresHistory[8] = "No"

    if featuresHistory[10] == "1":
        featuresHistory[10] = "Upsloping"
    elif featuresHistory[10] == "2":
        featuresHistory[10] = "Flat"
    elif featuresHistory[10] == "3":
        featuresHistory[10] = "Downsloping"

    if featuresHistory[12] == "0":
        featuresHistory[12] = "Normal"
    elif featuresHistory[12] == "3":
        featuresHistory[12] = "Fixed"
    elif featuresHistory[12] == "6":
        featuresHistory[12] = "Defect"
    elif featuresHistory[12] == "7":
        featuresHistory[12] = "Reversable"

    if featuresHistory[13] == "1":
        featuresHistory[13] = "Yes"
    elif featuresHistory[13] == "0":
        featuresHistory[13] = "No"

    add = History(name, featuresHistory[0], featuresHistory[1], featuresHistory[2], featuresHistory[3], featuresHistory[4], featuresHistory[5], featuresHistory[6],
                  featuresHistory[7], featuresHistory[8], featuresHistory[9], featuresHistory[10], featuresHistory[11], featuresHistory[12], featuresHistory[13], featuresHistory[14])
    db.session.add(add)
    db.session.commit()


@app.route("/")
def index():
    if current_user.is_authenticated:
        return render_template(
            "Heart Disease Prediction Test.html", name=current_user.name
        )
    else:
        return render_template("Log In.html")


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
            flash("Invalid Login, Please Log In back")
            return redirect("/")


@app.route("/hdps")
def hdps():
    if current_user.is_authenticated:
        return render_template(
            "Heart Disease Prediction Test.html", name=current_user.name
        )
    else:
        return redirect("/")


@app.route("/history")
def history():
    if current_user.is_authenticated:
        name2 = current_user.name
        user = History.query.filter(
            History.name == name2).all()
        ids = History.query.with_entities(History.age).all()
        return render_template(
            "History.html", user=user, name3=name2, ids=ids
        )
    else:
        return redirect("/")


@app.route('/delete/<id>/', methods=['GET', 'POST'])
def delete(id):
    my_data = History.query.get(id)
    db.session.delete(my_data)
    db.session.commit()

    return redirect("/history")


@app.route("/analysis")
def analysisHistory():
    if current_user.is_authenticated:
        name = current_user.name
        engine = sqlalchemy.create_engine('sqlite:///db.db')
        user = History.query.filter(
            History.name == name).count()
        if (user > 0):
            data = pd.read_sql_table("history", engine, columns=[
                'name', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'result'])
            data = data[(data.name == name)]
            cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            data[cols] = data[cols].apply(
                pd.to_numeric, errors='coerce', axis=1)

            fig1 = px.pie(data, names='result',
                          color_discrete_sequence=px.colors.sequential.Aggrnyl,
                          title='Result Distribution')
            graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

            fig2 = px.pie(data, names='sex',
                          color_discrete_sequence=px.colors.sequential.Aggrnyl,
                          hole=0.8,
                          title='Gender Distribution')
            graph2JSON = json.dumps(
                fig2, cls=plotly.utils.PlotlyJSONEncoder)

            fig3 = px.scatter(data, x='trestbps', y='chol', title='Cholestrol vs Blood Pressure',
                              facet_col='sex',
                              color='result', template='ggplot2')
            graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

            fig4 = pd.DataFrame(data.groupby('result')[
                                'slope'].value_counts().sort_index())
            fig4 = fig4.rename_axis(['result', 'Slope']).reset_index()
            fig4['Counts'] = fig4['slope']
            fig4 = px.bar(fig4, x='Slope', y='Counts',
                          facet_col='result', color_discrete_sequence=px.colors.sequential.Aggrnyl,
                          title='Slope Distribution Across Target')
            graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

            x = ['Typical Angina', 'Atypical Angina',
                 'Non Anginal Pain', 'Asymptomatic']
            y = [len(data[(data["cp"] == "Typical Angina") & (data["result"] == "No")]),
                 len(data[(data["cp"] == "Atypical Angina")
                     & (data["result"] == "No")]),
                 len(data[(data["cp"] == "Non Anginal Pain")
                     & (data["result"] == "No")]),
                 len(data[(data["cp"] == "Asymptomatic") & (data["result"] == "No")])]
            y2 = [len(data[(data["cp"] == "Typical Angina") & (data["result"] == "Yes")]),
                  len(data[(data["cp"] == "Atypical Angina")
                      & (data["result"] == "Yes")]),
                  len(data[(data["cp"] == "Non Anginal Pain")
                      & (data["result"] == "Yes")]),
                  len(data[(data["cp"] == "Asymptomatic") & (data["result"] == "Yes")])]

            trace1 = go.Bar(
                x=x,
                y=y,
                text=y,
                textposition='auto',
                name='No',
                marker=dict(
                    color='rgba(255, 135, 141,0.7)',
                    line=dict(
                        color='rgba(255, 135, 141,1)',
                        width=1.5),
                ),
                opacity=1
            )

            trace2 = go.Bar(
                x=x,
                y=y2,
                text=y2,
                textposition='auto',
                name='Yes',
                marker=dict(
                    color='rgba(50, 171, 96, 0.7)',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=1.5),
                ),
                opacity=1
            )
            fig5 = [trace1, trace2]
            graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

            trace = go.Histogram(
                x=data['age'], name='age', marker=dict(color='darkcyan'))

            layout = go.Layout(
                title="Histogram Frequency Counts of Age"
            )

            fig6 = go.Figure(data=go.Data([trace]), layout=layout)
            graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template(
                "Analysis.html", data=data, name=name, graph1JSON=graph1JSON, graph2JSON=graph2JSON, graph3JSON=graph3JSON, graph4JSON=graph4JSON, graph5JSON=graph5JSON, graph6JSON=graph6JSON
            )
        else:
            flash("No data to analyse. Please input some data.")
            return redirect("/")
    else:
        return redirect("/")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    name = current_user.name
    featuresHistory = [str(i) for i in request.form.values()]
    featuresPredict = [float(i) for i in request.form.values()]
    array_features = [np.array(featuresPredict)]
    array_features = scaler.transform(array_features)
    prediction = model.predict(array_features)
    output = prediction
    currentdate = datetime.now()
    currentdate = currentdate.strftime("%d/%b/%Y %H:%M:%S")
    featuresHistory.extend([str(int(prediction)), currentdate])
    addHistory(featuresHistory)

    if output == 0:
        return render_template(
            "Heart Disease Prediction Test.html",
            result_no="Result: Not Affected with Heart Disease(This prediction is 92.17"
            + "%"
            + " accurate)",
            name=name
        )
    else:
        return render_template(
            "Heart Disease Prediction Test.html",
            result_yes="Result: Affected with Heart Disease (This prediction is 92.17"
            + "%"
            + " accurate)",
            name=name
        )


@app.route("/logout")
def logout():
    logout_user()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
