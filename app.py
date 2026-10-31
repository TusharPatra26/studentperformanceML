import json
import os

from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    report_path = os.path.join(
        application.root_path, "artifacts", "model_report.json"
    )
    model_report = None
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as report_file:
            model_report = json.load(report_file)
    return render_template("about.html", model_report=model_report)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", form_data={}, results=None)

    form_data = request.form.to_dict()
    try:
        reading_score = int(form_data.get("reading_score", ""))
        writing_score = int(form_data.get("writing_score", ""))
        if not 0 <= reading_score <= 100 or not 0 <= writing_score <= 100:
            raise ValueError("Reading and writing scores must be between 0 and 100.")

        data = CustomData(
            gender=form_data.get("gender", ""),
            race_ethnicity=form_data.get("ethnicity", ""),
            parental_level_of_education=form_data.get(
                "parental_level_of_education", ""
            ),
            lunch=form_data.get("lunch", ""),
            test_preparation_course=form_data.get("test_preparation_course", ""),
            reading_score=reading_score,
            writing_score=writing_score,
        )
        prediction = PredictPipeline().predict(data.get_data_as_data_frame())
    except ValueError as error:
        return (
            render_template(
                "home.html", form_data=form_data, results=None, error=str(error)
            ),
            400,
        )

    return render_template(
        "home.html", form_data=form_data, results=float(prediction[0]), error=None
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)