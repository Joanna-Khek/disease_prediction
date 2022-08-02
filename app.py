import pandas as pd 
import numpy as np
import os 
import pickle
from decimal import Decimal, ROUND_DOWN

from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask import get_flashed_messages

# Configure application
app = Flask(__name__)
app.secret_key = "password"

def read_dataframe(data_path, filename):
    df = pd.read_csv(os.path.join(data_path, filename))
    return df

def preprocess_train(df):
    df["prognosis"] = df["prognosis"].apply(lambda x: x.replace("Paralysis (brain hemorrhageH", "Paralysis (brain hemorrhage)"))
    return df

def preprocess_symptom(df):
    df["Symptom Name"] = df["Symptom"].apply(lambda x: x.replace("_", " "))
    df["Symptom Name"] = df["Symptom Name"].apply(lambda x: x.title())
    return df

def preprocess_description(df):
    df["Disease"] = df["Disease"].apply(lambda x: x.title())
    return df

def preprocess_precaution(df):
    df["Disease"] = df["Disease"].apply(lambda x: x.title())
    df["Symptom_precaution_0"] = df["Symptom_precaution_0"].apply(lambda x: str(x).capitalize())
    df["Symptom_precaution_1"] = df["Symptom_precaution_1"].apply(lambda x: str(x).capitalize())
    df["Symptom_precaution_2"] = df["Symptom_precaution_2"].apply(lambda x: str(x).capitalize())
    df["Symptom_precaution_3"] = df["Symptom_precaution_3"].apply(lambda x: str(x).capitalize())
    df["Precautions"] = df["Symptom_precaution_0"] + ". " +  df["Symptom_precaution_1"] + ". " + df["Symptom_precaution_2"] + ". " + df["Symptom_precaution_3"]
    return df

data_path = os.path.join("disease-symptom-prediction")
df_symptoms = read_dataframe(data_path, "symptom_severity.csv")
df_train = read_dataframe(data_path, "Training.csv")
df_desc = read_dataframe(data_path, "disease_description.csv")
df_prec = read_dataframe(data_path, "disease_precaution.csv")
df_train_processed = preprocess_train(df_train)
df_symptoms_processed = preprocess_symptom(df_symptoms)
df_desc_processed = preprocess_description(df_desc)
df_prec_processed = preprocess_precaution(df_prec)
loaded_model = pickle.load(open("rf_model.sav", 'rb'))
    
@app.route("/")
def index():
    # severities tier
    S1 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 1]["Symptom Name"]
    S2 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 2]["Symptom Name"]
    S3 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 3]["Symptom Name"]
    S4 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 4]["Symptom Name"]
    S5 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 5]["Symptom Name"]
    S6 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 6]["Symptom Name"]
    S7 = df_symptoms_processed[df_symptoms_processed["Symptom_severity"] == 7]["Symptom Name"]

    # else:
    return render_template("index.html", s1=S1, s2=S2, s3=S3, s4=S4, s5=S5, s6=S6, s7=S7)

@app.route("/symptoms", methods=["POST", "GET"])
def symptoms():
    s1_value = request.form.getlist('s1')
    s2_value = request.form.getlist('s2')
    s3_value = request.form.getlist('s3')
    s4_value = request.form.getlist('s4')
    s5_value = request.form.getlist('s5')
    s6_value = request.form.getlist('s6')
    s7_value = request.form.getlist('s7')
    
    ALL_SYMPTOMS = s1_value + s2_value + s3_value + s4_value + s5_value + s6_value + s7_value
    ALL_SYMPTOMS_PROCESSED = [x.replace(" ", "_").lower() for x in ALL_SYMPTOMS]
    print(ALL_SYMPTOMS_PROCESSED)
    
    col_names = list(df_train.columns[:-1])
    
    df = pd.DataFrame(0, index=np.arange(1), columns=col_names)
    v = df.values.tolist()
    c = df.columns.values.tolist()
    df_dict = [dict(zip(c, x)) for x in v][0]
    
    for symp in ALL_SYMPTOMS_PROCESSED:
        df_dict[symp] = 1
    
    df_test = pd.DataFrame(df_dict.items(),columns=["Disease", "Indicator"])
    print(df_test)
    dict_df_test = df_test.to_dict("list")
    session["dict_df_test"] = dict_df_test
    session["all_symptoms"] = ALL_SYMPTOMS
    return render_template("prediction.html", all_symptoms=ALL_SYMPTOMS)

@app.route("/results", methods=["POST", "GET"])
def results():
    ALL_SYMPTOMS = session["all_symptoms"] 
    df = pd.DataFrame(session["dict_df_test"])
    test_values = np.array(df["Indicator"]).reshape(1, -1)
    PREDICTION = loaded_model.predict(test_values)[0].title()
    predict_prob = loaded_model.predict_proba(test_values)
    
    # dataframe for results
    results = pd.DataFrame()
    results["Disease"] = df_train["prognosis"].unique()
    results["Probability"] = predict_prob.tolist()[0]
    results["Disease"] = results["Disease"].apply(lambda x: x.title())

    # get top 3 diseases
    top3_symp = results.nlargest(3, "Probability")
    top3_symp = top3_symp.reset_index().drop("index", axis=1)
    top3_symp["Probability"] = top3_symp["Probability"].apply(lambda x: str(Decimal(x).quantize(Decimal(10)**-3, rounding=ROUND_DOWN)*100) + "%")
    
    # merge description
    RESULTS_DESC = top3_symp.merge(df_desc_processed, left_on="Disease", right_on="Disease", how="left")
    
    # merge precaution
    RESULTS_DESC = RESULTS_DESC.merge(df_prec_processed, left_on="Disease", right_on="Disease", how="left")
    
    # get description
    flash(PREDICTION)
    
    return render_template("results.html", prediction=PREDICTION, all_symptoms=ALL_SYMPTOMS, 
                           results_desc=RESULTS_DESC, zip=zip)

if __name__ == "__main__":
    app.run(debug=False)