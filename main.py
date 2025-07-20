import pickle
import numpy as np

stroke = False

with open('stroke_predictor.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define dictionaries for encoding
gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
ever_married_dict = {'No': 0, 'Yes': 1}
work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
residence_type_dict = {'Rural': 0, 'Urban': 1}
smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}

def encode_input(gender, age, hypertension, heartdisease, evermarried, worktype, residencetype, avgglucoselevel, bmi, smokingstatus):
    gender_encoded = gender_dict.get(gender, 2)  # 2 for 'Other' if not found
    ever_married_encoded = ever_married_dict.get(evermarried, 0)  # 0 for 'No' if not found
    work_type_encoded = work_type_dict.get(worktype, 0)  # 0 for 'children' if not found
    residence_type_encoded = residence_type_dict.get(residencetype, 0)  # 0 for 'Rural' if not found
    smoking_status_encoded = smoking_status_dict.get(smokingstatus, 0)  # 0 for 'Unknown' if not found
    
    return np.array([[gender_encoded, age, hypertension, heartdisease, ever_married_encoded,
                      work_type_encoded, residence_type_encoded, avgglucoselevel, bmi, smoking_status_encoded]])

def prediction(values):
    prediction = loaded_model.predict(values)
    return prediction[0]  # Return the predicted value

def query():
    gender = input('gender (Male = 0, Female = 1, Other = 2): ')
    age = int(input('age: '))
    hypertension = int(input('hypertension (0 for false, 1 for true): '))
    heartdisease = int(input('heart_disease (0 for false, 1 for true): '))
    evermarried = input('ever married (No = 0, Yes = 1): ')
    worktype = input('work type (children = 0, Never_worked = 1, Govt_job = 2, Private = 3, Self-employed = 4): ')
    residencetype = input('residence type (Rural = 0, Urban = 1): ')
    avgglucoselevel = float(input('Average Glucose Level: '))
    bmi = float(input('bmi: '))
    smokingstatus = input('Smoking Status (Unknown = 0, never smoked = 1, formerly smoked = 2, smokes = 3): ')
    
    array = encode_input(gender, age, hypertension, heartdisease, evermarried, worktype, residencetype, avgglucoselevel, bmi, smokingstatus)
    result = prediction(array)
    return result

def main():
    input_result = query()
    if int(input_result) == 1:
        print('Stroke is likely')
    elif int(input_result) == 0:
        print('Stroke is unlikely')
    else:
        print('Prediction error')

main()
