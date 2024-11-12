import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

connection = psycopg2.connect(user = "postgres", password = "Agave!", host = "127.0.0.1", port = "5432", database = "Diabeties")
cursor = connection.cursor()
cursor.execute("Select * from diabetesdata LIMIT 10")
rows = cursor.fetchall()

print()


for data in rows:
    print(data)

print()
print()


df = pd.read_csv("diabetes.csv")
print(df)
columns_to_check = ['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']


for column in columns_to_check:
    df = df[df[column] != 0]
    
    
#print(df.info())
print(df.head())
print()

#how big updated list is
print(df.shape)



y = df['Outcome']
print(y)

print()


x = df.drop('Outcome', axis=1)
print(x)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.2, random_state=100)
print()
print("X train Data:")
print(X_train)
print()
print("X test Data:")

print(X_test)


### Model building
###Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)


#applying the model to create prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
 
print()
print()
print("Y LR train pred")
print(y_lr_train_pred)
print()
print('Y LR test prediction')
print()
print(y_lr_test_pred)

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mean_squared_error = mean_squared_error(Y_train, y_lr_train_pred)
lr_train_r2 = r2_score(Y_train, y_lr_train_pred)



lr_test_mean_squared_error = mean_squared_error(Y_test, y_lr_test_pred)
lr_test_r2 = r2_score(Y_test, y_lr_test_pred)


print()
print(lr_train_mean_squared_error)
print(lr_train_r2)
print(lr_test_mean_squared_error)
print(lr_test_r2)
print()

lr_results = pd.DataFrame(['Linear Regression', lr_train_mean_squared_error, lr_train_r2, lr_test_mean_squared_error, lr_test_r2]).transpose()
lr_results.columns = ['Methods', 'Train MSE', 'Train R2', 'Test MSE', 'Test R2']
print(lr_results)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300)

# Instantiate the Random Forest Classifier
#86% accurage rate
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Detailed evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)









# Function to predict diabetes for user input
def predict_diabetes(user_input):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Predict the outcome and probability
    prediction = rf_model.predict(user_df)
    probability = rf_model.predict_proba(user_df)
    
    # Get the probability for the positive class (diabetes = 1)
    probability_positive = probability[0][1] * 100
    
    # Provide results
    outcome = 'Positive' if prediction[0] == 1 else 'Negative'
    return outcome, probability_positive

# Gather user input

def main():
    pregnancies = int(input("Enter number of pregnancies: "))
    glucose = float(input("Enter glucose level: "))
    blood_pressure = float(input("Enter blood pressure: "))
    skin_thickness = float(input("Enter skin thickness: "))
    insulin = float(input("Enter insulin level: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree_function = float(input("Enter diabetes pedigree function: "))
    age = int(input("Enter age: "))

    # Prepare user input
    user_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # Get prediction and probability
    outcome, probability_positive = predict_diabetes(user_input)

    # Display the results
    print(f"\nPredicted Outcome: {outcome}")
    print(f"Probability of Diabetes: {probability_positive:.2f}%")
    user_info = input("Want to analyze your chances? (Y) or (N)")
    if user_info.lower() == 'y':
        main()
    else:
        quit

main()
