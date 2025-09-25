from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

with open(r'C:\dl_projects\spam_ham_classifier\notebook\spam_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r"C:\dl_projects\spam_ham_classifier\notebook\vectorizer.pkl","rb") as file:
    vectorize=pickle.load(file)

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    vectorize_meassage=vectorize.transform([message])
  
    prediction = model.predict(vectorize_meassage)[0]
    
    result = "spam" if prediction == 1 else "ham"
    
    return f"The message is: {result}"

if __name__ == "__main__":
    app.run(debug=True)
