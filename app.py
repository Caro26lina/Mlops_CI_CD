from flask import Flask, request, render_template_string
import pickle
import numpy as np

# --- App Setup ---
app = Flask(__name__)

# Load model (assuming "artifacts/model.pkl" exists)
try:
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'artifacts/model.pkl' not found. Using dummy model.")
    class DummyModel:
        def predict(self, data):
            # Dummy logic for fallback
            return np.array([1 if np.sum(data) > 100 else 0])
    model = DummyModel()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    class DummyModel:
        def predict(self, data):
            return np.array([0])
    model = DummyModel()

# --- Feature Configuration (Simplified for Dense Layout) ---
# Features are maintained in a flat list for simplified layout, 
# relying on the two-column grid to manage density.

feature_config = {
    "age": {"label": "Age", "unit": "years", "type": "continuous", "placeholder": "e.g., 52", "desc": "Patient Age"},
    "sex": {"label": "Sex", "type": "categorical", "options": {1: "Male", 0: "Female"}, "desc": "1=Male, 0=Female"},
    "cp": {"label": "Chest Pain Type", "type": "categorical", "options": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}, "desc": "Type of chest pain"},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "type": "continuous", "placeholder": "e.g., 125", "desc": "Systolic BP on admission"},
    "chol": {"label": "Serum Cholesterol", "unit": "mg/dl", "type": "continuous", "placeholder": "e.g., 216", "desc": "Total serum cholesterol"},
    "fbs": {"label": "Fasting Blood Sugar", "type": "categorical", "options": {1: "True (> 120 mg/dl)", 0: "False (≤ 120 mg/dl)"}, "desc": "Is FBS > 120 mg/dl?"},
    "restecg": {"label": "Resting ECG Results", "type": "categorical", "options": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}, "desc": "ECG abnormality type"},
    "thalach": {"label": "Max Heart Rate Achieved", "unit": "bpm", "type": "continuous", "placeholder": "e.g., 140", "desc": "Maximum heart rate"},
    "exang": {"label": "Exercise Induced Angina", "type": "categorical", "options": {1: "Yes", 0: "No"}, "desc": "Chest pain during exercise?"},
    "oldpeak": {"label": "ST Depression (Oldpeak)", "unit": "mm", "type": "continuous", "placeholder": "e.g., 1.5", "desc": "ST depression induced by exercise"},
    "slope": {"label": "Slope of Peak Exercise ST Segment", "type": "categorical", "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}, "desc": "Slope of the ST segment"},
    "ca": {"label": "Number of Major Vessels", "type": "categorical", "options": {0: "0", 1: "1", 2: "2", 3: "3"}, "desc": "Vessels colored by fluoroscopy (0-3)"},
    "thal": {"label": "Thalassemia", "type": "categorical", "options": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}, "desc": "Thalassemia blood disorder type"},
}

# The ordered list of feature keys for model input (must match training order)
features_order = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# --- Dark Mode Dashboard HTML Template ---

form_html_dark_dashboard = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heart Disease Analysis Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {
      /* Dark Theme Variables */
      --bg-dark: #1e1e2f;
      --card-dark: #27293d;
      --text-light: #ffffff;
      --text-secondary: #a0a0a0;
      --accent-blue: #00bcd4; /* Cyan/Teal for highlights */
      --accent-green: #00e676; /* Green for Low Risk */
      --accent-red: #ff3d71; /* Pink/Red for High Risk */
      --border-dark: #3a3b50;
      --font-code: 'Roboto Mono', monospace;
      --font-display: 'Orbitron', sans-serif;
    }
    body {
      font-family: var(--font-code);
      background: var(--bg-dark);
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      color: var(--text-light);
      min-height: 100vh;
      padding: 30px 0;
    }
    .container {
      width: 95%;
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      font-family: var(--font-display);
      text-align: center;
      color: var(--accent-blue);
      font-weight: 700;
      margin-bottom: 40px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--border-dark);
      font-size: 1.8em;
    }

    /* --- Input Grid & Cards --- */
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr); /* Two columns for dense input */
      gap: 20px;
      margin-bottom: 30px;
    }
    .input-card {
      background: var(--card-dark);
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
      border-left: 3px solid var(--accent-blue);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    label {
      display: block;
      font-weight: 700;
      color: var(--accent-blue);
      margin-bottom: 2px;
      font-size: 0.9em;
      text-transform: uppercase;
    }
    .input-desc {
      display: block;
      font-size: 0.75em;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }
    .unit-text {
        color: var(--text-secondary);
        font-weight: 400;
    }
    input[type="number"], select {
      width: 100%;
      padding: 8px;
      border: 1px solid var(--border-dark);
      border-radius: 4px;
      background: var(--bg-dark);
      color: var(--text-light);
      font-family: var(--font-code);
      font-size: 0.9em;
      box-sizing: border-box;
      transition: border-color 0.3s;
    }
    input[type="number"]:focus, select:focus {
      border-color: var(--accent-blue);
      outline: none;
      box-shadow: 0 0 5px rgba(0, 188, 212, 0.5);
    }
    
    /* --- Action Section --- */
    .action-section {
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-dark);
      padding: 20px;
      border-radius: 8px;
      margin-top: 10px;
      border-top: 1px solid var(--border-dark);
    }

    input[type="submit"] {
      padding: 12px 30px;
      background-color: var(--accent-blue);
      color: var(--text-light);
      border: none;
      border-radius: 6px;
      font-size: 1.1em;
      font-weight: 700;
      cursor: pointer;
      transition: background-color 0.3s, transform 0.2s;
      box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3);
      font-family: var(--font-code);
      text-transform: uppercase;
    }
    input[type="submit"]:hover {
      background-color: #008fa3;
      transform: translateY(-1px);
    }

    /* --- Result Panel (Mimicking a Dashboard Block) --- */
    .result-panel {
      width: 100%;
      max-width: 350px;
      padding: 15px 25px;
      border-radius: 8px;
      text-align: center;
      font-family: var(--font-display);
      box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
      
    }
    .result-low {
      background-color: #1a362a; /* Darker green background */
      color: var(--accent-green);
      border: 1px solid var(--accent-green);
    }
    .result-high {
      background-color: #3b1b24; /* Darker red background */
      color: var(--accent-red);
      border: 1px solid var(--accent-red);
    }
    .result-label {
      font-size: 0.9em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .result-value {
      font-size: 1.8em;
      font-weight: 700;
      letter-spacing: 2px;
    }
    .disclaimer {
        font-size: 0.7em;
        font-weight: 400;
        color: var(--text-secondary);
        margin-top: 10px;
    }

    /* Responsive adjustments */
    @media (max-width: 800px) {
      .dashboard-grid {
        grid-template-columns: 1fr; /* Single column on small screens */
      }
      .action-section {
        flex-direction: column;
        align-items: stretch;
      }
      .result-panel {
        max-width: 100%;
        margin-bottom: 20px;
      }
      input[type="submit"] {
        width: 100%;
        margin-top: 15px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1><span style="color: var(--accent-blue);">🔬</span> CARDIAC RISK CLASSIFICATION PROTOCOL</h1>
    
    <form method="POST">
      <div class="dashboard-grid">
      {% for key, config in feature_config.items() %}
        <div class="input-card">
          <div>
            <label for="{{ key }}">{{ config['label'] }} 
              {% if config.get('unit') %}
                  <span class="unit-text">({{ config['unit'] }})</span>
              {% endif %}
            </label>
            <span class="input-desc">{{ config['desc'] }}</span>
          </div>

          {% if config['type'] == 'continuous' %}
            <input type="number" step="any" name="{{ key }}" id="{{ key }}" placeholder="{{ config['placeholder'] }}" required>
          {% elif config['type'] == 'categorical' %}
            <select name="{{ key }}" id="{{ key }}" required>
              {% for value, display_name in config['options'].items() %}
                <option value="{{ value }}">{{ display_name }}</option>
              {% endfor %}
            </select>
          {% endif %}
        </div>
      {% endfor %}
      
      <div class="action-section">
        {% if prediction is not none %}
          <div class="result-panel {% if prediction == 'High Risk' %}result-high{% else %}result-low{% endif %}">
            <div class="result-label">FINAL CLASSIFICATION</div>
            <div class="result-value">{{ prediction }}</div>
            <p class="disclaimer">Prediction executed at {{ now }}</p>
          </div>
        {% endif %}
        <input type="submit" value="EXECUTE RISK ANALYSIS">
      </div>
      
      </div>
    </form>
    
    {% if prediction is not none %}
        <div style="grid-column: 1 / -1; margin-top: 20px; text-align: center;">
            <p class="disclaimer" style="font-size: 0.8em; margin: 0; padding: 10px 0;">
                Disclaimer: This automated classification is for informational purposes only and does not substitute professional medical diagnosis.
            </p>
        </div>
    {% endif %}

  </div>
  <script>
    // Get the current time for the result panel display
    document.addEventListener('DOMContentLoaded', () => {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const dateString = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        const displayTime = dateString + ' ' + timeString;
        
        const disclaimerElement = document.querySelector('.result-panel .disclaimer');
        if (disclaimerElement) {
             disclaimerElement.innerHTML = 'Prediction executed at ' + displayTime;
        }
    });
  </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if request.method == "POST":
        try:
            # 1. Gather all feature values from the POST request
            input_data = [float(request.form[feat]) for feat in features_order]
            input_array = np.array(input_data).reshape(1, -1)

            # 2. Make prediction
            model_prediction = model.predict(input_array)[0]

            # 3. Format result
            if model_prediction == 1:
                prediction = "HIGH RISK"
            elif model_prediction == 0:
                prediction = "LOW RISK"
            else:
                prediction = "ERROR"

        except Exception as e:
            prediction = f"PROCESSING ERROR"
            print(f"Error during prediction: {e}")

    # Render the dark dashboard template
    return render_template_string(
        form_html_dark_dashboard,
        feature_config=feature_config,
        prediction=prediction,
        now=now_str
    )

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    