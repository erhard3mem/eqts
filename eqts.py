import numpy as np
from datetime import datetime, timedelta
import math, re
import numpy as np
from mpmath import mp
import numexpr


def string_to_function(eq_str, variables):
    # List of math functions we want to replace
    unary_operators = ["cos", "exp", "sin", "log", "sqrt", "abs", "tanh"]

    # Create regex: matches only allowed functions
    funcs_pattern = r"\b(" + "|".join(unary_operators) + r")\(([^)]+)\)"
    
    def repl(match):
        func = match.group(1)
        args = match.group(2)
        return f"math.{func}({args})"

    # First, handle normal math functions
    s1 = re.sub(funcs_pattern, repl, eq_str)

    # Special handling for inv(x) = 1/x
    s1 = re.sub(r"\binv\(([^)]+)\)", r"(1/(\1))", s1)

    # Debug print
    print("After replacement:", s1)

    # Prepare variables dict
    vars_dict = {var: None for var in variables}

    # Replace ^ with Python's ** and evaluate
    return eval(s1.replace("^", "**"), vars_dict, {"math": math})

def getEQ(input):
    
    X = list(map(int,input.split(' ')))
    y = list(range(1,len(X)+1))

    print(X,y)

    X = np.array(X).reshape(-1, 1)  # Umwandlung in 2D-Array mit Form (5, 1)
    y = np.array(y).reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = StandardScaler()  # Oder MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    from pysr import PySRRegressor

    model = PySRRegressor(
        procs = 1,
        maxsize=20,
        niterations=500,  # < Increase me for better results
        binary_operators=["+", "-", "*", "/"],   
        unary_operators=["cos", "exp", "sin", "log", "sqrt", "abs", "tanh","inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        constraints={"^": (9, 1)}
    )

    model.fit(y_scaled,X_scaled)  

    equation = str(model.get_best()["equation"])
    return equation
    

def julia(eq):
    return numexpr.evaluate(eq)


from flask import Flask, request, render_template_string

app = Flask(__name__)

# HTML template for the input form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>equation</title>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #74ebd5, #9face6);
            margin: 0;
            margin-top:auto;
            margin-bottom:auto;
            padding: 0;
            display: block;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }
        h3 {
            color: #ffffff;
            font-weight: 500;
            text-align: center;
            margin-bottom: 20px;
        }
        form {
            background: #fff;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
            box-sizing: border-box;
            margin-right:auto;
            margin-left:auto;
        }
        textarea {
            width: 100%;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            resize: none;
            box-sizing: border-box;
        }
        button {
            display: block;
            width: 100%;
            background: #4caf50;
            color: #fff;
            font-size: 16px;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.3s;
        }
        button:hover {
            background: #45a049;
        }
        h2 {
            text-align: center;
            margin-top: 30px;
            color: #ffffff;
            font-weight: bold;
        }
    </style>
</head>
<body>
    
    <div id="pipe" style="width:400px;margin-top:30px;margin-left:auto;margin-right:auto;">
        <div>This software uses symbolic regression to find a mathematical equation for a given time series.</div>
        <div><a href="https://github.com/erhard3mem/eqts">source code</a></div>
            
        <div>Example: "1 2 3 -1" - be aware not to end with a whitespace</div><br />
  
        <form style="display:block" id="aios" action="/" method="POST">
              <div>Time series input</div>  
              <textarea id="input" name="input" rows="5" cols="100" required></textarea><br><br>            
              <div>Prediction index</div>
              <textarea id="input" name="input2" rows="1" cols="10" required></textarea><br><br>                                                 
            <button type="submit">submit</button>
        </form>  
                    
    </div>
   <div style="width:800px;margin-top:30px;margin-left:auto;margin-right:auto;">
        {% if response %}
             {% for item in response %}
                <div style="border:1px solid black; padding:5px; margin:5px">{{ item }}</div><br />
            {% endfor %}
        </ul>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    result = ""
    results = []
    if request.method == "POST":               
        input = request.form.get("input")  
        input2 = request.form.get("input2")  
        eq = getEQ(input)
        eqc = eq.replace("x0", input2)
        result = str(julia(eqc)) 
        results.append(input) 
        results.append(eq)         
        results.append(eqc)         
        results.append(result)
    return render_template_string(HTML_TEMPLATE, response=results)

if __name__ == "__main__":
    app.run(debug=True)
