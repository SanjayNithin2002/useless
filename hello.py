from flask import Flask
from model import recommend_songs
import pandas as pd
from flask import request, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/login')
def hello():
    name = request.args.get('name')
    year = int(request.args.get('year'))
    name = name.replace("-"," ").title()
    spotify_data = pd.read_csv('datasetsong.csv')
    data = recommend_songs([{'track_name':name, 'year': year}], spotify_data)
    return (jsonify(data))

if __name__=="__main__":
    app.run(debug=True)
