from movie import run,get_merged_dataframe
from flask import Flask, render_template,request

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    movie_list = []
    error = ""
    if request.method == "POST":
        movie = request.form.get("movie")
        movie_list = run(movie)
        if movie_list.empty:
            error = "Couldn't find a movie with the name "+movie


    return render_template("index.html",movie_list= movie_list,error = error)

@app.route("/list",methods=["POST"])
def list():
    df = get_merged_dataframe()
    return render_template("index.html",movie_list=df['tittle'])

if __name__ == "__main__":
    app.run(port=5000,host='0.0.0.0',debug=True)
