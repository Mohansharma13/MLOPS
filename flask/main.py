from flask import Flask

app=Flask(__name__)

@app.route('/')

def hello():
    return "<h1>welcome to the world of flask </h1>"

@app.route('/welcome')
def welcome():
    return "<h1> welcome to the home page </h1>"

if __name__ == "__main__":
    app.run()