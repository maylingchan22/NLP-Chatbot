from flask import Flask, render_template, request
from chat import ChatBot

app = Flask(__name__)
bot = ChatBot()

@app.route('/')
def main():
    return render_template('base.html')

@app.route("/get", methods=['GET', 'POST'])
def chatbot():
    userText = request.form.get('response_text')
    botresponse = bot.response(str(userText))
    return render_template("base.html", output=botresponse)

if __name__ == '__main__':
    app.run(debug=True)