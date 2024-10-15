from flask import Flask
from controllers.email_controller import home, predict_emails

app = Flask(__name__, template_folder='views')

app.add_url_rule('/', 'home', home)
app.add_url_rule('/predict', 'predict_emails', predict_emails, methods=['GET'])

if __name__ == '__main__':
    app.run(debug=True)
