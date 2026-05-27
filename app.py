from flask import Flask, render_template, request
import joblib

#cria aplicação
app = Flask(__name__)

#carrega IA
model = joblib.load("model.pkl")

#cria rota principal que define endereço e métodos permitidos
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None

    #recebe o formulário
    if request.method == "POST":
        sintomas = request.form["sintomas"]

        #IA faz previsão
        categoria = model.predict([sintomas])[0]
        resultado = categoria

    return render_template(
        index.html,
        resultado=resultado
    )

#rodar servidor
if __name__ == "__main__":
    app.run(debug=True)