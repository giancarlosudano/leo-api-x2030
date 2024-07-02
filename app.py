import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import helpers.langchain_helper as lc_hlp
import helpers.utility_helper as utl_hlp

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/ChatCompletion', methods=['POST'])
@cross_origin()
def chat_completion():
    question = request.form.get('question')

    llm = lc_hlp.get_gpt()

    input_variables = ["question"]
    prompt_text = utl_hlp.read_file(os.path.join("prompt","prompt.txt"))
    prompt_template = PromptTemplate(template=prompt_text, input_variables=input_variables)
    chain = prompt_template | llm | StrOutputParser()				
    generation = chain.invoke({"question": question})

    response = {
        'question': question,
        'answer': generation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Quali sono gli impianti a distanza superiore ai 300 metri? 

# Quali sono le zone di danno e a che metratura si trovano? Che azioni vanno intraprese per ciascuna zona? 

# Quali sono gli insediamenti produttivi limitrofi all'impianto? 

# Quanti sono i dipendenti dell'impianto? 

# Che tipi di danni potrebbero verificarsi sulla popolazione e quindi vanno controllati? 

# Quali sono le misure di autotutela per le persone presenti? 