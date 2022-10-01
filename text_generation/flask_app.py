from transformers import pipeline
from flask import Flask, render_template, request, jsonify


print(f'{"*"*5}initilizing EleutherAI/gpt-neo-1.3B (GPT3based model){"*"*5}')
generator_EleutherAI = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

print(f'{"*"*5}initilizing openai-gpt model{"*"*5}')

generator_Open_ai = pipeline('text-generation', model='openai-gpt')

def get_model_gen(model, text):
    result = model(text, do_sample=True, min_length=150, max_length = 200)
    result = result[0]['generated_text'].replace("\n", ' ')
    return result



# building flask app
app = Flask(__name__)#__name__
@app.route('/textgen', methods=["GET", "POST"])
def textgen_api():

    #retrieving model type as path parameter
    model_type = request.args.get('model')
    print(f'{"*"*5}using : {model_type}{"*"*5}\n')

    # getting input with name = text_input in HTML form
    text = request.args.get("text_input")
    print(f'{"*"*5}given text : {text}{"*"*5}')

    #checking model type
    if model_type == 'EleutherAI':
        result = get_model_gen(generator_EleutherAI,text)

    elif model_type == 'OpenAI':
        result = get_model_gen(generator_Open_ai,text)

    return jsonify(result)



# if __name__=='__main__':
#    app.run()#host='0.0.0.0'


# http://127.0.0.1:5000/textgen?model=EleutherAI&text_input=Hyderabad%20is%20the%20one%20of%20the%20beautiful%20city%20and%20
# http://127.0.0.1:5000/textgen?model=OpenAI&text_input=Hyderabad%20is%20the%20one%20of%20the%20beautiful%20city%20and%20
