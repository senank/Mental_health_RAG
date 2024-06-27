PROMPT_TEMPLATE = """\
This is the context that you should exclusively use when constructing your answers, which comes in\
the form of <question : answer> with seperation between each question answer pair ([SEP]):
"{context}"

Using the context provided above, please respond to the following question: "{query}". If \
there is nothing relevant to the question in the context, please respone with "I am sorry, \
I am not qualified to help with this specific topic" and refer them to seek professional help.

"""

def context_from_data(db_data):
    data = []
    for d in db_data:
        data.append("{} : {}".format(d['question'], d['answer']))
    context= " \n\n [SEP] \n\n ".join(data)
    return context
