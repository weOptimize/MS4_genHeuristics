import os
import openai
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview" 
openai.api_base = "https://gpt-4-uks.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value .
openai.api_key = "355cb31066ff4fc982c0187ae23603a8"

# Load text from strategic plan as plain code so that I can concatenate it at the prompt following the conversation
# with the user
strategic_plan = open("EU_LIFE_Valid_Statement.txt", "r")

# concatenate an initial instruction with the strategic plan in the same string
initialization_prompt = "You are a reliable and ojective portfolio manager, who only writes statements that \
can be reasonably justified based only on the information you have available. You are provided with a proposal of a portfolio \
consisting of several project for which you will be provided with an executive summary. You will compare the project respect to \
a strategic plan I am providing at this message. Your evaluation should be on a scale from 1 to 100, with 100 being the maximum alignment.\
provide first the rating and then the justification for the rating, detailing how well the combination of projects aligns with the\
strategic plan and the key factors that influenced your assessment. The strategic plan is the following:" + strategic_plan.read()


# The conversation is initialized with a message to the user, which is the first message in the conversation list
# First there is an instruction under "content", and then - also inside "content" - the strategic plan is concatenated
conversation=[{"role": "system", "content": initialization_prompt}]



# The conversation list is a list of dictionaries, where each dictionary has a "role" key and a "content" key. The "role" key
# is a string that can be either "user" or "assistant", and the "content" key is a string that contains the message.
# The conversation list is passed to the ChatCompletion.create method, which returns a response that is then appended to the
# conversation list. The response is then printed to the console.
while(True):
    user_input = input()
    conversation.append({"role": "user", "content": user_input,})
    try:
        response = openai.ChatCompletion.create(
            engine="GPT4_turbo_128k", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages = conversation,
            temperature=0.1,
            max_tokens=1250,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    except Exception as e:
        print("An error occurred: ", e)

    # response = openai.Completion.create(
    #     engine="weO_vs00_gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    #     messages = conversation
    # )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")