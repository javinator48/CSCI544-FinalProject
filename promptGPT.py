import openai
import os
import re
import argparse
import pandas as pd
import pickle
import json
import time
from tqdm.auto import tqdm
from openai import OpenAI

api_key='your_api_key_here'
client = OpenAI(api_key=api_key)
# openai.api_key = 'your_api_key_here'



def request(
        request_prompt,
        request_temperature=1,
        request_top_p=1,
        model="gpt-3.5-turbo"):
    """
    Make an request to openai ChatCompletion API to request the Language model with a prompt.

    ### Parameters:
    For detailed descriptions of each parameter, please refer to the [official documentation](https://platform.openai.com/docs/models).
    1. model (string): Model to use for prompting. Selection can be made from the available models\
      ""gpt-4", and "gpt-3.5-turbo".

    2. request_temperature (number): Sampling temperature used. Value ranges between 0 and 2. Higher values makes output more\
        random and lower values makes output more focused and deterministic.\
     For more details, see the API documentation [here](https://platform.openai.com/docs/api-reference/chat/create).

    3. request_top_p (number): Alternative to sampling with temperature, called nucleus sampling.Values ranges between 0 and 1. \
     For more details, see the API documentation [here](https://platform.openai.com/docs/api-reference/chat/create).

    ### Returns:
    - Output of the model in the format of a dictionary with keys id, object, created, model, usage, and choices.

    """
    # print("Requesting model with temperature: ", str(request_temperature))
    response = client.chat.completions.create(
        model=model,
        temperature=request_temperature,
        top_p=request_top_p,
        messages=request_prompt


    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def split_string(output):
    items = re.split('d+\\.\\s*', output)
    items = [item for item in items if item]
    return items


def callChatGPT(item, temperature=1.0):
    """
    Takes input item and makes a call to chatgpt asking it to summarize papers. We ask chatgpt to
    give the following information in the summarizations: Key Takeaways, Key Points, Importance:,  Uniqueness,
                    Model/Method Proposed.


    ### Parameters:
    - item (str): A string containing the title, abstract, and conclusion of the paper.
    ### Returns:
    - output (str): Choices[0] of the output dictionary of the response from the openAI module.

    """

    systemPrompt = {
        "role": "system", "content": """I want you to act as a natural and no-bias ner tagger, label the words in the sentences with the following ner tags: 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'. If a word does not have an tag label it with 'O'. Please expect input sentences from various languages and only output the tags in a list format where each element is a tag.  """}
    userPrompt = {"role": "user", "content": item}
    prompt = [systemPrompt, userPrompt]
    promptReAttempts = 0

    while True:
        stop_execution = False
        # stop chatgpt if we continually fail more than 10 times
        # print("Attempt number", promptReAttempts)
        if promptReAttempts > 10:
            stop_execution = True
            break
        try:
            # print("requesting chatgpt")
            output = request(request_prompt=prompt, request_temperature=1)
            # print('finished generating result, parsing the output ')

            stop_execution = True
        except Exception as e:
            print("Error Type:", e)
            print("Retrying in 10 seconds...")
            promptReAttempts += 1
            time.sleep(10)
        if stop_execution:

            break
    return output

def generateTags(csvInputPath,csvOutputPath, numberOfSamples): 
    """
    Calls the chatgpt on the dataset to generate the tags and then saves the tags to the dataset 
    """
    count = 0 
    words = []
    tags = []
    labels = []
    df = pd.read_csv(csvInputPath)
    for _, row in tqdm(df.head(numberOfSamples).iterrows(), total = numberOfSamples): 
        
        tokens = row['tokens']
        
        tags.append(callChatGPT(tokens)) 
        words.append(str(row['tokens']))
        
        labels.append(row['tags'])
    outputDf = pd.DataFrame({'tags': words, 'predicted_tags': tags,'ground_truth':labels })
    outputDf.to_csv(csvOutputPath, index=False)



if __name__ == "__main__":
    generateTags('/home/javin/Coding/CSCI544/FinalProject/Data/wikiann_english_train.csv','/home/javin/Coding/CSCI544/FinalProject/Data/wikiann_english_train_chatgpt.csv', 100)
