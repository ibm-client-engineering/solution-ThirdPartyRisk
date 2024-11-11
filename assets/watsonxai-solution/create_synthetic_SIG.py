from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd
import json, time, os
import random

from dotenv import load_dotenv
load_dotenv()

golden_df = pd.read_csv("../data/golden_sig.csv")
issue_rate = .05
gap_rate = .025

#-----------------Functions--------------------#
#Get credentials for LLM call
def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.environ['API_KEY']
	}

def get_justification(question, answer) :
    model_id = os.environ['GRANITE_MODEL_ID']

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 400,
        "repetition_penalty": 1,
        "stop_sequences": ["}"]
    }

    project_id = os.environ['PROJECT_ID']

    model = ModelInference(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
        project_id = project_id
    )

    prompt_temp = """<|system|>
        I will give you a question from IBM BANK's standardized vendor assessment form, along with a corresponding answer to the question as a yes or a no. This form is used to assess the risk associated with a third party vendor's product. For this task, you are assuming the role of the vendor who is filling out the form.
        Your task is to come up with a short supplementary response to the yes or no answer given that justifies why that answer was given, even if it would have originally been a problem for IBM BANK. Please limit your response to at most 2 sentences, and output in a json format ( {{"supplementary_response": <YOUR RESPONSE> }} ).

        Here is an example:
        Question: Do all IT personnel (e.g., centralized, decentralized, shadow IT) receive the same security awareness training?
        Answer: No
        {{"supplementary_response": "While not all IT personnel receive the same training, we have a robust onboarding process for new hires that varies based on the employee's exposure to critical IT systems. We also provide regular refresher courses for existing staff to ensure they are up-to-date with our security protocols."}}

        <|user|>
        Question: {}
        Answer: {}

        <|assistant|>"""
    
    response = model.generate_text(prompt=prompt_temp.format(question, answer))
    try: 
        additional_comments = json.loads(response)["supplementary_response"]
        return additional_comments
    except:
        print("Failed to parse additional comments")
        return ""

def get_nonjustification(question, answer) :
    model_id = os.environ['GRANITE_MODEL_ID']

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 400,
        "repetition_penalty": 1,
        "stop_sequences": ["}"]
    }

    project_id = os.environ['PROJECT_ID']

    model = ModelInference(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
        project_id = project_id
    )

    prompt_temp = """<|system|>
        I will give you a question from IBM BANK's standardized vendor assessment form, along with a corresponding answer to the question as a yes or a no. This form is used to assess the risk associated with a third party vendor's product. For this task, you are assuming the role of the vendor who is filling out the form.
        Your task is to come up with a short supplementary response from the perspective of the vendor to the yes or no answer given that confirms that the answer is a problem for IBM BANK. The response should be an attempt to justify the answer that would fail to assuage the concerns of the bank. Please limit your response to at most 2 sentences, and output in a json format ( {{"supplementary_response": <YOUR RESPONSE> }} ).

        Here is an example:
        Question: Do all IT personnel (e.g., centralized, decentralized, shadow IT) receive the same security awareness training?
        Answer: No
        {{"supplementary_response": "While not all IT personnel receive security awareness training, we have a robust training program in place for our centralized IT team, which handles the majority of our systems and data."}}

        <|user|>
        Question: {}
        Answer: {}

        <|assistant|>"""
    
    response = model.generate_text(prompt=prompt_temp.format(question, answer))
    try: 
        additional_comments = json.loads(response)["supplementary_response"]
        return additional_comments
    except:
        print("Failed to parse additional comments")
        return ""


def create_synthetic_sig():

    df = pd.DataFrame(columns=['SIG Question', 'Response', 'Additional Information'])
    reversal_dict = {"Yes": "No", "No": "Yes"}
    for index, row in golden_df.iterrows() :
        randfloat = random.random()
        if randfloat > issue_rate : # going to be a non-problematic answer (might still have a bad justification)
            if randfloat > (1 - gap_rate/2) : #im doing it this way to ensure that ~gap_rate of all responses have additional comments
                #we wanna give it some additional information
                add_i = get_nonjustification(row["SIG Question"], row["Response"])
                new_row = {"SIG Question": row["SIG Question"], "Response": row["Response"], "Additional Information": add_i}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else :
                new_row = {"SIG Question": row["SIG Question"], "Response": row["Response"], "Additional Information": ""}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else :
            if randfloat < gap_rate/2 :
                add_i = get_justification(row["SIG Question"], row["Response"])
                new_row = {"SIG Question": row["SIG Question"], "Response": reversal_dict[row["Response"]], "Additional Information": add_i}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else :
                new_row = {"SIG Question": row["SIG Question"], "Response": reversal_dict[row["Response"]], "Additional Information": ""}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    
    # Add to new reference SIG csv
    df.to_csv('synthetic_SIG.csv', index=False)
    
    return df


create_synthetic_sig()

