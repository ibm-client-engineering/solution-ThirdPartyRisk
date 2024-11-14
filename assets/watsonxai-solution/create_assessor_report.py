import pandas as pd
import os, openpyxl, re
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
import json
import time
from create_reference_SIG import process_question
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

GOLDEN_SIG_PATH = '/Users/madisonlee/Documents/GitHub/solution-ThirdPartyRisk/assets/data/golden_sig.csv'
gs_df = pd.read_csv(GOLDEN_SIG_PATH)

# ---------------------------------------------------------------------------------------------------------------------------------

# def get_credentials(): get credentials for LLM call
def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.environ['API_KEY']
	}

# def if_apropriate_response(): prompts the LLM to reason whether or not the reference SIG generated an apropriate response
def if_apropriate_response(input_ques, resp, info):
      return (""" <|begin_of_text|><|start_header_id|>user<|end_header_id|>
{INSTRUCTION}
You are provided with an input SIG question, vendor response in either "Yes" or "No" format, and additional information provided by the vendor to support their response. Based on your analysis, categorize the output into one of two categories:
      1. "provided, contradicts response" in case additional information is provided but it does not support the response given by vendor.
      2. "provided, supports response" in case additional information is provided and it supports the response given by vendor.
- Be very exact in generating output.
- Do not generate any reasoning in the output.\n\n""" + 

"#### START OF QUESTION ####\n\n" + 
"[input SIG Question: ]" +  f"{input_ques}\n\n" + 

"[vendor response: ]" + f"{resp}\n\n" + 

"[additional info: ]" + f"{info}\n\n" + 
"#### END OF QUESTION ####  <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + 
"Output: ")

# def generate_gap(): sends if_apropriate_response() to LLM to determine if the Additional Information provided supports the response provided in the SIG
def generate_gap(question, response, additional_info):

    project_id = os.getenv("PROJECT_ID")

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 50,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.REPETITION_PENALTY: 1.12,
    }

    model = ModelInference(
        model_id = "ibm/granite-13b-chat-v2",
        params=parameters, 
        credentials=get_credentials(),
        project_id=project_id
        )

    granite_prompt = if_apropriate_response(question, response, additional_info)
    response = model.generate_text(prompt = granite_prompt)

    return response

# UPDATED FUNCTION to work with csv 
def get_sig_df_from_csv(csv_path):
    # Load the CSV file
    sig_df = pd.read_csv(csv_path)

    # Initialize a new DataFrame with the same structure as `new_df` from `get_sig_df`
    new_df = pd.DataFrame(columns=["Ques Num", "SIG Question Text", "Response", "Additional Information", "Category"])

    # Iterate over the rows of the CSV
    for index, row in sig_df.iterrows():
        # Assuming "Ques Num" is created based on the index here, as it was in the original function
        ques_num = index + 1  # or another identifier if available in your CSV
        
        # Check if "Response" is either "Yes" or "No", otherwise set to "N/A"
        response = row["Response"] if row["Response"] in ["Yes", "No"] else "N/A"
        
        # Add row to the new DataFrame
        new_df.loc[-1] = [
            ques_num,                        # Question number or unique identifier
            row["SIG Question"],             # SIG Question Text
            response,                        # Response ("Yes", "No", or "N/A")
            row["Additional Information"],   # Additional Information
            None                             # Category (left blank if not in CSV)
        ]
        # Adjust the index to maintain order and sort after adding each row
        new_df.index += 1
        new_df = new_df.sort_index()

    return new_df

# def get_sig_df(): create the sig_df dataframe by going through the individual category sheets
def get_sig_df(sig_path):
    # open an Excel workbook in openpyxl and pandas
    workbook = openpyxl.load_workbook(sig_path)
    sig_df = pd.read_excel(sig_path, sheet_name=None)

    pattern = re.compile(r"[A-Z]\..*")
    new_df = pd.DataFrame()
    new_df['Ques Num'] = pd.Series(dtype='string')
    new_df['SIG Question Text'] = pd.Series(dtype='string')
    new_df['Response'] = pd.Series(dtype='string')
    new_df['Additional Information'] = pd.Series(dtype='string')
    new_df['Category'] = pd.Series(dtype='string')
    for key in sig_df.keys():
        if pattern.match(key) :
            sig_df[key].drop(sig_df[key].index[0], inplace=True)
            sig_df[key].drop(sig_df[key].index[0], inplace=True)
            sig_df[key].columns = sig_df[key].iloc[0]
            sig_df[key].drop(sig_df[key].index[0], inplace=True)

            # find hidden rows
            worksheet = workbook[key]
            # list of indices corresponding to all hidden rows
            hidden_rows_idx = [
                row - 2 for row, dimension in worksheet.row_dimensions.items() if dimension.hidden
            ]

            for index, row in sig_df[key].iterrows() :
                if index in hidden_rows_idx :
                    #print(key, index, row["Response"])
                    continue
                # if row["Response"] not in ['Yes', 'No'] :
                #     print(row["Ques Num"], row["Response"])
                new_df.loc[-1] = [row["Ques Num"], row["Question/Request"], row["Response"] if row["Response"] in ['Yes', 'No'] else "N/A", row["Additional Information"], row["Category"]]
                new_df.index += 1
                new_df = new_df.sort_index()
    return new_df

# def generate_recommendation(): leverage LLM to identify the gaps between the Additional Infomation provided and the Standard context
def generate_recommendation(question, std_context, additional_info):
    """
    Takes in 1 SIG question + its corresponding Standard Requirements context and Additional Information text
    Asks the model to identify if (1) Additional Info answers the SIG question and (2) Additional Info relates to the Standard Requirements context
    :param question: SIG question from filled-out SIG file
    :param std_context: Standard Requirements Context from filled-out golden_SIG file
    :param additional_info: additional info column from filled-out SIG file
    :return: status (Yes, Partially Relevant, No) and explanation (if applicable)
    """
    
    model_id = os.environ['MISTRAL_MODEL_ID']

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 3000,
        "repetition_penalty": 1,
        "stop_sequence": "}"
    }
    question = question if type(question) == str else "No question provided"
    std_context = std_context if type(std_context) == str else "No Standard Requirements context provided"
    additional_info = additional_info if type(additional_info) == str else "No additional info provided"


    model = ModelInference(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
        project_id = os.environ["PROJECT_ID"]
    )


    instruction = """You are a IBM BANK security analyst. I will give you three items, a Standardized Information Gathering (SIG) question, a Standard Reference text, and an Additional Information text. Determine if the Additional Information text answers the SIG question and/or if the Additional Information text supports the Standard Reference text. Output should be either Yes, No or Partially relevant. If applicable, please provide an explanation in detail and any follow-up questions that you'd like to ask the vendor. 

    For the explanation, explain in detail how you got to the conclusion of the status, be specific if the content was gathered from additional information to back up the selected status, and why you conclude with the selected status. 

    For the follow-up questions, you are expected to generate more than one follow-up question until you think it is enough to gather the answer we need to answer the SIG's question. Lastly, make sure the follow-up questions are in a numbered list. 

    Please answer in the following manner and in json output:
    {"status": "Yes", "explanation and follow-up questions":"<if any explanation> <if any follow-up questions> "}
    {"status": "Partially relevant", "explanation and follow-up questions":"<if any explanation> <if any follow-up questions> "}
    {"status": "No", "explanation and follow-up questions":"<if any explanation> <if any follow-up questions> "}


    Below is the status definition: 
    status: Yes, the Additional Information text addresses the SIG question and is in alignment with the Standard Requirement policies.
    status: Partially relevant, either the Additional Information text addresses the SIG question or the Additional Information text is in alignment with the Standard Requirement policies. 
    status: No, the Additional Information text does not addresses the SIG question and is not in alignment with the Standard Requirement policies.

    Question:
    {{Question}}

    Standard Reference Text:
    {{StandardContext}}

    Vendor document:
    {{AdditionalInfo}}

    Result:"""

    examples = """[
    {
        "input": "**DISCLAIMER: Follow examples for structure of what the output should look like**\n\nquestion = question content \n\nStandardContext = Standard content \n\nAdditionalInfo = additional vendor info content ",
        "output": "{\"status\": \"status answer\", \"explanation and follow-up questions\":\"Explanation content.\n\nFollow-up questions: \n1. Question 1 \n2. Question 2\n3. Question 3\"}"
    }
    ]"""

    input = """question = """+question+"""

    """+std_context+"""

    """+additional_info+"""
    """

    input_prefix = "Input:"
    output_prefix = "Output:"

    prompt_input = instruction + examples + input + input_prefix + output_prefix

    #print("Submitting generation request...")
    data = model.generate_text(prompt=prompt_input)
    print("this is the DATA "+data)
    try: 
        json_data = json.loads(data, strict=False)
    except :
        return "Failed", "LLM generation failed"
    status = json_data["status"]
    explanation = json_data["explanation and follow-up questions"]

    print("Raw JSON Data:", json_data)  # Print raw data for debugging

    return status, explanation

# def format_rec() to format recommendation input
def format_rec(issue_num, issue_rec, explanation=None) :
    body = f"""Issue Recommendation:
    Issue Number: {issue_num if issue_num != "" else "n/a"},
    Issue Recommendation: {issue_rec if issue_rec != "" else "n/a"}
    """

    if explanation is not None :
        body = f"Explanation: {explanation}\n\n" + body
    return body

# def compare_to_reference(): for each question in sig_df, compare against the reference sig: 
    # 1. Compare the SIG Response to the Reference Response
    # 2. Check if "Additional Information" is provided
    # 3. IF "Additional Information" is provided then "generate_gap"
def compare_to_reference(sig_df):
    """
    Takes in a vendor SIG (in dataframe form) and outputs three lists of booleans
    goes thru each q, what'll pass into generate_recommendation will be 1-1 std_context + additional_info
    :param sig_df: The vendor SIG as a pandas dataframe
    :return: three lists of booleans, is the answer the same as the reference, is additional information provided, and does the additional information support the response
    """
    global gs_df
    num_rows, _ = sig_df.shape
    answer_same, ai_provided, ai_support = [''] * (num_rows + 1), [''] * (num_rows + 1), [''] * (num_rows + 1)
    for index, row in sig_df.iterrows() :
        ai_provided[index] = type(row["Additional Information"]) == str
        try:
            context = gs_df.loc[gs_df['SIG Question'] == row["SIG Question Text"]].iloc[0]["Standards Context"]
            sig_df.loc[index, "Standards Context"] = context
        except:
            #This question is not in the golden SIG, so add it
            reference = process_question(row['SIG Question Text'], row["Ques Num"])
            gs_df = pd.concat([gs_df, pd.DataFrame([reference])], ignore_index=True)
            sig_df.loc[index, "Standards Context"] = reference['Standards Context']

        ref_answer = gs_df.loc[gs_df['SIG Question'] == row["SIG Question Text"]].iloc[0]["Response"]
        answer_same[index] = row["Response"] == ref_answer
        if ai_provided[index] :
            ai_support[index] = generate_gap(row['SIG Question Text'], row['Response'], row['Additional Information'])
        #ai_support[index] = generate_gap(row) #ignore for now

    return answer_same, ai_provided, ai_support

# def create_assessor_report(): assembles the assessor report by calling all above funcs
def create_assessor_report(sig_path):
    """
    Reads in a path to vendor SIG and outputs the watsonx assessor report to a CSV called "assessor_report.csv"
    :param sig_path: The path to the vendor SIG
    """
    global gs_df
    start = time.time()
    #read the sheet and clean it up
    sig_df = get_sig_df_from_csv(sig_path)#.iloc[:100, :]

    #Add the 4 new columns
    sig_df['Issue'] = pd.Series(dtype='string')
    sig_df['Gap'] = pd.Series(dtype='string')
    sig_df['Standards Context'] = pd.Series(dtype='string')
    sig_df['Recommendation'] = pd.Series(dtype='string')

    #fill in the Standard context (probably will come from reference sig, TBD on that front)
    # for index, row in sig_df.iterrows() :
    #     sig_df.loc[index, "Standard Context"] = get_Standard_Context(row["SIG Question Text"])

    #Get three lists of booleans, one for if the answer is the same as the reference, another for if there is additional information, 
    #and a third or if that additional information supports what the respondent says
    answer_same, ai_provided, ai_support = compare_to_reference(sig_df)
    gs_df.to_csv(GOLDEN_SIG_PATH)

    #Logic time
    for index, row in sig_df.iterrows() :
        if answer_same[index] : #if we're alright
            sig_df.loc[index, "Issue"] = 'No'
            sig_df.loc[index, "Gap"] = 'No'
            sig_df.loc[index, "Recommendation"] = ''
        elif row['Response'] not in ['n/a', 'N/A', 'N/a'] : #if the answer is not N/A, then we need to do an issue analysis
            ref = gs_df.loc[gs_df['SIG Question'] == row["SIG Question Text"]].iloc[0]
            issue_num, issue_rec = "", ""
            # print(ref.columns)
            # if type(ref["Issue Description"]) == str and len(ref["Issue Description"]) > 0 : #there is an associated issue
            #     issue_num = ref[" Ques\n #"]
            #     issue_rec = ref["Recommendation "]
            if ai_provided[index] :
                _, explanation = generate_recommendation(row['SIG Question Text'], row['Standards Context'], row['Additional Information'])
                print("It's an issue!")
                sig_df.loc[index, "Issue"] = 'Yes' if "contradicts" in ai_support[index] else 'No'
                sig_df.loc[index, "Gap"] = 'Yes' if "contradicts" not in ai_support[index] else 'No'
                sig_df.loc[index, "Recommendation"] = format_rec(issue_num, issue_rec, explanation)
            else :
                sig_df.loc[index, "Issue"] = 'Yes'
                sig_df.loc[index, "Gap"] = 'No'
                sig_df.loc[index, "Recommendation"] = format_rec(issue_num, issue_rec)
        else : #for the N/A responses
            if ai_provided[index] :
                _, explanation = generate_recommendation(row['SIG Question Text'], row['Standards Context'], row['Additional Information'])
                sig_df.loc[index, "Issue"] = 'No'
                sig_df.loc[index, "Gap"] = 'Yes' if "contradicts" in ai_support[index] else 'No'
                sig_df.loc[index, "Recommendation"] = str(explanation)
            else :
                sig_df.loc[index, "Issue"] = 'No'
                sig_df.loc[index, "Gap"] = 'Yes'
                sig_df.loc[index, "Recommendation"] = 'N/A Response but no additional information given (This row may have been hidden)'
    #save it as a csv
    sig_df.to_csv("Assessor_report.csv")
    

    end = time.time()
    print("Total time elapsed: " + str(end - start) + " seconds")
    return

#-----------------RUNNER--------------------#


vendor_sig = '/Users/madisonlee/Documents/GitHub/solution-ThirdPartyRisk/assets/data/synthetic_SIG.csv'

create_assessor_report(vendor_sig)