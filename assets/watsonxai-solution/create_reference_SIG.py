import pandas as pd
from processStandards import create_standards_db
import warnings
import os
from ibm_watsonx_ai.foundation_models import ModelInference
import json, time
import traceback

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

from dotenv import load_dotenv
load_dotenv()

vdb = create_standards_db()

BATCH_SIZE = 8

#-----------------Functions--------------------#
#Get credentials for LLM call
def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.environ['API_KEY']
	}

#Leverage LLM to determine the "appropriate response" for the reference SIG
def determine_response(questions):
    """
    Feed in SIG question to LLM to ask for best Yes or No result as to what IBM BANK would want from their vendors

    :param question: SIG question from blank_SIG file
    :return: status (Yes, No)
    """

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

    prompt_input = """<|system|>
    You are Granite Chat, an AI language model developed by the IBM DMF Alignment Team. You are a cautious assistant that carefully follows instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior. You respond in a comprehensive manner unless instructed otherwise, providing explanations when needed while maintaining a neutral tone. You are capable of coding, writing, and roleplaying. You are cautious and refrain from generating real-time information, highly subjective or opinion-based topics. You are harmless and refrain from generating content involving any form of bias, violence, discrimination or inappropriate content. You always respond to greetings (for example, hi, hello, g'day, morning, afternoon, evening, night, what's up, nice to meet you, sup, etc) with "Hello! I am Granite Chat, created by the IBM DMF Alignment Team. How can I help you today?". Please do not say anything else and do not start a conversation.
    <|user|>
    I will give you a question from a Standardized Information Gathering (SIG) Questionnaire regarding security. Answer Yes or No. Use your best judgement as to what response IBM BANK would want from their vendors. 
    Please answer only in the following manner and in json output:
    {{"Response": "Yes", "Additional Context":"<if any additional context>"}}
    {{"Response": "No", "Additional Context":"<if any additional context>"}}
    <|assistant|>
    Here are a few examples: 
    SIG Question #1: Does the password policy require password expiration within 90 days or less?
    Answer: {{"Response": "Yes", "Additional Context": "To ensure proper security protocol and risk mitigation, password policy should require password expiration within 90 days or less."}}

    SIG Question #2: Does the password policy require changing passwords when there is an indication of possible system or password compromise?
    Answer: {{"Response": "Yes", "Additional Context": "When any indication of system or password comprise occurs, passwords need to be changed."}}

    SIG Question #3: Does the password policy apply to all network devices including routers, switches, and firewalls?
    Answer: {{"Response": "Yes", "Additional Context": "Passwords for every system should adhere to password policy."}}

    SIG Question #4: Is there an administrative process to revoke authenticators if required?
    Answer: {{"Response": "Yes", "Additional Context": "The ability to revoke access to certain personnel should always be required."}}

    SIG Question #5: Is access to systems that store, or process scoped data limited?
    Answer: {{"Response": "Yes", "Additional Context": "Access should always be limited to systems that store confidential data."}}
    <|user|>
    Please provide the answer to the following SIG Question in JSON:
    {}
    
    Answer:
    <|assistant|>
    """

    #print("Submitting generation request...")
    response = model.generate_text(prompt=[prompt_input.format(q) for q in questions])

    # Split the data if necessary and parse each JSON object
    responses = []
    for text in response :
        for part in text.split('\n'):
            try:
                response_data = json.loads(part)
                responses.append(response_data.get("Response", "Unknown"))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {part}")
                responses.append("ERROR")

    return responses
#Method 1: Leverage RAG to retrieve the most relevant context as it pertains to the security standards for each SIG question
def get_standards_context(question, num_results=1) :
    vdb_output = vdb.query(
                query_texts=[question],
                n_results=num_results
            )
    f_string_lst = [f"Heading: {vdb_output['metadatas'][0][i]['Heading']}, Content: {vdb_output['documents'][0][i]}" for i in range(num_results)]
    return f_string_lst



def test_context_relevant_any(questions, contexts):
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

    prompt_input = """I will give you an question from a vendor security assessment form and a three contexts from a firm's minimum security requirements document, both related to IBM BANK's third party vendor assessment process. For the input given, indicate whether or not any of the provided contexts is relevant to the question being asked. Be generous with your assessment, stating true if the context is only slightly relevant.
    Please answer only in the following manner and in json output, where true indicates that an context is relevant, and false indiates it is not.
    {{"Response": true}} or
    {{"Response": false}}

    Example Input 1:
    Question: Does the pandemic plan include an oversight program to ensure ongoing review and updates to the pandemic plan so that policies, standards, and procedures include up-to-date, relevant information provided by governmental sources? Contexts: ['Heading: II. IT Security Program Standard, Content: 2.10 Institutions must create an Incident Response Plan based on the "USM IT Incident Response Plan" Template. Incidents involving the compromise of personal information (as defined under State Government Article 10-301, see Section III) or confidential information (as defined in Appendix A of these standards) must be reported to security@usmd.edu.', "Heading: VI. Disaster Recovery & Incident Response Standard, Content: 6.4 Institutions must update their IT Incident Response and IT Disaster Recovery Plans annually.\n6.5 The institution must test the institution's IT Incident Response Plan at least annually and their disaster recovery plan at least annually. The tests must be documented. If an institution uses their incident response plan or disaster recovery plan to handle a real security or service interruption event, that event may be documented and take the place of the annual test. If a single event or test exercises both the disaster recovery and incident response plans, the one event or test can be used to meet both annual testing requirement.", "Heading: III. Auditability Standard, Content: Examples of significant events which should be reviewed and documented (where possible) include additions/changes to critical applications, actions performed by administrative level accounts, additions and changes to users' access control profiles, and direct modifications to critical data outside of the application. Where it is not possible to maintain such audit trails, the willingness to accept the risk of not auditing such actions should be documented ."]  
    Output:
    {{"Response": true}}

    Example Input 2:
    Question: "Are whitelisted and/or blacklisted applications documented and enforced?" Context:["Heading: 9.3 Third-party contracts should include the following as applicable:, Content: · Requirements for recovery of institutional resources such as data, software, hardware, configurations, and licenses at the termination of the contract.\nProvisions stipulating that the third-party service provider is the owner or authorized user of their software and all of its components, and the thirdparty's software and all of its components, to the best of third-party's knowledge, do not violate any patent, trademark, trade secret, copyright or any other right of ownership of any other party.\n· Service level agreements including provisions for non-compliance.\n· Provisions that stipulate that all institutional data remains the property of the institution.\n· Provisions that block the secondary use of institutional data.\n· Provisions that require the consent of the institution prior to sharing institutional data with any third parties.\nProvisions that manage the retention and destruction requirements related to institutional data.\n· Requirements to establish and maintain industry standard technical and organizational measures to protect against:\n· Provisions that require any vendor to disclose any subcontractors related to their services.\no accidental destruction, loss, alteration, or damage to the materials;\nunauthorized access to confidential information\no unauthorized access to the services and materials; and\nindustry known system attacks (e.g., hacker and virus attacks)\n· Requirements for reporting any confirmed or suspected breach of institutional data to the institution.\n· Requirements that the institution be given notice of any government or third-party subpoena requests prior to the contractor answering a request.\n· The right of the Institution or an appointed audit firm to audit the vendor's security related to the processing, transport, or storage of institutional data.\n· Requirement that the Service Provider must periodically make available a third-party review that satisfies the professional requirement of being performed by a recognized independent audit organization (refer to 9.1). In addition, the Service Provider should make available evidence of their business continuity and disaster recovery capabilities to mitigate the impact of a realized risk.\n· Requirement that the Service Provider ensure continuity of services in the event of the company being acquired or a change in management.\n· Requirement that the contract does not contain the following provisions:", "Heading: III. Auditability Standard, Content: Examples of significant events which should be reviewed and documented (where possible) include additions/changes to critical applications, actions performed by administrative level accounts, additions and changes to users' access control profiles, and direct modifications to critical data outside of the application. Where it is not possible to maintain such audit trails, the willingness to accept the risk of not auditing such actions should be documented .", 'Heading: 9.3 Third-party contracts should include the following as applicable:, Content: o The unilateral right of the Service Provider to limit, suspend, or terminate the service (with or without notice and for any reason).\n· Requirement that the Service Provider make available audit logs recording privileged user and regular user access activities, authorized and unauthorized access attempts, system exceptions, and information security events (as available) [reference Section III - Auditability Standard]\no A disclaimer of liability for third-party action.']  
    Output:
    {{"Response": false}}

    Input:
    Question: {} Context: {}

    Output: 
    """
    prompts = [prompt_input.format(questions.iloc[i], str(contexts[i])) for i in range(len(questions))]

    response = model.generate_text(prompt=prompts)
    responses = []
    for text in response :
        try:
            response_data = json.loads(text)
            responses.append(response_data)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {text}")
            responses.append("ERROR")


    return responses

   
#Save reference SIG to a new csv
def process_question(questions, nums):
    print(questions)
    if isinstance(questions.iloc[0], str):
        standards_contexts = [""] * len(questions)
        try :
            contexts = [get_standards_context(q, num_results=3) for q in questions]
            relevant_response = test_context_relevant_any(questions, contexts)
            for i in range(len(questions)) :
                if relevant_response[i]["Response"]:
                    standards_contexts[i] = contexts[i][0] + "; " + contexts[i][1] + "; " + contexts[i][2]
        except: 
            pass #if its not relevant or there was an error then dont include the context
        ref_responses = determine_response(questions)
        output = [{'SIG Question': questions.iloc[i], 'Response': ref_responses[i], 'Standards Context': standards_contexts[i]} for i in range(len(questions))]

        return output
    return None

def create_ref_sig(bs_df, batch_size=1):
    questions = bs_df["Content Library"].iloc[:, 2:4]

    df = pd.DataFrame(columns=['SIG Question', 'Response', 'Standards Context'])
    num_rows, _ = questions.shape
    for increment in range(0, num_rows, batch_size) :
        end = min(increment + batch_size, num_rows)
        try :
            result = process_question(questions.iloc[increment:end, 1], questions.iloc[increment:end, 0])
            df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
        except:
            try :
                result = process_question(questions.iloc[increment:end, 1], questions.iloc[increment:, 0])
                df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
            except Exception as e: 
                #give up
                print(f"failed a second time: {repr(e)}, {e}")
                print(traceback.format_exc())
                pass

    
    # Add to new reference SIG csv
    df.to_csv('golden_sig.csv', index=False)
    
    return df
	
	

#-----------------RUNNER--------------------#
start = time.time()
bs_df = pd.read_excel("../data/SIG_BLANK.xlsm", sheet_name=None)
create_ref_sig(bs_df, BATCH_SIZE)
end = time.time()
print(f"Program took {end - start} seconds to complete")