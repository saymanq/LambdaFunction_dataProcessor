import requests
import boto3
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_batch
import json
import re
import os

def insert_vector_data(vectors_data):
    """Insert multiple vector records into the vectorData table"""
    sql = "INSERT INTO vector_data(vector_id, vector_text) VALUES(%s, %s)"
    
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            database=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD")
        )
        
        with conn.cursor() as cur:
            execute_batch(cur, sql, vectors_data)
        conn.commit()
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database error: {error}")
    finally:
        if conn is not None:
            conn.close()


def lambda_handler(event, context=None):
    if isinstance(event.get('body'), str):
        body = json.loads(event['body'])
    else:
        body = event.get('body', {})
    
    
    fileName = body.get('fileName')
    user, semester, location, course, name = fileName.split('/')[2:]
    if '.' in name:
        nametxt = name.split('.')[0] + '.txt'
        namejson = name.split('.')[0] + '.json'
        name = name.split('.')[0]

    #  Get data from the API
    response = requests.get(f'https://api.dataextract.sayman.me/{fileName}')
    docs = response.json()
    print(docs)
    #print(f"{"-"*100}")
    if not docs:
        responsult = {
            "response_type": "no_data",
            "response_code": "404",
            "ranked_topics_w_subtopics": [
                "Nothing"
            ]
        }
        return responsult

    # Create a text file and write the data to it
    with open('/tmp/upload.txt', 'w') as txt_file:
        for doc in docs:
            markdown_text = doc['text']
            txt_file.write(markdown_text + '\n\n')

    # First write the JSON data to a temporary file
    with open('/tmp/upload.json', 'w') as json_file:
        json.dump(docs, json_file)
    
    # Initialize the S3 client
    s3 = boto3.resource('s3',
        endpoint_url = os.environ.get("ENDPOINT_URL"),
        aws_access_key_id = os.environ.get("ACCESS_KEY_ID"),
        aws_secret_access_key = os.environ.get("SECRET_ACCESS_KEY"),
        region_name="wnam",
        )
    
    # Upload text file to R2 Object Store
    bucket = s3.Bucket('study-platform')
    bucket.upload_file('/tmp/upload.txt', f"{user}/{semester}/txtfiles/{course}/{nametxt}", ExtraArgs={'ContentType': 'text/plain'})
    bucket.upload_file('/tmp/upload.json', f"{user}/{semester}/jsonfiles/{course}/{namejson}", ExtraArgs={'ContentType': 'application/json'})
    
    # Delete the big file from R2 Object Store
    #bucket.Object(f"{'/'.join(fileName.split('/')[2:])}").delete()

    # Read markdown from file
    with open('/tmp/upload.txt', 'r') as file:
        markdown_document = file.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    chunk_size = 500
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    # print(type(splits))
    # print(len(splits))
    # print(splits)
    # return
    context = ''
    for i, doc in enumerate(splits):
        context += f'[PageTitle: {doc.metadata}\nPageContent: {doc.page_content}]\n\n'
    
    system_msg = SystemMessage(content="""You are a topic extraction API. Your task is to extract topics and subtopics from text and return them in a specific JSON format.""")
    ask = """You are a structured lecture notes/slides analyzer API made to help identify the top 5 main topics within a given piece of text from the notes/slides. Your task is to carefully process the text step by step, ensuring no relevant information is missed. Follow these detailed instructions to extract and organize the topics effectively:

    - Slowly read the text step by step to identify every distinct idea, concept, or subject mentioned.
    - Gradually build up from smaller ideas to broader, overarching topics.
    - Focus only on the top 5 **most important and relevant topics** from the text, based on the frequency and significance of their presence.
    - Avoid providing subtopics; instead, provide only the five overarching topics.
    - Use a hierarchy to represent relationships between topics in the JSON, limiting it to the top 5 topics ranked by importance.
    - If fewer than five topics are present, include only the topics identified, but maintain the rank ordering.
    - If no distinct topics are found, explain why (e.g., "The text is too vague to identify meaningful topics").
    - For highly technical or dense text, prioritize clarity and conciseness in topic names.
    - Ensure the Output Is Clear. Avoid redundancy or overly verbose topic names.
    - Ensure each topic reflects the content's ideas accurately.

    Format the response as a JSON object with exactly these fields:
    - response_type: either 'answer' or 'no_answer'
    - response_code: either '200' or '404'
    - ranked_topics_w_subtopics: an array containing exactly 5 strings (if available), each representing a distinct topic.

    Expected output format:
    {
        "response_type": "answer",
        "response_code": "200",
        "ranked_topics_w_subtopics": [
            "Topic1",
            "Topic2",
            "Topic3",
            "Topic4",
            "Topic5"
        ]
    }
    OR
    {
        "response_type": "no_answer",
        "response_code": "404",
        "ranked_topics_w_subtopics": [
            "Text is too vague or unclear to extract meaningful information"
        ]
    }

    Here is the text you need to analyze:
    """
    ask += context
    prompt = str([system_msg, HumanMessage(content=ask)])

    '''
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
    for i in range(3):
        try:
            print(f'Take {i+1}')
            completion = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct:free",
                messages=[
                    {
                    "role": "user",
                    "content": f"{prompt}"
                    }
                ]
            )
            output = completion.choices[0].message.content
            parsed_output = parse_llm_response(output)
            break
        except:
            continue
    else:
        parsed_output = {
            "response_type": "no_answer",
            "response_code": "404",
            "ranked_topics_w_subtopics": [
                "verybigerror2"
            ]
        }
    print(parsed_output)
    '''

    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
    )

    class APIResponse(BaseModel):
        response_type: str
        response_code: str
        ranked_topics_w_subtopics: list[str]

    completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                        "role": "user",
                        "content": f"{prompt}"
                        }
                    ],
                    response_format=APIResponse
                )
    output = completion.choices[0].message

    if (output.refusal):
        print(f"Refusal Message: {output.refusal}")
        response = {
            "response_type": "no_answer",
            "response_code": "404",
            "ranked_topics_w_subtopics": [
                "Text is too vague or unclear to extract meaningful information"
            ]
        }
        return response
    else:
        print("Output Parsed")
        # Convert the APIResponse object's fields to a dictionary and print as formatted JSON
        response = {
            "response_type": output.parsed.response_type,
            "response_code": output.parsed.response_code,
            "ranked_topics_w_subtopics": output.parsed.ranked_topics_w_subtopics
        }
        print(json.dumps(response, indent=2))
    

    docs_added = 0
    total_docs = len(splits)
    vectors_data = []
    for i, doc in enumerate(splits, 1):
        # Get title from metadata
        title = '|'.join(doc.metadata.values())

        url = f"https://api.vectorsadd.sayman.me/{user}/{semester}/{course}/{name}/{i}"

        payload = {
            "title": title,
            "text": doc.page_content
        }

        try:
            responseadd = requests.post(url, json=payload)
            responseadd.raise_for_status()

            result = responseadd.json()
            print(f"Successfully added document: {result}")

            vectors_data.append((result['docId'], doc.page_content))

            docs_added += 1
        except requests.exceptions.RequestException as e:
            print(f"Failed to add document: {e}")
            continue
    
    if vectors_data:
        insert_vector_data(vectors_data)
    
    response.update({
        "docs_added": docs_added,
        "total_docs": total_docs,
        "all_docs_added": docs_added == total_docs
    })
    return response

    '''
    Output Parsed:
    response_type='answer' response_code='200' ranked_topics_w_subtopics=['ANOVA Analysis', 'Model Summary', 'Coefficients Interpretation', 'Residuals Statistics', 'Regression Analysis']

    Only Output:
    ParsedChatCompletionMessage[lambda_handler.<locals>.APIResponse](content='{"response_type":"answer","response_code":"200","ranked_topics_w_subtopics":["ANOVA Analysis","Model Summary","Coefficients Interpretation","Residuals Statistics","Regression Analysis"]}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=APIResponse(response_type='answer', response_code='200', ranked_topics_w_subtopics=['ANOVA Analysis', 'Model Summary', 'Coefficients Interpretation', 'Residuals Statistics', 'Regression Analysis']))

    Parsed Type:
    <class '__main__.lambda_handler.<locals>.APIResponse'>

    Output Content Type:
    <class 'str'>
    '''


def parse_llm_response(llm_response):
    """
    Parse the output from the LLM and return the JSON in the desired format.

    Args:
        llm_response (str): The response string from the LLM.

    Returns:
        dict: Parsed JSON output.
    """
    # Regex pattern to extract JSON from the response
    pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'

    # Extract JSON from the text
    match = re.search(pattern, llm_response, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            raise Exception




#lambda_handler({'fileName': 'study/extract/user_2qecf04Tg3QKPPl7rfjf8hWosLr/Fall2024/bigfiles/3387612d-0b06-4720-9ae7-30ce1b620833/file1.pdf'}, None)