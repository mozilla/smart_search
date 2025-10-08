from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def evaluation_prompt_batch_gpt(
    query_results_pairs,
    custom_prompt=None,
    model="gpt-5-nano", #"gpt-5-mini", #"gpt-5",
    max_workers=8,
    openai_api_key=None
):
    """
    Batch version using OpenAI GPT API.
    Returns: list of dicts like [{'generated_text': '...'}, ...]
    """

    client = OpenAI(api_key=openai_api_key)

    if custom_prompt:
        print("Using custom prompt")

    def build_prompt(query, results):
        if custom_prompt:
            return custom_prompt
        return f'''Given the retrieval results URL metadata below, is the website primarily about query or strongly relevant to the query?

# Search Query: {query} 

# Retrieved result: {results}

# Evaluation 
Based on your reasoning, determine a relevance score between 0 and 1 (where 0 is not relevant, and 1 is highly relevant). Then, based on this score, assign a binary rating of 0 = not relevant or 1 = relevant. 

# Output format as a Dictionary
 search_query: <search query>,
 retrieved_result: <retrieved result>,
 relevance_score: <score>,
 binary_decision: 0 or 1,
 Decision_Reason: <reason for decision>
'''

    def one_call(qr):
        query, results = qr
        prompt = build_prompt(query, results)
        messages = [
            {"role": "system", "content": "You a judge who determines whether a returned web page is relevant to a user search query. The system cannot answer questions directly, only return relevant web pages. Return a JSON object only (valid JSON with double quotes, no markdown)."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                if text == "":
                    print(resp)
                return {"generated_text": text}
            except Exception as e:
                raise Exception(e)
                return str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        outputs = list(ex.map(one_call, query_results_pairs))

    return outputs
