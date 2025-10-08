from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

def query_prompt_batch_gpt(
    pairs,                               # list[(url, combined_text)]
    model="gpt-5-mini",
    max_workers=8,
    openai_api_key=None,
):
    client = OpenAI(api_key=openai_api_key)

    def build_prompt(url, combined_text):
        return f"""You are given a web page URL and its title+description text.
Your task: write a SHORT natural-language search query that, under a SEMANTIC search engine (not keyword matching),
would retrieve this page. Do NOT include the URL, quotes, site: operators, or copy the title verbatim. 
Try to be less exact matched keywords. Try to be like how user type in browser search bar when search. Try to be less upper case. Prefer 3–8 words.

# URL:
{url}

# Title+Description:
{combined_text}

# Return JSON only with this schema:
{{
  "url": "{url}",
  "search_query": "<one concise semantic query>",
}}"""

    def one_call(item):
        url, combined_text = item
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Return a JSON object only (valid JSON, double quotes, no markdown). The word json appears here."},
            {"role": "user", "content": build_prompt(url, combined_text)},
        ]

        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                if not text:
                    print(resp)
                return {"generated_text": text}
            except Exception as e:
                print(e)
                return str(e)

    # keep input order
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        outputs = list(ex.map(one_call, pairs))

    return outputs
