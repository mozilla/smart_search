from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch



class llm_as_judge:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model =  AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            )
        else:
            self.model =  AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            )


        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
     

    def evaluation_prompt(self, query, results, custom_prompt=None):
        tokenizer = self.tokenizer
        model = self.model
        prompt = f'''Given the retrieval results URL metadata below, is the website primarily about query or strongly relevant to the query?

        # Search Query: {query} 
        
        # Retrieved result: {results}
        
        # Evaluation 
        Based on your reasoning, determine a relevance score between 0 and 1 (where 0 is not relevant, and 1 is highly relevant). Then, based on this score, assign a binary rating of 0 = not relevant or 1 = relevant. 

        # Output format as a Dictionary
         search_query: <search query>,
         retrieved_result: <retrieved result>,
         relevance_score: <score>,
         binary_decison: 0 or 1,
         Decision_Reason: <reason for decision>
         
         '''
        if custom_prompt: 
            print("Using custom prompt") 
        else: 
            pass
   
        messages = [
            {"role": "system", "content": "You a judge who determines whether a returned web page is relevant to a user search query. The system cannot answer questions directly, only return relevant web pages."},
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 600,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        print(output[0]['generated_text'])
        return output


