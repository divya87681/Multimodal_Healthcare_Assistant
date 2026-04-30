from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LLMEngine:

    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_prompt(self, query, context):
        prompt = f"""
You are a safe medical assistant.

PATIENT HISTORY:
{context.get('patient_history', 'None')}

RELEVANT MEDICAL DATA:
{context.get('relevant_info', 'None')}

USER QUESTION:
{query}

If there is any contraindication or emergency risk, clearly warn the user.
"""
        return prompt

    def generate(self, query, context, max_tokens=256):
        prompt = self.build_prompt(query, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_tokens
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response