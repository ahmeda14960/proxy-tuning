import argparse
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uuid

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Flask server for LLM inference')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help='Name of the model to use')
args = parser.parse_args()

# Load your model and tokenizer
port = 8001
model_name = args.model_name
if model_name == 'tar':
    model_name = 'lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Cyber'
    port = 8002
    print(f"Using model {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
model = model.half().to('cuda')

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 16)
    temperature = data.get('temperature', 1.0)
    logprobs = data.get('logprobs', None)
    echo = data.get('echo', False)

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    attention_mask = torch.ones_like(input_ids).to('cuda')

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            return_dict_in_generate=True,
            top_p=0.9 if temperature > 0 else None,
            output_scores=True
        )
    
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids[input_ids.shape[1]:])
    
    # Prepare logprobs if requested
    logprobs_output = None
    if logprobs is not None:
        logprobs_output = {
            "token_logprobs": [],
            "top_logprobs": []
        }
        for i, token_scores in enumerate(outputs.scores):
            probs = torch.softmax(token_scores[0], dim=-1)
            top_probs, top_indices = probs.topk(logprobs)
            token_id = generated_ids[input_ids.shape[1] + i].item()
            token_logprob = torch.log(probs[token_id]).item()
            
            logprobs_output["token_logprobs"].append(token_logprob)
            logprobs_output["top_logprobs"].append({
                tokenizer.decode(idx.item()): torch.log(prob).item() 
                for idx, prob in zip(top_indices, top_probs)
            })

    response = {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "text": (prompt + generated_text) if echo else generated_text,
            "index": 0,
            "logprobs": logprobs_output,
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(generated_ids) - input_ids.shape[1],
            "total_tokens": len(generated_ids)
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=port)