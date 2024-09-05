from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from eval.utils import load_dexperts_model_and_tokenizer
from collections import defaultdict

app = Flask(__name__)

# Load DExperts model and tokenizer
base_model_name =  "lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Cyber" 
expert_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
antiexpert_model_name = "meta-llama/Meta-Llama-3-8B"
alpha = 1.0  # Set your desired alpha value

print("Loading DExperts model and tokenizer...")
print(f"Base model: {base_model_name}")
print(f"Expert model: {expert_model_name}")
print(f"Anti-expert model: {antiexpert_model_name}")

model, tokenizer = load_dexperts_model_and_tokenizer(
    base_model_name_or_path=base_model_name,
    expert_model_name_or_path=expert_model_name,
    antiexpert_model_name_or_path=antiexpert_model_name,
    alpha=alpha,
    load_in_8bit=False,
    use_fast_tokenizer=True
)

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 1.0)
    logprobs = data.get('logprobs', 1)
    seed = data.get('seed', 1234)
    echo = data.get('echo', True)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['attention_mask'].to('cuda')

    # Prepare kwargs for generate function
    generate_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_tokens,
        'do_sample': temperature > 0,
        'temperature': temperature,
        'top_p': 0.7,  # You can adjust this or make it configurable
        'return_logits_for_analysis': True
    }

    # Generate using DExperts model
    with torch.no_grad():
        generated_ids, analysis_data = model.generate(**generate_kwargs)

    # Process the output
    generated_text = tokenizer.decode(generated_ids[0])
    response_text = generated_text if echo else generated_text[len(prompt):]

    # Process logits and token probabilities
    logits = analysis_data['logits_dexperts'][0]  # Assuming batch size 1
    log_probs = torch.log_softmax(logits, dim=-1)

    # Get top logprobs
    top_logprobs, top_tokens = log_probs.topk(logprobs, dim=-1)

    token_logprobs = log_probs[0, generated_ids[0]].tolist()
    
    tokens = [tokenizer.decode([token]) for token in generated_ids[0].tolist()]
    top_logprobs = [{tokenizer.decode([token.item()]): prob.item() for token, prob in zip(top_tok, top_prob)} 
                    for top_tok, top_prob in zip(top_tokens, top_logprobs)]

    

    response = {
        "choices": [{
            "text": response_text,
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }
        }]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8002)