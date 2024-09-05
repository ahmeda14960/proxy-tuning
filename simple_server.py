import argparse
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    max_tokens = data.get('max_tokens', 1)
    temperature = data.get('temperature', 0)
    logprobs = data.get('logprobs', 1)
    seed = data.get('seed', 1234)
    echo = data.get('echo', True)

    
    # if tar don't unsqueeze i guess
    # if model_name == 'tar':
    #     # Convert the pre-tokenized input to tensor
    #     input_ids = torch.tensor(prompt).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU
    # else:
        # Convert the pre-tokenized input to tensor
    input_ids = torch.tensor(prompt).to('cuda')  # Add batch dimension and move to GPU
    # Generate logits
    # import ipdb; ipdb.set_trace()
    print(f'input_ids shape: {input_ids.shape}')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Get top logprobs
    top_logprobs, top_tokens = log_probs.topk(logprobs, dim=-1)

    # Convert to list of dictionaries
    token_logprobs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1).tolist()[0]  # Remove batch dimension
    tokens = [tokenizer.decode([token]) for token in input_ids[0].tolist()]
    top_logprobs = [{tokenizer.decode([token.item()]): prob.item() for token, prob in zip(top_tok, top_prob)} 
                    for top_tok, top_prob in zip(top_tokens[0], top_logprobs[0])]

    response = {
        "choices": [{
            "text": tokenizer.decode(input_ids[0]) if echo else "",
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }
        }]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=port)