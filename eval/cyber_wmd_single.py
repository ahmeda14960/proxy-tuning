import argparse
import json
import os
import random
from collections import defaultdict
from datasets import load_dataset

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval.utils import load_lm_and_tokenizer, load_dexperts_model_and_tokenizer

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    mode="single",
    sample=False,
    **generation_kwargs
):
    all_logits = []
    if not disable_tqdm:
        progress = tqdm(total=len(prompts), desc="Generating Completions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True
        )
        batch_input_ids = tokenized_prompts['input_ids'].to(model.device)
        attention_mask = tokenized_prompts['attention_mask'].to(model.device)

        if mode == "single":
            with torch.no_grad():
                outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
            batch_logits = outputs.logits
        else:  # dexperts mode
            raise NotImplementedError("DExperts mode is not implemented in this version.")

        all_logits.append(batch_logits.detach().cpu())

        if not disable_tqdm:
            progress.update(len(batch_prompts))

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    all_logits = torch.cat(all_logits, dim=0)
    
    if sample:
        generations = tokenizer.batch_decode(torch.argmax(all_logits, dim=-1), skip_special_tokens=True)
    else:
        generations = None

    return generations, all_logits

def custom_evaluate(tokenizer, eval_dataset, eval_mode, batch_size=16, logits=None, debug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    
    correct = 0
    total = 0

    print(f"Logits shape: {logits.shape}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        questions = batch['question']
        choices_batch = batch['choices']
        correct_answers = batch['answer']

        choices = list(zip(*choices_batch))

        if eval_mode == "logits":
            predicted_indices = []
            for question_idx, (question, question_choices) in enumerate(zip(questions, choices)):
                num_choices = len(question_choices)
                if debug:
                    print(f"\nQuestion {batch_idx * batch_size + question_idx + 1}:")
                    print(f"Question text: {question}")
                    print(f"Number of choices: {num_choices}")
                    print(f"Choices: {question_choices}")
                    print(f"Correct answer index: {correct_answers[question_idx]}")
                
                # Extract logits for this question
                question_logits = logits[batch_idx * batch_size + question_idx]
                if debug:
                    print(f"Question logits shape: {question_logits.shape}")
                
                # Calculate scores for each choice
                choice_scores = []
                for choice in question_choices:
                    # Tokenize the choice
                    choice_tokens = tokenizer.encode(choice, add_special_tokens=False)
                    # Sum the logits for the choice tokens
                    choice_score = question_logits[:, choice_tokens].sum().item()
                    choice_scores.append(choice_score)
                
                if debug:
                    print(f"Choice scores: {choice_scores}")
                
                predicted_index = choice_scores.index(max(choice_scores))
                predicted_indices.append(predicted_index)
                
                if debug:
                    print(f"Predicted answer index: {predicted_index}")
                    print(f"Predicted answer text: {question_choices[predicted_index]}")
                    print(f"Correct: {'Yes' if predicted_index == correct_answers[question_idx] else 'No'}")

        correct += sum(pred == true for pred, true in zip(predicted_indices, correct_answers))
        total += len(correct_answers)

        print(f"Batch accuracy: {correct}/{total}")

    accuracy = correct / total
    print(f"Final accuracy: {accuracy}")
    return accuracy

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    eval_dataset = load_dataset("cais/wmdp", "wmdp-cyber")['test']
    if args.num_samples and args.num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(args.num_samples))
    

    if args.mode == "single":
        print("Loading single HuggingFace model and tokenizer for generations...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.mode == "dexperts":
        raise NotImplementedError("DExperts mode is not implemented in this version.")

    prompts = [f"{example['question']} {' '.join(example['choices'])}" for example in eval_dataset]

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])
    print(prompts[0], flush=True)

    generations, logits = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        do_sample=False,
        mode=args.mode,
        sample=args.sample  # Pass the sample flag here
    )

    # If sampling is enabled, save the generations
    if args.sample and generations:
        with open(os.path.join(args.save_dir, "generations.json"), 'w') as fout:
            json.dump(generations, fout, indent=2)

    accuracy = custom_evaluate(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        eval_mode="logits",
        batch_size=args.eval_batch_size,
        logits=logits
    )

     # Convert accuracy to a Python float here
    accuracy = float(accuracy)

    results = {
        "accuracy": accuracy
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        print(f"Accuracy: {accuracy:.4f}")  # Format to 4 decimal places
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results/mc_evaluation_cyber_wmd")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from the dataset")
    parser.add_argument("--mode", type=str, choices=["single", "dexperts"], default="single", help="Mode to run the model: single for HuggingFace model, dexperts for DExperts ensemble")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Generate and save completions")
    args = parser.parse_args()

    main(args)