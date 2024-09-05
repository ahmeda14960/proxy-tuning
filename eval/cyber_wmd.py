import argparse
import glob
import json
import os
import random
from collections import defaultdict
from datasets import load_dataset

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict




from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
from eval.utils import load_lm_and_tokenizer, load_dexperts_model_and_tokenizer


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def calculate_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

def write_samples(batch_inputs, batch_outputs, tokenizer, save_dir, 
                  base_ent, expert_ent, antiexpert_ent, dexpert_ent, 
                  file_name='samples.json'):
    file_path = os.path.join(save_dir, file_name)
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as f:
        for input_ids, output_ids in zip(batch_inputs, batch_outputs):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Find where the input ends and the actual output begins
            output_text = full_output[len(input_text):].strip()
            
            sample = {
                'input': input_text,
                'output': output_text,
                'avg_base_entropy': base_ent,
                'avg_expert_entropy': expert_ent,
                'avg_antiexpert_entropy': antiexpert_ent,
                'avg_dexpert_entropy': dexpert_ent
            }
            json.dump(sample, f)
            f.write('\n')

@torch.inference_mode()
def generate_and_evaluate(
    model,
    tokenizer,
    eval_dataset,
    save_dir,
    batch_size=16,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    temperature=1.0,
    top_p=0.9,
    debug=False,
    sample=False,
    **generation_kwargs
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating and Evaluating")):
        questions = batch['question']
        choices_batch = batch['choices']
        correct_answers = batch['answer']
        

        if debug:
            print(f"\n questions are: {questions}")
            print(f"\n choices_batch are: {choices_batch}")
            print(f"\n correct_answers are: {correct_answers}")
        
        choices_batch = [[choice for choice in choices] for choices in zip(*choices_batch)]
        prompts = []
        for idx, (q,c) in enumerate(zip(questions, choices_batch)):
            if debug:
                print(f"\nQuestion {idx}:")
                print(f"Question: {q}")
                print(f"Choices: {c}")
                print(f"Correct Answer: {correct_answers}")
            prompts.append(f"{q} {' '.join(choice[0] for choice in choices_batch)}")


        
        # Generate completions
        tokenized_prompts = tokenizer(
            prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts['input_ids'].to(device)
        attention_mask = tokenized_prompts['attention_mask'].to(device)

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        logits_processor = None
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)

        # Generate using DExperts model (keep this part exactly the same)
        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
            return_logits_for_analysis=True,
            **generation_kwargs
        )

        
        batch_outputs, analysis_data = outputs
        
        batch_logits = analysis_data['logits_dexperts']
        if debug:
            print('\n analysis_data:', analysis_data.keys())
        base_next_token_logits = analysis_data['logits_base']
        expert_next_token_logits = analysis_data['logits_expert']
        antiexpert_next_token_logits = analysis_data['logits_antiexpert']


        base_entropy = calculate_entropy(base_next_token_logits)
        expert_entropy = calculate_entropy(expert_next_token_logits)
        antiexpert_entropy = calculate_entropy(antiexpert_next_token_logits)
        dexpert_entropy = calculate_entropy(batch_logits)

        avg_base_entropy = base_entropy.mean().item()
        avg_expert_entropy = expert_entropy.mean().item()
        avg_antiexpert_entropy = antiexpert_entropy.mean().item()
        avg_dexpert_entropy = dexpert_entropy.mean().item()
        if debug:
            print('\n shape of base_next_token_logits:', base_next_token_logits.shape)
            print('\n shape of expert_next_token_logits:', expert_next_token_logits.shape)
            print('\n shape of antiexpert_next_token_logits:', antiexpert_next_token_logits.shape)
            print('\n avg_base_entropy:', avg_base_entropy)
            print('\n avg_expert_entropy:', avg_expert_entropy)
            print('\n avg_antiexpert_entropy:', avg_antiexpert_entropy)
            print('\n avg_dexpert_entropy:', avg_dexpert_entropy)

        if sample:
            write_samples(batch_input_ids, batch_outputs, tokenizer, save_dir, avg_base_entropy, avg_expert_entropy, avg_antiexpert_entropy, avg_dexpert_entropy, 'samples.json')

        # Evaluate
        for q_idx, (question, question_choices, correct_answer) in enumerate(zip(questions, choices_batch, correct_answers)):
            question_logits = batch_logits[q_idx]
            
            choice_scores = []
            for choice in question_choices:
                choice_tokens = tokenizer.encode(choice, add_special_tokens=False)
                choice_score = question_logits[:len(choice_tokens), choice_tokens].sum().item()
                choice_scores.append(choice_score)
            
            predicted_index = choice_scores.index(max(choice_scores))
            
            if predicted_index == correct_answer:
                correct += 1
            total += 1

            if debug:
                print(f"\nEval Question {batch_idx * batch_size + q_idx + 1}:")
                print(f"Eval Question text: {question}")
                print(f"Eval Choices: {question_choices}")
                print(f" eval Predicted index: {predicted_index}")
                print(f" eval Correct index: {correct_answer}")
                print(f" eval Correct: {'Yes' if predicted_index == correct_answer else 'No'}")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        if debug:
            print(f"Batch {batch_idx + 1} accuracy: {correct}/{total} = {correct/total:.4f}")

    accuracy = correct / total
    print(f"Final accuracy: {accuracy}")
    return accuracy

def custom_evaluate(tokenizer, eval_dataset, eval_mode, batch_size=16, logits=None, debug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    
    correct = 0
    total = 0
    logits_index = 0

    print(f"Starting evaluation with mode: {eval_mode}")
    print(f"Total number of samples: {len(eval_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total logits shape: {logits.shape}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        questions = batch['question']
        choices_batch = batch['choices']
        correct_answers = batch['answer']

        choices = list(zip(*choices_batch))

        print(f"\nBatch {batch_idx + 1}:")
        print(f"Number of questions in this batch: {len(questions)}")
        print(f"Number of choices per question: {len(choices[0]) if choices else 0}")
        print(f"Correct answers: {correct_answers}")

        if eval_mode == "logits":
            for q_idx, (question, question_choices, correct_answer) in enumerate(zip(questions, choices, correct_answers)):
                if debug:
                    print(f"\n  Question {batch_idx * batch_size + q_idx + 1}: {question}")
                    print(f"    Choices: {question_choices}")
                    print(f"    Correct answer index: {correct_answer}")
                
                # Extract logits for this question's choices
                question_logits = logits[logits_index:logits_index + len(question_choices)]
                
                if debug:
                    print(f"    Logits shape for this question: {question_logits.shape}")
                
                if question_logits.numel() == 0:
                    print(f"    Warning: Empty logits tensor for question {q_idx + 1}")
                    predicted_index = 0  # Default to first choice if logits are empty
                else:
                    # Calculate scores for each choice
                    choice_scores = []
                    for choice_idx, choice in enumerate(question_choices):
                        choice_tokens = tokenizer.encode(choice, add_special_tokens=False)
                        choice_score = question_logits[choice_idx, :len(choice_tokens), choice_tokens].sum().item()
                        choice_scores.append(choice_score)
                    
                    if debug:
                        print(f"    Choice scores: {choice_scores}")
                    
                    predicted_index = choice_scores.index(max(choice_scores))
                
                if debug:
                    print(f"    Predicted index: {predicted_index}")
                    print(f"    Predicted choice: {question_choices[predicted_index]}")
                    print(f"    Correct choice: {question_choices[correct_answer]}")
                
                if predicted_index == correct_answer:
                    correct += 1
                    if debug:
                        print("    Correct!")
                else:
                    if debug:
                        print("    Incorrect.")
                total += 1
                
                logits_index += len(question_choices)  # Move to the next set of choices

            print(f"    Current accuracy: {correct}/{total} = {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.4f}")
    return accuracy

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # fix eval dataset from TAR
    eval_dataset = load_dataset("cais/wmdp", "wmdp-cyber")['test']
    eval_dataset = eval_dataset.select(range(100))
    #eval_dataset = eval_dataset.map(lambda x: {'question': x['question'], 'choices': x['choices'], 'answer': x['answer']['idx']})


    if args.mode == "single":
        print("Loading single HuggingFace model and tokenizer for generations...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.mode == "dexperts":
        print("Loading DExperts model and tokenizer for generations...")
        model, tokenizer = load_dexperts_model_and_tokenizer(
            base_model_name_or_path=args.base_model_name_or_path,
            expert_model_name_or_path=args.expert_model_name_or_path,
            antiexpert_model_name_or_path=args.antiexpert_model_name_or_path,
            #system_prompt=args.system_prompt,
            alpha=args.alpha,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer
        )


    prompts = [f"{example['question']} {' '.join(example['choices'])}" for example in eval_dataset]

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])
    print(prompts[0], flush=True)

    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]

    accuracy = generate_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        save_dir=args.save_dir,
        batch_size=args.eval_batch_size,
        stop_id_sequences=[[new_line_token]],
        do_sample=False,
        max_new_tokens=512,
        debug=args.print_completions,
        sample=args.sample
    )

    results = {
        "accuracy": accuracy
    }

    # Convert any tensors in results to regular Python types
    results = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results.items()}

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        print(f"Accuracy: {accuracy}")
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results/mc_evaluation_cyber_wmd")
    parser.add_argument("--sample", action="store_true", help="Write samples to samples.json")
    parser.add_argument("--mode", type=str, choices= ["dexperts"], default="dexperts", help="Mode to run the model: single for HuggingFace model, dexperts for DExperts ensemble")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--expert_model_name_or_path", type=str, default=None)
    parser.add_argument("--antiexpert_model_name_or_path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--print_completions", action="store_true", help="Print completions during generation")
    args = parser.parse_args()

    main(args)