import os
import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

norm_dict = {
    'housework_qa': 'un',
    'neg_housework_qa': 'un',
    'act_infer': 'un',
    'act_recog': 'ln',
    'count': 'ln',
    'obj_move': 'ln',
    'eval_dataset_48': 'ln'
}

SURFACE_LIST = [
    "coffeetable",
    "desk"
    "kitchentable",
    "sofa",
]

CONTAINER_LIST = [
    "kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

NEW_CONTAINER_LIST = [
    "1st kitchencabinet from left to right",
    "2nd kitchencabinet from left to right",
    "3rd kitchencabinet from left to right",
    "4th kitchencabinet from left to right",
    "5th kitchencabinet from left to right",
    "6th kitchencabinet from left to right",
    "7th kitchencabinet from left to right",
    "8th kitchencabinet from left to right",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

OBJECT_LIST = [ 
    "apple",
    "book",
    "chips",
    "condimentbottle",
    "cupcake",
    "dishbowl",
    "plate",
    "remotecontrol",
    "salmon",
    "waterglass",
    "wine",
    "wineglass",
]

POSSIBLE_BELIEF = [
    "1st kitchencabinet",
    "2nd kitchencabinet",
    "3rd kitchencabinet",
    "4th kitchencabinet",
    "5th kitchencabinet",
    "6th kitchencabinet",
    "7th kitchencabinet",
    "8th kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
    "coffeetable",
    "desk"
    "kitchentable",
    "sofa"
]

def remove_item(s, item_to_remove):
    items = [item.strip() for item in s.split(',')]
    if item_to_remove in items:
        items.remove(item_to_remove)
    return ', '.join(items)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='erp.json')
    parser.add_argument('--model_name_or_path', type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument('--load_lora', type=int, default=1)
    parser.add_argument('--lora_name_or_path', type=str, default=None)
    parser.add_argument('--output_log', action="store_true")
    parser.add_argument('--output_path', type=str, default='output/ewc-lora-6B/qa-metric.txt')
    args = parser.parse_args()
    return args

def main(args):
    norm = 'ln'
    acc_list = []
    
    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token="REDACTED", add_bos_token = False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=False, device_map={"": 0}, token="REDACTED")

    if args.load_lora:
        model = PeftModel.from_pretrained(model, args.lora_name_or_path, device_map={"": 0})
    model.eval()

    def compute_prob(inp, contxt_len, answer_tokens):
        inputs = tokenizer(inp, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        start_idx = contxt_len - 1
        end_idx = contxt_len + answer_tokens.size(1)

        pred_token_ids = logits[:, start_idx:end_idx].argmax(dim=-1)[0].tolist()
        predicted_text = tokenizer.decode(pred_token_ids)
        logits = logits[:, contxt_len - 1:inputs['attention_mask'].sum()]
        
        '''
        # Take argmax to get most probable token at each step
        pred_token_ids = logits.argmax(dim=-1)  # shape: [batch, seq_len]

        # Convert to a Python list
        pred_token_ids = pred_token_ids[0].tolist()

        # Decode to text
        predicted_text = tokenizer.decode(pred_token_ids)

        print("Predicted Tokens:", pred_token_ids)
        print("Predicted Text:", predicted_text)
        
        # Example generation
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,  # only need yes/no
            do_sample=False  # greedy decoding
        )
        predicted_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
        print("Predicted Text:", predicted_text)
        #import pdb; pdb.set_trace()
        '''
        vocab_log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            vocab_log_probs, dim=2, index=answer_tokens[:, :, None]
        )
        log_prob = token_log_probs.sum()
        return log_prob.cpu().item()

    answer_contxt_len = tokenizer('Answer:', return_tensors="pt").input_ids.size(1)
    with torch.no_grad():
        with open(args.data_path, "r") as f:
            d = {}
            acc_list = []
            correct = {}
            all = {}
            lines = list(f)
            count = 0
            for line in tqdm(lines):
                count += 1
                sample = json.loads(line)
                print(f"\n\n\nQuestion {count}")
                print(f"Type: {sample['question_type']}")
                actions = sample['actions']
                actions[0] = 'None'
                actions[-1] = actions[-1].replace("about to open", "walktowards").replace("prepare to open", "walktowards").replace("ready to open", "walktowards")
                T = len(actions) - 1


                if sample['question_type'] in [1.1, 1.2, 1.3]:
                    # Belief inference
                    if sample['question_type'] == 1.3:
                        sample['answer'] = 'b'
                    else:
                        sample['answer'] = 'a'
                    print('answer', sample['answer'])
                    
                    hypo_belief = sample['hypo_belief']
                    hypo_goal = sample['hypo_goal']
                    print(f'hypo_belief: {hypo_belief}')
                    print(f'hypo_goal: {hypo_goal}')

                    beliefs = ['init_belief']
                    for i in range(1, T+1):
                        beliefs.append(sample[f'belief_{i}'])


                    last_belief = [b.strip() for b in beliefs[-1].split(",")]
                    hypo_belief_choice = beliefs.copy()
                    hypo_belief_choice[i] = remove_item(hypo_belief_choice[i], hypo_belief)
                    choices = [hypo_belief_choice]
                    for belief in last_belief:
                        if belief == hypo_belief:
                            continue
                        
                        other_belief_choice = beliefs.copy()
                        other_belief_choice[i] = remove_item(other_belief_choice[i], belief)
                        choices.append(other_belief_choice)

                    prob_choices = []
                    #P_acc = 0.5
                    
                    for beliefs in choices:  # choices = sequences of hypothetical beliefs, still kept if you need it
                        P_acc = 0.5
                        print("Testing belief sequence:", choices)
                        prob_list = []

                        for i in range(0, T+1):
                            state = sample[f'state_{i}']
                            prompt = f'The following describes the environment state and behavior. ONLY use the DRIVERS action and signals (eye movements, EEG, speech) to infer awareness. DO NOT use the environment truth. goal: {hypo_goal}\nstate: {state}\n given the current driver action: {actions[i]}, has the driver noticed the pedestrian/car/cyclist? Answer can only be yes or no. Answer:'

                            # reset for each step
                            prob_1 = 0.0  # yes
                            prob_2 = 0.0  # no

                            for answer in ["yes","no"]:
                                prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.size(1)
                                answer_tokens = tokenizer(f'{prompt} {answer}', return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)[:, prompt_len:]
                                #print(answer)
                                log_prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                                #log_prob = compute_prob(f'{prompt}', prompt_len, answer_tokens)
                                prob = math.exp(log_prob)  # convert to probability

                                if answer == "yes":
                                    prob_1 = prob
                                else:
                                    prob_2 = prob

                            # normalize to ensure P_yes + P_no = 1
                            total = prob_1 + prob_2
                            P_yes = prob_1 / total
                            P_no = prob_2 / total

                            print(prompt)
                            print(f"Step {i}: state = {state}")
                            print(f"Action: {actions[i]}")
                            print(f"P_yes = {P_yes:.4f}, P_no = {P_no:.4f}, Sum = {P_yes+P_no:.4f}")
                            #P_acc *= P_yes
                            P_acc = (P_yes * P_acc) / (P_yes * P_acc + P_no * (1 - P_acc))

                            print(f"Probabilities of noticing at each step: {P_acc}")
                            prob_list.append(P_yes)  # or store both if you want [P_yes, P_no]

                        #print(f"Probabilities of noticing at each step: {prob_list}")
                        total_prob = sum(math.log(p) for p in prob_list)#sum(prob_list)  # total accumulated yes-prob (can change to product if Bayesian)
                        prob_choices.append(total_prob)
                        print(f"Total notice probability (summed across steps): {total_prob}\n")

                    if (sample['answer'] == 'a' and prob_choices[0] == min(prob_choices)) or (sample['answer'] == 'b' and prob_choices[0] != min(prob_choices)):
                        print("correct")
                        acc_list.append(1)
                        correct[sample['question_type']] = correct.get(sample['question_type'], 0) + 1
                    else:
                        print("wrong")
                        acc_list.append(0)
                    all[sample['question_type']] = all.get(sample['question_type'], 0) + 1

                else:
                    # Goal inference
                    prob_choices = []
                    for num in ["1", "2"]:
                        prob_list = []
                        for i in range(1, T+1):
                            goal = sample[f'hypo_goal{num}']
                            state = sample[f'state_{i}']
                            belief = sample[f'goal{num}_belief_{i}']
                            if f'goal{num}_remove_belief' in sample.keys():
                                belief = remove_item(belief, sample[f'goal{num}_remove_belief'])

                            prompt = f'goal: {goal} \nstate: {state} \nbelief (possible locations the person suspects the {goal} could be): {belief} \naction: '

                            answer = actions[i]
                            prompt_len = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.size(1)
                            answer_tokens = tokenizer(f'{prompt} {answer}', return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)[:,prompt_len:]
                            prob = compute_prob(f'{prompt} {answer}', prompt_len, answer_tokens)
                            final_prob = prob / answer_tokens.size(1)
                            print(f'prompt: {prompt}')
                            print(f'answer: {answer}')
                            print(f"P(A_{i} | G, S_{i}, B_{i}): {final_prob}")
                            prob_list.append(final_prob)
                        total_prob = sum(prob_list)
                        print(f"total probabilities: {total_prob}")
                        prob_choices.append(total_prob)
                    
                    if (prob_choices[0] >= prob_choices[1] and sample['answer'] == 'a') or (prob_choices[0] < prob_choices[1] and sample['answer'] == 'b'):
                        print("correct")
                        acc_list.append(1)
                        correct[sample['question_type']] = correct.get(sample['question_type'], 0) + 1
                    else:
                        print("wrong")
                        acc_list.append(0)
                    all[sample['question_type']] = all.get(sample['question_type'], 0) + 1


    for question_type in [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]:
        print(f"type {question_type}: {correct.get(question_type, 0)} / {all.get(question_type, 0)}")      
    acc = sum(acc_list) / len(acc_list)
    return acc

if __name__ == "__main__":
    args = parse_args()
    args.task_name = os.path.basename(args.data_path).split('.')[0]
    acc = main(args)

    task_name = args.task_name.replace('_', ' ')
    output_str = f"{task_name}: {acc:.4f}"
    print(output_str)
    if args.output_log:
        with open(args.output_path, 'a') as f:
            f.write(output_str + '\n')
