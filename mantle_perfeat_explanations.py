import os
import json
import torch
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from utils.tokenize_txt import tokenize_t5_explanation_txt
from utils.mantle_utils import get_features, parse

global device; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    parser = argparse.ArgumentParser(description="MaNtLE PerFeat Explanations Generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset-name', required=True, help='Dataset name')
    parser.add_argument('--model-name', choices=['lr', 'dt', 'nn', 'xgb', ''], required=True, help='Model name')
    parser.add_argument('--num-subsets', type=int, default=100, required=False, help='Number of subsets')
    args = parser.parse_args()


    DATASET_PATH = f'data/{args.dataset_name}/mantle_subsets/{args.model_name}'
    tokenizer = T5Tokenizer.from_pretrained('t5-large', model_max_length=1024)
    t5_config = T5Config.from_pretrained('pretrained_mantle')
    model = T5ForConditionalGeneration.from_pretrained('pretrained_mantle', config=t5_config).to(device)
    mantle_config = json.load(open('pretrained_mantle/mantle_config.json', 'r'))
    model.eval()

    output_txts = []
    for sub_idx in range(args.num_subsets):

        data_path = os.path.join(DATASET_PATH, f'{sub_idx+1}', 'data.jsonl')
        data = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            example = json.loads(lines[0][:-1])['samples']
            for idx, ex in enumerate(example):
                if args.dataset_name == 'adult':
                    income = ex['Income'] if ex['Income'] == '>50K' else 'not >50K'
                    example[idx]['Income'] = income
                elif args.dataset_name == 'recidivism':
                    recidivism = ex['Recidivism']
                    example[idx]['Recidivism'] = recidivism
                elif args.dataset_name == 'travel_insurance':
                    insurance = ex['Travel Insurance']
                    example[idx]['Travel Insurance'] = insurance
                else:
                    raise NotImplementedError('Please implement the equivalent fn for this dataset')
            data.extend(example)
        
        txt = get_features(data)
        input_ids = tokenize_t5_explanation_txt(tokenizer, mantle_config['max_text_length'], \
                        txt, prompt=mantle_config['prompt_exp'], lm_adapt=False).to(device)
        
        start_token = tokenizer.additional_special_tokens[0] # tokenizer.additional_special_tokens_ids[0]
        features = [f"{''.join(col.lower().split())}" for col in data[0]][:-1] # skip the last column which is the target
        outputs = []
        for feat in features:
            decoder_prompts = f'{start_token} If {feat}'
            tokenized_prompt = tokenizer(decoder_prompts, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                output = model.generate(input_ids, 
                                        decoder_start_token_id=tokenized_prompt, 
                                        max_length=mantle_config['exp_max_text_length'],
                                        num_beams=20//len(features),
                                        num_return_sequences=20//len(features))
            outputs.extend(output)
        eos_token_id = tokenizer.eos_token_id
        sequences = [seq.cpu().numpy().tolist() + [eos_token_id] for seq in outputs]
        sequences = [seq[:seq.index(eos_token_id)] for seq in sequences] # read until first eos token
        output_txt = [tokenizer.decode(seq[1:]) for seq in sequences] # skip the start_token
        best_exp = output_txt[parse(output_txt, data, args.dataset_name)]
        output_txts.append(best_exp)

    print(*output_txts, sep='\n')
    # Print out the different texts from the model
    with open(f'data/{args.dataset_name}/mantle_subsets/{args.model_name}/mantle_perfeat_explanations.txt', 'w') as f:
        for exp in output_txts:
            f.write(exp + '\n')

if __name__=='__main__':
    main()