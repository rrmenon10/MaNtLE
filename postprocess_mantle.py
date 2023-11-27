import os
import json
import argparse

def postprocess_explanation(exp, data):
    
    feature_names = {''.join(key.lower().split()):key for key in data[0]}
    feature_values = {''.join(key.lower().split()):key for d in data for key in d.values()}
    for feat, fname in feature_names.items():
        if feat in exp.split():
            exp = ' '.join([e if e!=feat else fname for e in exp.split()])
    for feat, fname in feature_values.items():
        if feat in exp.split():
            exp = ' '.join([e if e!=feat else fname for e in exp.split()])
        elif f'{feat},' in exp.split():
            exp = ' '.join([e if e!=f'{feat},' else f'{fname},' for e in exp.split()])
    target_class = list(data[0].keys())[-1]
    exp = ' '.join([target_class if e=='it' else e for e in exp.split()])
    return exp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Postprocessing MaNtLE Explanations", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset-name', required=True, help='Dataset name')
    parser.add_argument('--model-name', choices=['lr', 'dt', 'nn', 'xgb', ''], required=True, help='Model name')
    parser.add_argument('--num-subsets', type=int, default=100, required=False, help='Number of subsets')
    parser.add_argument('--explanations-file', required=True, help='Path to explanations')
    args = parser.parse_args()

    DATASET_PATH = f'data/{args.dataset_name}/mantle_subsets/{args.model_name}'
    with open(args.explanations_file, 'r') as f:
        explanations = [line[:-1] for line in f.readlines()]

    pp_exp = []
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
                    pass
                elif args.dataset_name == 'travel_insurance':
                    insurance = ex['Travel Insurance']
                    example[idx]['Travel Insurance'] = insurance
                else:
                    raise NotImplementedError('Please implement the equivalent fn for this dataset')
            data.extend(example)
        
        exp = postprocess_explanation(explanations[sub_idx], data)  
        pp_exp.append(exp)
    
    pp_file = args.explanations_file.replace('_explanations.txt', '_explanations_pp.txt')
    with open(pp_file, 'w') as f:
        for exp in pp_exp:
            f.write(exp + '\n')