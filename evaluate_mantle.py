import json
import re
import copy
import numpy as np
import argparse

import functools, operator

from utils.semparse import simple_parse, conjunctive_parse

QUANTIFIER_MAP_2_LIST = {
    "0.95" : ["always", "certainly", "definitely"],
    "0.70" : ["usually", "normally", "generally", "likely", "typically"],
    "0.50" : ["often"],
    "0.30" : ["sometimes", "frequently"],
    "0.20" : ["occasionally"],
    "0.10" : ["rarely", "seldom"],
    "0.05" : ["never"]
}
quantifiers = list(QUANTIFIER_MAP_2_LIST.values())
quantifiers = functools.reduce(operator.iconcat, quantifiers, [])
QUANTIFIER_SCORE_MAP = json.load(open("merged_quantifier_utils/quantifier_score_map.json", "r"))

def parse_eval(explanations, faith=False, model='lr', dataset_name='adult'):

    '''
    :return:
    Computes the faithfulness metric for the reference
    '''
    assert isinstance(explanations, list), "Expected exp_txts to be of type 'list' for this function"
    if 'adult' == dataset_name:
        task_lbls = ['>50K'.lower(), 'not >50K'.lower()]
        col_names = ['capitalgain', 'workclass', 'education', 'maritalstatus', 'hoursperweek', 'age',
                     'education-num', 'occupation', 'relationship', 'race', 'sex', 'capitalloss',
                     'country']
    elif dataset_name == 'recidivism':
        task_lbls = ['commit', 'not commit']
        col_names = ['race', 'alcohol', 'married', 'priors', 'prisonviolations']
    elif 'travel_insurance' == dataset_name:
        task_lbls = ['interested', 'not interested']
        col_names = ['age', 'annualincome', 'familymembers', 'frequentflyer', 'evertravelledabroad']
    else:
        raise NotImplementedError('Not implemented for this dataset')
    
    faith_scores = []
    precision = []
    coverage = []
    quants = []
    for exp_idx, exp_txt in enumerate(explanations):
        np.random.seed(1)
        exp_txt = exp_txt.lower()
        if_then = re.compile(r'if [\w\W\s]*, then [\w\s]*')
        # If the explanations are not of the form 'if .., then ..', skip for now
        if if_then.search(exp_txt) is None:
            faith_scores.append(0)
            continue

        if_then_then = re.compile(r'if [\w\W\s]*, then [\w\s<>]*, then [\w\s<>]*')
        if if_then_then.search(exp_txt):
            faith_scores.append(0)
            continue

        if ' or ' in exp_txt:
            # An extra test for 'or' because of the operation definitions >= and <=
            test_txt = copy.deepcopy(exp_txt)
            test_txt = test_txt.replace("greater than or equal to ", "")
            test_txt = test_txt.replace("lesser than or equal to ", "")
            if ' or ' in test_txt:
                cand_rule, cand_target = conjunctive_parse(exp_txt)
            elif ' and ' in test_txt:
                cand_rule, cand_target = conjunctive_parse(exp_txt)
            else:
                cand_rule, cand_target = simple_parse(exp_txt)
        elif ' and ' in exp_txt:
            cand_rule, cand_target = conjunctive_parse(exp_txt)
        else:
            cand_rule, cand_target = simple_parse(exp_txt)
        
        if cand_rule is None:
            faith_scores.append(0)
            continue 

        if isinstance(cand_rule, tuple):
            conj = cand_rule[1]
            cand_rule = cand_rule[0]
        
        quant = 1.0
        if any(f'it is {quant} ' in cand_target for quant in quantifiers):
            quant = [quant for quant in quantifiers if f'it is {quant} ' in cand_target][0]
            cand_target = cand_target.split(f"it is {quant} ")[-1]
            quant = QUANTIFIER_SCORE_MAP[quant]

        # If the target does not belong to the set of possible values,
        # this explanation cannot be evaluated.
        if cand_target not in task_lbls:
            faith_scores.append(0)
            continue
        
        # Let's assume that the function is going to give a list of candidate_rules ALWAYS!
        if not all(isinstance(crule, list) for crule in cand_rule):
            faith_scores.append(0)
            continue
        if any(crule[0] not in col_names for crule in cand_rule):
            faith_scores.append(0)
            continue

        if faith:
            with open(f'data/{dataset_name}/mantle_subsets/{model}/{exp_idx+1}/data.jsonl', 'r') as f:
                line = f.readline()
                batch = json.loads(line)['samples']
                # if 'budget_constraint' in path_prefix: batch = batch[:10]
        else:
            with open(f'data/{dataset_name}/mantle_subsets/{model}/test_data.jsonl', 'r') as f:
                line = f.readline()
                batch = json.loads(line)['samples']
        
        # make all the feature names lower
        batch = [{''.join(key.lower().split()): value for key, value in b.items()} for b in batch]
        if dataset_name == 'adult':
            targets = [b['income'].lower() if b['income']=='>50K' else 'not >50k' for b in batch]
        elif dataset_name == 'recidivism':
            targets = [b['recidivism'] for b in batch]
        elif dataset_name == 'travel_insurance':
            targets = [b['travelinsurance'] for b in batch]
        
        bs = len(batch)
        # Target check for negations
        target_op = '!=' if 'not not ' in cand_target else '=='

        col_evals = []
        for crule in cand_rule:
            data_type = type(batch[0][crule[0]])
            col_values = [b[crule[0]] for b in batch]
            if data_type == str: # not str.isdigit(crule[2]):
                col_values = [col_value.lower() for col_value in col_values]
                col_eval = [eval(f"str(col_val){crule[1]}crule[2]") for col_val in col_values]
            else:
                col_eval = [eval(f"col_val{crule[1]}{eval(crule[2])}") for col_val in col_values]

            col_evals.append(col_eval)
        col_evals = np.array(col_evals).T

        if col_evals.shape[-1] == 1:
            f = lambda x: x[0]
        elif col_evals.shape[-1] == 2:
            assert len(conj) == 1
            f = lambda x: eval(f'x[0]{conj[0]}x[1]')
        elif col_evals.shape[-1] == 3:
            assert len(conj) == 2
            f = lambda x: eval(f'x[0]{conj[0]}(x[1]{conj[1]}x[2])')
        else:
            raise ValueError('Not supposed to happen!')
        
        targets = np.array(targets)
        evaluate_col_evals = np.array(list(map(f, col_evals)))
        p = quant if cand_target == task_lbls[0] else 1.-quant
        predictions = np.random.choice(task_lbls, p=[p, 1.-p], size=np.sum(evaluate_col_evals))
        correct_instances = predictions == targets[evaluate_col_evals]
        flip_predictions = np.random.choice(task_lbls, p=[0.5, 0.5], size=evaluate_col_evals.shape[0]-np.sum(evaluate_col_evals))
        flip_correct_instances = flip_predictions == targets[~evaluate_col_evals]
        cands_pred = np.concatenate((flip_correct_instances, correct_instances), axis=0)
        assert cands_pred.shape[0] == col_evals.shape[0]
        faith_scores.append(np.mean(cands_pred))
        cov = correct_instances.shape[0] / cands_pred.shape[0]
        prec = np.mean(correct_instances) if cov > 0.0 else 1.0
        precision.append(prec)
        coverage.append(cov)
        quants.append(quant)
    
    return np.array(faith_scores), np.array(precision), np.array(coverage), np.array(quants)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp-file', required=True)
    parser.add_argument('-d', '--dataset-name', default='adult')
    parser.add_argument('-m', '--model-name', default='lr')
    args = parser.parse_args()

    with open(args.exp_file, 'r') as f:
        explanations = [line[:-1] for line in f.readlines()]
    
    print('\t\t\taccuracy\tprec.\tcov.\tprec.*cov.') # \tquants
    score, precision, coverage, quants = parse_eval(explanations, faith=True, model=args.model_name, dataset_name=args.dataset_name)
    print(f'Faithfulness Score:\t{np.mean(score):.3f} +- {np.std(score):.3f}\t{np.mean(precision):.3f}\t{np.mean(coverage):.3f}\t{np.mean(precision*coverage):.3f}')

    score, precision, coverage, quants = parse_eval(explanations, faith=False, model=args.model_name, dataset_name=args.dataset_name)
    print(f'Simulatability Score:\t{np.mean(score):.3f} +- {np.std(score):.3f}\t{np.mean(precision):.3f}\t{np.mean(coverage):.3f}\t{np.mean(precision*coverage):.3f}')