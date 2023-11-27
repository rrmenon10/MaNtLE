import re
import operator
import functools

from itertools import zip_longest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def cond2LF(sent_rule):

    ops = re.compile(r'\snot equal to\s|\sequal to\s|\snot greater than\s|\sgreater than or equal to\s|\sgreater than\s|\slesser than or equal to\s|\snot lesser than\s|\slesser than\s')
    string_to_op_map = {"equal to": "==",
                        "greater than": ">",
                        "greater than or equal to": ">=",
                        "lesser than": "<",
                        "lesser than or equal to": "<=",
                        "not equal to": "!=",
                        "not greater than": "<=",
                        "not lesser than": ">="}
    op_search = ops.search(sent_rule)
    if op_search is not None: 
        op = op_search.group()[1:-1]
        sent_rule = sent_rule.split(f" {op} ")
        sent_rule.insert(1, string_to_op_map[op])

        # removing stop words from target and op
        # stop_words = set(stopwords.words('english'))
        
        word_tokens = word_tokenize(sent_rule[0]) # for column_name
        sent_rule[0] = " ".join([w for w in word_tokens]) #  if not w.lower() in stop_words

        word_tokens = word_tokenize(sent_rule[2]) # for column_value
        sent_rule[2] = " ".join([w for w in word_tokens])
    
    return sent_rule

def simple_parse(sent):
    '''
    :return:
    Computes the semantic parse of an explanation into a rule and target.
    '''
    if len(sent.split(", then ")) > 2:
        sent = ", then ".join(sent.split(", then ")[:2])
    sent_rule, sent_target = sent.split(", then ")
    
    # Remove if from the front
    sent_rule = " ".join(sent_rule.split()[1:])
    sent_rule = cond2LF(sent_rule)

    # Remove period from sent_target if present
    if len(sent_target) > 0 and sent_target[-1] == '.': sent_target = sent_target[:-1]

    return [sent_rule], sent_target

def conjunctive_parse(sent):
    '''
    :return:
    Computes the semantic parse of a conjunctive explanation
    '''
    assert any(conj in sent.lower() for conj in [' and ', ' or ']), 'Sentences need to have conjunction(s) to use this function!'
    if len(sent.split(", then ")) > 2:
        sent = ", then ".join(sent.split(", then ")[:2])
    sent_rules, sent_target = sent.split(", then ")
    
    # remove if from the front
    sent_rules = " ".join(sent_rules.lower().split()[1:])
    if sent_rules not in sent: return None, None # accounting for case where there is some spacing mismatch. we reject such exps.
    sent_rules = sent_rules.split(" and ")
    replaced=False
    for i, sent_rule in enumerate(sent_rules):
        if 'than or equal to' in sent_rule:
            sent_rule = sent_rule.replace('than or equal to ', 'than xxx ')
            replaced = True
        sent_rules = sent_rules[:i] + sent_rule.split(' or ')
    
    if replaced:
        sent_rules = [[sent_rule.replace('than xxx ', 'than or equal to ')] for sent_rule in sent_rules]
    else:
        sent_rules = [[sent_rule.replace('than xxx ', 'than or equal to ')] for sent_rule in sent_rules]
    
    sent_rules = functools.reduce(operator.iconcat, sent_rules, [])
    sent_rules = [sent_rule.split("( ")[-1].split(" )")[0] for sent_rule in sent_rules]
    conj_in_rules = [' and ' if ' and ' in sent.lower().split(sent_rules[idx])[1][:5] else ' or ' for idx in range(len(sent_rules)-1)]
    sent_rules = [cond2LF(sent_rule) for sent_rule in sent_rules]
    # sent_rules = list(functools.reduce(operator.add, zip_longest(sent_rules, conj_in_rules), )[:-1])
    
    # Remove period from sent_target if present
    if len(sent_target) > 0 and sent_target[-1] == '.': sent_target = sent_target[:-1]

    return (sent_rules, conj_in_rules), sent_target