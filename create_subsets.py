import argparse
import os
import json
import numpy as np
import dill as pickle

import utils

def reformat_data(array, dataset_utils):
    reformatted_data = []
    for feature in range(array.shape[1]):
        if feature in dataset_utils.categorical_features:
            mapped_data = map_array_values(array[:,feature], dataset_utils.categorical_names[feature])
        else:
            mapped_data = array[:,feature].astype(np.int64).tolist()
        reformatted_data.append(mapped_data)
    array = np.array(reformatted_data).T
    return array

def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy().astype(np.int64).astype(np.string_)
    for src, target in enumerate(value_map):
        if isinstance(ret[0], np.bytes_):
            ret = [r.decode() for r in ret]
        ret = [target if r==str(src) else r for r in ret]
    return ret

def jsonify(data, labels, feature_names, class_names, target_class_name=''):

    samples = [{feat: d[idx] for idx, feat in enumerate(feature_names)} for d in data]
    samples = [{**d, **{target_class_name: class_names[lbl]}} for d, lbl in zip(samples, labels)]
    return samples

def save_data(data, labels, sub_idx, dataset_name, model_name):

    # Save the data for each subset
    os.makedirs(f'data/{dataset_name}/subsets/{model_name}/{sub_idx+1}', exist_ok=True)
    pickle.dump(data, open(f'data/{dataset_name}/subsets/{model_name}/{sub_idx+1}/data.pkl', 'wb'))
    pickle.dump(labels, open(f'data/{dataset_name}/subsets/{model_name}/{sub_idx+1}/labels.pkl', 'wb'))

def dump_mantle_data(data, labels, dataset, class_names, target_class_name, save_loc=''):

    data = reformat_data(data, dataset)
    samples = jsonify(data, labels, dataset.feature_names, class_names, target_class_name)
    with open(save_loc, 'w') as f:
        f.write(json.dumps({"samples": samples}) + '\n')

def create_subsets(num_subsets=100, num_samples_per_subset=10, dataset_name='adult', model_name='lr'):

    np.random.seed(0)

    # Load datasets
    dataset = utils.data_utils.load_data(dataset_name)
    # Create classifier and save predictions
    utils.model_utils.model_train_and_save(model_name, dataset, dataset_name)
    if dataset_name == 'adult':
        class_names = ['not >50K', '>50K']
        target_class_name = 'Income'
    elif dataset_name == 'recidivism':
        class_names = ['not commit', 'commit']
        target_class_name = 'Recidivism'
    elif dataset_name == 'travel_insurance':
        class_names = ['not interested', 'interested']
        target_class_name = 'Travel Insurance'
    else:
        raise NotImplementedError('The class names for this dataset has not been specified into MaNtLE format (x; not x)!')

    data_idx = np.arange(dataset.validation.shape[0])
    model_labels = pickle.load(open(f'data/{dataset_name}/dev/{model_name}_labels.pkl', 'rb'))

    subset_idxs = []
    while len(subset_idxs) < num_subsets:
        sample_idxs = np.random.choice(data_idx, size=num_samples_per_subset, replace=False)
        sub_labels = model_labels[sample_idxs]

        # Both classes need to be represented sufficiently in the batch in order to draw distinctions
        # with global explanations
        if np.sum(sub_labels) < int(0.3 * num_samples_per_subset) or np.sum(sub_labels) > int(0.7 * num_samples_per_subset):
            continue
        subset_idxs.append(sample_idxs)

    # dump subsets into MaNtLE readable format.
    for sub_idx, subset in enumerate(subset_idxs):

        sub_data = dataset.validation[subset]
        sub_labels = model_labels[subset]
        save_data(sub_data, sub_labels, sub_idx, dataset_name, model_name)

        # Converting dataset into format suitable for MaNtLE (np array to jsonl)
        os.makedirs(f'data/{dataset_name}/mantle_subsets/{model_name}/{sub_idx+1}', exist_ok=True)
        dump_mantle_data(sub_data, sub_labels, dataset, class_names, target_class_name,
                         save_loc=f'data/{dataset_name}/mantle_subsets/{model_name}/{sub_idx+1}/data.jsonl')

    # dump test_data into MaNtLE readable format.
    test_data = pickle.load(open(f'data/{dataset_name}/test/data.pkl', 'rb'))
    test_labels =  pickle.load(open(f'data/{dataset_name}/test/{model_name}_labels.pkl', 'rb'))
    dump_mantle_data(test_data, test_labels, dataset, class_names, target_class_name,
                     save_loc=f'data/{dataset_name}/mantle_subsets/{model_name}/test_data.jsonl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create subsets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset-name', choices=['adult', 'travel_insurance', 'recidivism'], required=True, help='Dataset name')
    parser.add_argument('--model-name', choices=['lr', 'dt', 'nn', 'xgb'], required=True, help='Model name')
    parser.add_argument('--num-subsets', type=int, default=100, required=False, help='Number of subsets')
    parser.add_argument('--num-examples-per-subset', type=int, default=10, required=False, help='Numer of examples per subset')
    args = parser.parse_args()

    create_subsets(args.num_subsets, args.num_examples_per_subset, args.dataset_name, args.model_name)
