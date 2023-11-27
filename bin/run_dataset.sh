#!/usr/bin/env bash

set -exu

dataset=$1

models="lr dt nn xgb"
results_dir="simulation_results/${dataset}"

mkdir -p ${results_dir}

for model in ${models}; do
    
    # Create the subsets for the dataset
    python create_subsets.py --dataset-name ${dataset} --model-name ${model}

    # Compute and evaluate the MaNtLE explanations
    python mantle_explanations.py --dataset-name ${dataset} --model-name ${model}
    python postprocess_mantle.py --dataset-name ${dataset} --model-name ${model} --explanations-file data/${dataset}/mantle_subsets/${model}/mantle_explanations.txt
    OUT_FILE="${results_dir}/${model}_mantle.txt"
    python evaluate_mantle.py --dataset-name ${dataset} --model-name ${model} --exp-file data/${dataset}/mantle_subsets/${model}/mantle_explanations.txt > ${OUT_FILE}

    # Compute the MaNtLE beam-search variant explanations
    python mantle_beam_explanations.py --dataset-name ${dataset} --model-name ${model}
    python postprocess_mantle.py --dataset-name ${dataset} --model-name ${model} --explanations-file data/${dataset}/mantle_subsets/${model}/mantle_beam_explanations.txt
    OUT_FILE="${results_dir}/${model}_mantle_beam.txt"
    python evaluate_mantle.py --dataset-name ${dataset} --model-name ${model} --exp-file data/${dataset}/mantle_subsets/${model}/mantle_beam_explanations.txt > ${OUT_FILE}

    # Compute the MaNtLE per-feature variant explanations
    python mantle_perfeat_explanations.py --dataset-name ${dataset} --model-name ${model}
    python postprocess_mantle.py --dataset-name ${dataset} --model-name ${model} --explanations-file data/${dataset}/mantle_subsets/${model}/mantle_perfeat_explanations.txt
    OUT_FILE="${results_dir}/${model}_mantle_perfeat.txt"
    python evaluate_mantle.py --dataset-name ${dataset} --model-name ${model} --exp-file data/${dataset}/mantle_subsets/${model}/mantle_perfeat_explanations.txt > ${OUT_FILE}

done