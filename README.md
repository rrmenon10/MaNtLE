# MaNtLE: Model-agnostic Natural Language Explainer

Code for the EMNLP 2023 paper: [MaNtLE: Model-agnostic Natural Language Explainer](https://arxiv.org/abs/2305.12995)

## Environment Setup

Setup the environment and dependencies with the following command:
`bash bin/init.sh`

Next, each time you access this repository, make sure to run:
`source bin/setup.sh`
This allows the model to access the internal directories.

Link to download pre-trained model: [Google Drive Link](https://drive.google.com/drive/folders/1KHFlBtT1ZO845J8so9zMFEylsAMtZ2vZ?usp=sharing). Please place the contents at this link inside a folder named `pretrained_mantle` in the root of this repository.

Codes:

`create_model_explanations.py` parititons the dataset into train, val and test. It also trains a specified model and creates explanations for all the examples in the validation set.

`create_subsets.py` parititons the dataset into train, val and test. It also trains a specified model and creates explanations for all the examples in the validation set. It also converts the subset data into json files to be used with mantle.

`mantle_explanations.py` computes the mantle explanations.

`evaluate_mantle.py` evaluates mantle using the semantic parser. Note: parts of this code is still hard-coded for the adult dataset. But, this can be updated.

## Run MaNtLE

You can run MaNtLE for any of the datasets provided using the command: `bash bin/run_dataset.sh {dataset_name}`, where `{dataset_name}` is one of 'adult', 'recidivism', or 'travel_insurance'

## Contact ##

For any doubts or questions regarding the work, please contact Rakesh ([rrmenon@cs.unc.edu](mailto:rrmenon+mantle@cs.unc.edu)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

## Citation ##

Please cite us if MaNtLE is useful in your work:

```
@inproceedings{menon2023mantle,
          title={MaNtLE: Model-agnostic Natural Language Explainer},
          author={Menon, Rakesh R and Zaman, Kerem and Srivastava, Shashank},
          journal={Empirical Methods in Natural Language Processing (EMNLP)},
          year={2023}
}
```