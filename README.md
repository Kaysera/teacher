[![codecov](https://codecov.io/gh/Kaysera/teacher/branch/main/graph/badge.svg?token=QFA17A64EW)](https://codecov.io/gh/Kaysera/teacher)
![GitHub - License](https://img.shields.io/github/license/Kaysera/teacher?logo=github&style=flat&color=green)
# TEACHER: Trustable Extensible Aggregator of CompreHensive Explainers

TEACHER is an open source Python Library that incorporates several state-of-the-art 
explainability techniques that can be used for model interpretation and explanation. 
The objective of the library is to be extensible with new tools and algorithms while 
keeping compatibility with the most used machine learning models such as scikit-learn.

This project was started in 2020 as the Ph.D. Thesis of Guillermo Tomás Fernández Martín,
whose advisors are José Antonio Gámez Martín and José Miguel Puerta Callejón.

Website: TBD

## Installation

### Dependencies

Teacher requires:

    * Python (>=3.9)
    * scikit-learn 
    * scikit-fuzzy
    * matplotlib (for plotting functions)
    * deap (for compatibility with the LORE algorithm)
    * imblearn (for compatibility with the LORE algorithm)

### User installation

If you already have a working installation, you can install teacher with 

```
TBD
```

The documentation includes more detailed instructions.

## Usage

TBD

## Supported Methods

The following list summarizes the models and explainers currently supported
- **Anchors**: Future work
- **Bayesian Networks**: Future work
- **Fuzzy Factuals and Counterfactuals**: Explainer obtained from a fuzzy tree that can be used for global or local explanations
- **LIME**: Future work
- **LORE**: Local explainer generated from a neighborhood
- **SHAP**: Future work
  
## Metrics

The following list summarizes the metrics and scores that can be extracted from the explainers

### General metrics
- **Coverage**: How many instances are covered by the rules forming the explanation
- **Precision**: How many of the instances covered by the rules forming the explanation are properly classified

### Local neighborhood metrics (To be implemented)
- **Fidelity**: How good is the local explainer at mimicking the global classifier in the neighborhood
- **L-fidelity**: How good is the local explainer at mimicking the global classifier in the instances of the neighborhood covered by the factual explanation
- **Cl-fidelity**: How good is the local explainer at mimicking the global classifier in the instances of the neighborhood covered by the counterfactual explanation
- **Hit**: Does the local explainer match the global classifier result?
- **C-hit**: Does the local explainer match the global classifier result for tan instance built from the counterfactual rule?

#### References and Examples
- Fuzzy Factuals and counterfactuals([ref])
  - Documentation (TBD)
  - Examples: Beer (TBD)
- LORE ([Guidotti et al., 2018](https://arxiv.org/pdf/1805.10820.pdf))
  - Documentation (TBD)
  - Examples: (TBD)
