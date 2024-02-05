[![codecov](https://codecov.io/gh/Kaysera/teacher/branch/main/graph/badge.svg?token=QFA17A64EW)](https://codecov.io/gh/Kaysera/teacher)
[![GitHub - License](https://img.shields.io/github/license/Kaysera/teacher?logo=github&style=flat&color=green)](https://github.com/Kaysera/teacher/blob/main/LICENSE)
[![Lint](https://github.com/Kaysera/teacher/actions/workflows/linting.yml/badge.svg)](https://github.com/Kaysera/teacher/actions/workflows/linting.yml)
[![Continuous Integration](https://github.com/Kaysera/teacher/actions/workflows/integration.yml/badge.svg)](https://github.com/Kaysera/teacher/actions/workflows/integration.yml)
[![Continuous Deployment](https://github.com/Kaysera/teacher/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Kaysera/teacher/actions/workflows/python-publish.yml)
[![PyPi package](https://badge.fury.io/py/teacher-xai.svg)](https://pypi.org/project/teacher-xai/)

# TEACHER: Trustable Extensible Aggregator of CompreHensive Explainers

TEACHER is an open source Python Library that incorporates several state-of-the-art 
explainability techniques that can be used for model interpretation and explanation. 
The objective of the library is to be extensible with new tools and algorithms while 
keeping compatibility with the most used machine learning models such as scikit-learn.

This project was started in 2020 as the Ph.D. Thesis of Guillermo Tomás Fernández Martín,
whose advisors are José Antonio Gámez Martín and José Miguel Puerta Callejón.

Website: <https://xai-teacher.readthedocs.io/en/latest/>

## Installation

### Dependencies

Teacher requires:

    * Python (>=3.9)
    * scikit-learn 
    * scikit-fuzzy
    * matplotlib (for plotting functions)
    * deap (for compatibility with the LORE algorithm)
    * imblearn (for compatibility with the LORE algorithm)

**IMPORTANT** Install scikit-fuzzy from their [GitHub](https://github.com/scikit-fuzzy/scikit-fuzzy) as the PyPi version
is obsolete:
```Shell
pip install git+https://github.com/scikit-fuzzy/scikit-fuzzy
```

### User installation

If you already have a working installation, you can install teacher with 

```shell
pip install -U teacher-xai
```

The documentation includes more [detailed instructions](https://xai-teacher.readthedocs.io/en/latest/users/installing/index.html).

## Usage

For detailed instructions on how to use teacher, please refer to the [API Reference](https://xai-teacher.readthedocs.io/en/latest/api/index.html)

## Supported Methods

The following list summarizes the models and explainers currently supported
- **Fuzzy Factuals and Counterfactuals**: Explainer obtained from a fuzzy tree that can be used for global or local explanations
- **LORE**: Local explainer generated from a neighborhood
- **FLARE**: Fuzzy local explainer generated from a neighborhood
  
## Metrics

The following list summarizes the metrics and scores that can be extracted from the explainers

### General metrics
- **Coverage**: How many instances are covered by the rules forming the explanation
- **Precision**: How many of the instances covered by the rules forming the explanation are properly classified

### Local neighborhood metrics 
- **Fidelity**: How good is the local explainer at mimicking the global classifier in the neighborhood
- **L-fidelity (Rule fidelity)**: How good is the local explainer at mimicking the global classifier in the instances of the neighborhood covered by the factual explanation
- **Cl-fidelity**: How good is the local explainer at mimicking the global classifier in the instances of the neighborhood covered by the counterfactual explanation (To be implemented)
- **Hit**: Does the local explainer match the global classifier result? (To be implemented)
- **C-hit**: Does the local explainer match the global classifier result for tan instance built from the counterfactual rule? (To be implemented)

#### References and Examples
- Fuzzy Factuals and counterfactuals([Fernandez et al., 2022](https://doi.org/10.1109/TFUZZ.2022.3179582))
  - Documentation <https://xai-teacher.readthedocs.io/en/latest/>
  - Experiments: <https://github.com/Kaysera/teacher-experiments>
- LORE ([Guidotti et al., 2018](https://doi.org/10.1109/MIS.2019.2957223))
  - Documentation and examples: <https://doi.org/10.1109/MIS.2019.2957223>
- FLARE ([Fernandez et al., 2023 preprint](https://dsi.uclm.es/descargas/technicalreports/DIAB-24-02-1/FLARE_Tech_Rep.pdf))