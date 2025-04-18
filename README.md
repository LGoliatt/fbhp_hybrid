<p align="center"><h1 align="center">Development of hybrid computational data-intelligence model for flowing bottom-hole pressure of oil wells: New strategy for oil reservoir management and monitoring</h1></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/LGoliatt/fbhp_hybrid?style=default&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
	<a href="https://github.com/ITMO-NSS-team/Open-Source-Advisor"><img src="https://img.shields.io/badge/improved%20by-OSA-blue"></a>
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=default&logo=GNU-Bash&logoColor=white"alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white"alt="Python">
</p>
<br>


---
## Overview

<overview>
Among several metric parameters concerning the assessment of oil and gas well production, the flowing bottom-hole pressure (FBHP) is considered essential. Accurate prediction of FBHP is crucial for petroleum engineering and management. Several related parameters are associated with the FBHP magnitude influence; thus, proper inspection of those parameters is another vital concern. This research proposes a hybrid modeling framework based on the hybridization of machine learning (ML) models (i.e., Extreme Learning Machine (ELM), Support Vector Machine Regressor (SVR), Extreme Gradient Boosting (XGB), and Multivariate Adaptive Regression Spline (MARS)) and nature-inspired Differential Evolutional (DE) optimization for FBHP prediction. The adjustment of the internal parameters of the ML-based models and the input feature selection is formulated as an incremental learning problem that is solved by the evolutionary algorithm. Problem-specific samples were collected from the open-source literature for this investigation. Modeling results are adaptable, automatically determining the most relevant variables for the context of the ML model. The adaptive polynomial structure of hybridized MARS model attained the best average performance for the FBHP modeling with correlation (R = 0.94) and minimum root mean square (RMSE = 97.88). The proposed modeling framework produces an alternative efficient computer aid model for FBHP prediction, resulting in reliable automated technology to assist oil and gas well management.

https://doi.org/10.1016/j.fuel.2023.128623
 
</overview>

---


## Table of contents

- [Core features](#core-features)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Getting started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Core features

<corefeatures>

1. **Hybrid ML Modeling**: Predicts bottom-hole pressure using combined machine learning techniques.
2. **Data Handling**: Extracts, cleans, and prepares raw data for model input.
3. **Automated Updates**: `gitupdate.sh` script automates codebase synchronization.
4. **LaTeX Reporting**: Generates reports with training/testing results via LaTeX.
5. **Well Performance Analysis**: Supports reservoir management through pressure estimation.

</corefeatures>

---


## Installation

Install fbhp_hybrid using one of the following methods:

**Build from source:**

1. Clone the fbhp_hybrid repository:
```sh
❯ git clone https://github.com/LGoliatt/fbhp_hybrid
```

2. Navigate to the project directory:
```sh
❯ cd fbhp_hybrid
```

3. Install the project dependencies:

echo 'INSERT-INSTALL-COMMAND-HERE'


---


## Examples

Examples of how this should work and how it should be used are available in [Not found any examples](https://github.com/LGoliatt/fbhp_hybrid/tree/main/).

---



## Getting started

### Usage

Run fbhp_hybrid using the following command:
 
 echo 'INSERT-RUN-COMMAND-HERE'

---


## Contributing


- **[Report Issues](https://github.com/LGoliatt/fbhp_hybrid/issues )**: Submit bugs found or log feature requests for the fbhp_hybrid project.


---


## License

This project is protected under the Not found any License. For more details, refer to the [LICENSE](https://github.com/LGoliatt/fbhp_hybrid/blob/main/) file.

---


## Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---


## Citation

If you use this software, please cite it as below.

### APA format:

    LGoliatt (2022). fbhp_hybrid repository [Computer software]. https://github.com/LGoliatt/fbhp_hybrid

### BibTeX format:

    @misc{fbhp_hybrid,

        author = {LGoliatt},

        title = {fbhp_hybrid repository},

        year = {2022},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/LGoliatt/fbhp_hybrid.git}},

        url = {https://github.com/LGoliatt/fbhp_hybrid.git}

    }

---
