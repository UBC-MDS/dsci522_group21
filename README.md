# Bank Marketing Campaign on Term Deposit Analysis

  - Authors: Jerry Yu, John Shiu, Sophia Zhao, Zeily Garcia
  - Contributors: 

An analysis project investigating the effectiveness of direct marketing campaigns on term deposit of a Portuguese banking institution.

## About

This project analyzes the success rate of bank marketing campaigns in selling term deposits. Using a dataset from the UCI repository with 45,211 instances and 16 features, we aim to predict whether a client will subscribe to a term deposit. Our predictive models, using `altair` visualizations, aim to improve targeting efficiency and ultimately the success rate of the campaigns.

The dataset includes demographic information, contact details, and campaign outcomes. It is fully detailed, with categorical and integer types, and is used for classification tasks.

## Report

The complete report can be found [here]([Link to the report]).

## Usage

To replicate the analysis:

1. Clone this repository.
2. Install the required dependencies.
3. Follow the instructions in the File Consumption Order section.

## Dependencies

- python
    - ipython
    - ipykernel
    - matplotlib>=3.8.0
    - pandas>=2.1.1
    - scikit-learn>=1.3.1    
    - graphviz
    - python-graphviz
    - altair=5.1.2
    - vl-convert-python  # For saving altair charts as static images
    - vegafusion  # For working with charts > 5,000 graphical objects
    - vegafusion-python-embed  # Same as the previous one
    - vegafusion-jupyter  # For working with charts > 100,000 graphical objects
    - vega_datasets  # Example data 
    - pip>=23.2.1    
    - pip:
        - mglearn
        - psutil>=5.7.2

## File Consumption Order

1. Datasets: `bank-full.csv`, `bank-names.csv`, `bank.csv`, `bank-additional-full.csv`, `bank-additional-names.csv`, `bank-additional.csv`
2. Scripts: `term_deposit_analysis.ipynb`
3. Report: `term_deposit_analysis.pdf`

## License

This analysis is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please provide proper attribution when using or modifying the analysis.  

# References

<div id="refs" class="references hanging-indent">

<div id="ref-Moro">

Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>.

</div>

<div id="ref-Suman">

Bera, Suman Kalyan et al. “Fair Algorithms for Clustering.” Neural Information Processing Systems (2019). <https://www.semanticscholar.org/paper/Fair-Algorithms-for-Clustering-Bera-Chakrabarty/34a46c62cb3a7809db4ed7d0c1a651f538b9fe87#citing-papers>.

</div>

<div id="ref-Ziko">

Ziko, Imtiaz Masud et al. “Clustering with Fairness Constraints: A Flexible and Scalable Approach.” ArXiv abs/1906.08207 (2019): n. pag. <https://www.semanticscholar.org/paper/Clustering-with-Fairness-Constraints%3A-A-Flexible-Ziko-Granger/d56841fe68f2a913583a40edf541efeaed0a7e5b>.

</div>

<div id="ref-Lamy">

Lamy, Alexandre Louis et al. “Noise-tolerant fair classification.” ArXiv abs/1901.10837 (2019): n. pag. <https://www.semanticscholar.org/paper/Noise-tolerant-fair-classification-Lamy-Zhong/c4ac496bf57410638260196a25d8ae3366ea03c7>.

</div>

<div id="ref-Iosifidis">

Iosifidis, Vasileios and Eirini Ntoutsi. “AdaFair: Cumulative Fairness Adaptive Boosting.” Proceedings of the 28th ACM International Conference on Information and Knowledge Management (2019): n. pag.<https://www.semanticscholar.org/paper/AdaFair%3A-Cumulative-Fairness-Adaptive-Boosting-Iosifidis-Ntoutsi/18fe4800f3c85f315d79063d6b0fe38c7610ad45>.

</div>

<div id="ref-Lamy">

Vaz, Afonso Fernandes et al. “Quantification under prior probability shift: the ratio estimator and its extensions.” ArXiv abs/1807.03929 (2018): n. pag.<https://www.semanticscholar.org/paper/Quantification-under-prior-probability-shift%3A-the-Vaz-Izbicki/50adf7b8fd1274149a195ef4a7b4ab9f84b3dd13>.

</div>
