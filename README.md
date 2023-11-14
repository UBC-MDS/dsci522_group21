# Identifying the Top Three Predictors of Term Deposit Subscriptions Analysis

  - Authors: Jerry Yu, John Shiu, Sophia Zhao, Zeily Garcia
  - Contributors: 

This repository contains the code and analysis for the project "Identifying the Top Three Predictors of Term Deposit Subscriptions". Our team has explored a dataset from a Portuguese bank's marketing campaigns to understand what factors contribute to clients' decisions to subscribe to term deposits.

## About

This repository documents our analysis on the determinants of term deposit subscriptions within a Portuguese bank, harnessing a dataset that tracks 45,211 client interactions across 17 distinct input features. We have implemented logistic regression and decision tree classifiers to unearth the principal three factors that predict a client's propensity to subscribe to a term deposit. The data preprocessing phase was meticulous, involving the rectification of missing entries, encoding of categorical variables, and normalization of numerical variables to prepare for robust analysis.

The crux of our exploratory data analysis was the strategic use of visualizations to unravel the nuances in feature distributions and inter-feature correlations. Our model evaluation was meticulously tailored to emphasize precision and recall, a decision dictated by the inherent class imbalance present within the dataset. In this rigorous analytical process, logistic regression emerged as a marginally more precise model compared to the decision tree classifier.

Significantly, the analysis culminated in pinpointing the outcome of prior marketing campaigns, the timing of client contact within the year, and the duration of the calls as pivotal indicators of subscription likelihood. These insights not only shed light on the client's decision-making dynamics but also carve out potential avenues for further scholarly inquiry and practical application in marketing strategies for banking products.

## Insights and Future Directions

The report's findings are instrumental for banking institutions to comprehend and predict customer behavior concerning term deposit subscriptions. This predictive understanding is essential for refining marketing approaches and enhancing the efficiency of future campaigns. We believe that these insights can serve as a cornerstone for further research, potentially exploring more sophisticated analytical models and integrating additional datasets to delve into the observed seasonal patterns and other underlying phenomena influencing client decisions.

## Report

The complete report can be found [here](https://github.com/UBC-MDS/dsci522_group21/tree/report/src).

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
3. Report: `term_deposit_analysis.html`

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.  

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
