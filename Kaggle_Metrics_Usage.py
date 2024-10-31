"""

Tasks
Regression
Classification
Ordinal
    order is important
    e..g predicting magnitude of an earthquake
    the most common way is treat it as multiclass problem
          here the prediction will not take into account that the classes have certain order
          u get a feeling that there is problem if u look at the prediction probability for the classes
          u often get a asymtreic distribution whereas u shd get a gaussian distribution around the max distribution probability class
    the other way is to treat it as a regression and post process ur result.
        the order will be taken into account but some soesticated post processing may be needed as this might lead to inaccuracies


Metrics
AUC , Logloss - classification binary
MAP@K - recomender systems
RMSE, RMSLogError , Quadratic weightd kappa - regression

Handling never seen metrics before
    1) Kaggle Discussion forums
    2) Try to experiment with it by coding the evaluation function; how metric reacts to different types of errors 
            Page 107 from kaggle book; sample articles  
            1) https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho
            2) https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa
            3) https://www.kaggle.com/code/rohanrao/osic-understanding-laplace-log-likelihood
                        
"""