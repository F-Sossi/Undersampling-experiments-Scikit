### Imbalance class classification (100 points)

Q1. Given the Porto Seguro dataset from Kaggle (sourceLinks to an external site.), the goal is to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. In this task, you will predict the probability that an auto insurance policy holder files a claim. Here is the description of the data from the source:

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.  

The target class is imbalanced so you will need to use techniques for handling imbalanced data. Build the following classifiers 

(i) Logistic Regression 

(ii) Naïve Bayes 

(iii) Random Forest (with the bullet point setting below)

(iv) Balanced Random Forest (https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)

(v) AdaBoost with random undersampling (https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.htmlLinks to an external site.)

(vi) Gradient Boost classifier



```
├── CSS581_Hw3.ipynb
├── data
│   ├── porto-seguro-safe-driver-prediction.zip
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
└── README.md
```
kaggle competitions download -c porto-seguro-safe-driver-prediction

 

You can use the imblearn in Python for resampling techniques.  

Report your performance on Precision, Recall, F-Score, AUC when using

No re-balancing techniques
Random Sampling to balance classes via under-sampling
Random Sampling to balance classes via over-sampling
Under-sampling: TomekLinks
Oversampling: SMOTE
 

Given the results, comment on the following:

1. How do the under-sampling techniques fare as compared to the oversampling techniques?
2. How does Logistic Regression fare as compared to other models when there is 
   - (i) no re-balancing 
   - (ii) oversampling 
   - (iii) under-sampling. 
   - What is your hypothesis regarding its performance?
   
3. Comment on the performance of random forest vs. its balanced version
4. Build an ensemble classifier model using the first classifiers from Q1 and VotingClassifier. Report the results of this ensemble classifier and comment on whether it performs better than Adaboost and Gradient boost. 


## Current results
```
# No Sampling naive
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression: Precision=0.053006016424399456, Recall=0.5563493892601982, F1-Score=0.09679036105932118, AUC=0.6274003303704707
Naive Bayes: Precision=0.05364053318045005, Recall=0.34501037105323806, F1-Score=0.09284584612522095, AUC=0.5859438630115047
Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5806530281125448
Balanced Random Forest: Precision=0.06173015646549674, Recall=0.29914726895598065, F1-Score=0.1023417172593235, AUC=0.6185823341571003
RUSBoost: Precision=0.05448766005937566, Recall=0.41876008296842593, F1-Score=0.09642838189247997, AUC=0.599242599752571
Gradient Boost: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.6385989613805131

# With Sampling naive
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Resampling: Random Under-Sampling, Model: Logistic Regression: Precision=0.05177223675958565, Recall=0.551740032265499, F1-Score=0.09466192170818505, AUC=0.6226078472357891
Resampling: Random Under-Sampling, Model: Naive Bayes: Precision=0.0515572452112882, Recall=0.406314819082738, F1-Score=0.09150360720402761, AUC=0.5827413054484701
Resampling: Random Under-Sampling, Model: Random Forest: Precision=0.051250992022054215, Recall=0.5655681032495967, F1-Score=0.0939851783765152, AUC=0.6179168548252828
Resampling: Random Under-Sampling, Model: Balanced Random Forest: Precision=0.051247997301627456, Recall=0.5602673427056926, F1-Score=0.09390632544664414, AUC=0.6181216325340748
Resampling: Random Under-Sampling, Model: RUSBoost: Precision=0.04884481903817551, Recall=0.5676423138972113, F1-Score=0.08994960192827406, AUC=0.5972225873457558
Resampling: Random Under-Sampling, Model: Gradient Boost: Precision=0.054035420347309, Recall=0.5780133671352846, F1-Score=0.09883159616180325, AUC=0.6361519322570472
Resampling: Random Over-Sampling, Model: Logistic Regression: Precision=0.05265812790595666, Recall=0.5533533072136437, F1-Score=0.09616501451887453, AUC=0.6272277708229502
Resampling: Random Over-Sampling, Model: Naive Bayes: Precision=0.05270646645676919, Recall=0.4012445263885688, F1-Score=0.09317385138208771, AUC=0.5853415883381918
Resampling: Random Over-Sampling, Model: Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5922819398494023
Resampling: Random Over-Sampling, Model: Balanced Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.6013491732267277
Resampling: Random Over-Sampling, Model: RUSBoost: Precision=0.052752293577981654, Recall=0.5300760543904125, F1-Score=0.09595527649721522, AUC=0.612683148643469
Resampling: Random Over-Sampling, Model: Gradient Boost: Precision=0.054576879611779566, Recall=0.5676423138972113, F1-Score=0.09957952615832456, AUC=0.6386436046811238
Resampling: Tomek Links, Model: Logistic Regression: Precision=0.05302066423902075, Recall=0.5570407928094031, F1-Score=0.09682523785678518, AUC=0.6274435651939346
Resampling: Tomek Links, Model: Naive Bayes: Precision=0.053548433252253544, Recall=0.34501037105323806, F1-Score=0.09270784951230841, AUC=0.5858164902639791
Resampling: Tomek Links, Model: Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5831154922166709
Resampling: Tomek Links, Model: Balanced Random Forest: Precision=0.06346866977274888, Recall=0.3076745793961742, F1-Score=0.10522996886454104, AUC=0.617282427290994
Resampling: Tomek Links, Model: RUSBoost: Precision=0.048468140848925645, Recall=0.5713297994929707, F1-Score=0.08935587355368922, AUC=0.6098397718406865
Resampling: Tomek Links, Model: Gradient Boost: Precision=0.5, Recall=0.00023046784973496196, F1-Score=0.00046072333563694994, AUC=0.6389516151250563
Resampling: SMOTE, Model: Logistic Regression: Precision=0.049520898727903516, Recall=0.5526619036644388, F1-Score=0.09089703011580083, AUC=0.6057042006390283
Resampling: SMOTE, Model: Naive Bayes: Precision=0.04047780329439754, Recall=0.5646462318506569, F1-Score=0.07554034471063423, AUC=0.5460905450363722
Resampling: SMOTE, Model: Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5823771849720045
Resampling: SMOTE, Model: Balanced Random Forest: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5904238350049512
Resampling: SMOTE, Model: RUSBoost: Precision=0.13157894736842105, Recall=0.00115233924867481, F1-Score=0.0022846698652044784, AUC=0.6126212741017564
Resampling: SMOTE, Model: Gradient Boost: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5679606578617811

# Ensemble 1 naive
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Ensemble Classifier: Precision=0.1068264721208963, Recall=0.04724590919566721, F1-Score=0.0655161393416427, AUC=0.6238781690494698

# Ensemble 2 With undersampling Naive
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Ensemble Classifier UnderSampling: Precision=0.234375, Recall=0.0034570177460244295, F1-Score=0.006813536225300931, AUC=0.6235592363776189

# Early results with tuning 10 itterations
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Best parameters for Logistic Regression: {'penalty': 'l2', 'C': 100}
Logistic Regression: Precision=0.05296675304141596, Recall=0.5558884535607282, F1-Score=0.09671792609820157, AUC=0.627394162004078

Best parameters for Naive Bayes: {'var_smoothing': 1e-05}
Naive Bayes: Precision=0.05365906078733388, Recall=0.345240838902973, F1-Score=0.09288194444444443, AUC=0.5859999911271967

Best parameters for Random Forest: {'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 8, 'n_estimators': 171}
Random Forest: Precision=0.053060841848692225, Recall=0.5217792117999539, F1-Score=0.09632607909460293, AUC=0.6259227996677587

Best parameters for Balanced Random Forest: {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 6, 'n_estimators': 64}
Balanced Random Forest: Precision=0.052540006443990976, Recall=0.563724360451717, F1-Score=0.09612135025739774, AUC=0.6236933893050766

Best parameters for RUSBoost: {'learning_rate': 1, 'n_estimators': 142}
RUSBoost: Precision=0.04846671367855305, Recall=0.5860797418760083, F1-Score=0.08952964371215322, AUC=0.610634026369457

Best parameters for Gradient Boost: {'learning_rate': 0.01, 'max_depth': 4, 'min_samples_leaf': 8, 'min_samples_split': 9, 'n_estimators': 180}
Gradient Boost: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.6273889138695449

# Early results with tuning 20 itterations
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Best parameters for Logistic Regression: {'penalty': 'l2', 'C': 100}
Logistic Regression: Precision=0.05296675304141596, Recall=0.5558884535607282, F1-Score=0.09671792609820157, AUC=0.627394162004078

Best parameters for Naive Bayes: {'var_smoothing': 1e-05}
Naive Bayes: Precision=0.05365906078733388, Recall=0.345240838902973, F1-Score=0.09288194444444443, AUC=0.5859999911271967

Best parameters for Random Forest: {'max_depth': 8, 'min_samples_leaf': 7, 'min_samples_split': 5, 'n_estimators': 89}
Random Forest: Precision=0.05307105767881368, Recall=0.5229315510486288, F1-Score=0.09636251672223048, AUC=0.6244503885082281

Best parameters for Balanced Random Forest: {'max_depth': 1, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 104}
Balanced Random Forest: Precision=0.05032731426562981, Recall=0.5280018437427979, F1-Score=0.09189546940494575, AUC=0.6099895445988722

Best parameters for RUSBoost: {'learning_rate': 1, 'n_estimators': 171}
RUSBoost: Precision=0.04925538803070946, Recall=0.490896519935469, F1-Score=0.08952777252380051, AUC=0.5904665072412524

# With sampling and tuning 20 itterations
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Resampling: Random Under-Sampling, Model: Logistic Regression: Precision=0.0526076901463246, Recall=0.5609587462548974, F1-Score=0.09619412717859542, AUC=0.6231023975182384
Resampling: Random Under-Sampling, Model: Naive Bayes: Precision=0.05196168899354059, Recall=0.3763539986171929, F1-Score=0.09131577475815021, AUC=0.5823320453891465
Resampling: Random Under-Sampling, Model: Random Forest: Precision=0.05165899418049162, Recall=0.5482830145194746, F1-Score=0.09442162290885278, AUC=0.6239971502066897
Resampling: Random Under-Sampling, Model: Balanced Random Forest: Precision=0.04898408179112204, Recall=0.5233924867480987, F1-Score=0.08958403187313861, AUC=0.6023511751208139
Resampling: Random Under-Sampling, Model: RUSBoost: Precision=0.04145438176498271, Recall=0.46692786356303295, F1-Score=0.07614823723972036, AUC=0.5340499330183724
Resampling: Random Under-Sampling, Model: Gradient Boost: Precision=0.05323629948175102, Recall=0.5563493892601982, F1-Score=0.09717414056839224, AUC=0.6259812203261391
Resampling: Random Over-Sampling, Model: Logistic Regression: Precision=0.052837659480862297, Recall=0.5535837750633786, F1-Score=0.09646780055824414, AUC=0.6268906665857399
Resampling: Random Over-Sampling, Model: Naive Bayes: Precision=0.052687166178822534, Recall=0.4023968656372436, F1-Score=0.09317466246864826, AUC=0.5850050065033469
Resampling: Random Over-Sampling, Model: Random Forest: Precision=0.05234203080792204, Recall=0.5372205577321963, F1-Score=0.09539009268921492, AUC=0.6258724862922423
Resampling: Random Over-Sampling, Model: Balanced Random Forest: Precision=0.05108506440756644, Recall=0.5072597372666513, F1-Score=0.09282219973009447, AUC=0.6095191443750076
Resampling: Random Over-Sampling, Model: RUSBoost: Precision=0.05106145511195852, Recall=0.5061073980179764, F1-Score=0.09276390824990495, AUC=0.5959821238411226
Resampling: Random Over-Sampling, Model: Gradient Boost: Precision=0.05551835732619127, Recall=0.4586310209725743, F1-Score=0.09904686061269691, AUC=0.618156991137259
Resampling: Tomek Links, Model: Logistic Regression: Precision=0.05302880712608877, Recall=0.5570407928094031, F1-Score=0.09683881565767859, AUC=0.6274378629711912
Resampling: Tomek Links, Model: Naive Bayes: Precision=0.05356312797225301, Recall=0.345240838902973, F1-Score=0.09273819104810252, AUC=0.5858670748868774
Resampling: Tomek Links, Model: Random Forest: Precision=0.05280098616627859, Recall=0.533072136436967, F1-Score=0.09608474400249246, AUC=0.6252780446405519
Resampling: Tomek Links, Model: Balanced Random Forest: Precision=0.05113701160960701, Recall=0.5167089191057848, F1-Score=0.09306379975924618, AUC=0.6090707041382722
Resampling: Tomek Links, Model: RUSBoost: Precision=0.05203547650050365, Recall=0.4881309057386495, F1-Score=0.09404555747968563, AUC=0.6081971931336976
Resampling: Tomek Links, Model: Gradient Boost: Precision=0.4, Recall=0.0004609356994699239, F1-Score=0.0009208103130755064, AUC=0.6314095635831349
Resampling: SMOTE, Model: Logistic Regression: Precision=0.04974005611487044, Recall=0.5556579857109933, F1-Score=0.09130673533922856, AUC=0.6062736111804522
Resampling: SMOTE, Model: Naive Bayes: Precision=0.04049020585985567, Recall=0.5611892141046324, F1-Score=0.07553080943592289, AUC=0.5475683931989835
Resampling: SMOTE, Model: Random Forest: Precision=0.05739421868454127, Recall=0.06314819082737957, F1-Score=0.06013387468451662, AUC=0.5647755605931932
Resampling: SMOTE, Model: Balanced Random Forest: Precision=0.045829608746475965, Recall=0.3072136436967043, F1-Score=0.07976065818997756, AUC=0.5531441744774394
Resampling: SMOTE, Model: RUSBoost: Precision=0.035398230088495575, Recall=0.0009218713989398478, F1-Score=0.001796945193171608, AUC=0.5976054309239247
Resampling: SMOTE, Model: Gradient Boost: Precision=0.0, Recall=0.0, F1-Score=0.0, AUC=0.5688727944935641


```