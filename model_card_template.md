# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was created by Natalie Laffranchi (nlaffra).
The RandomForestClassifier from sklearn.model.RandomForestClassifier was used for classification tasks.
The default parameters were used. 

## Intended Use

This model is intended for educational purposes only. 
It predicts whether a person earns over 50k or not based on the census data provided by Udacity.

## Training Data

More details about the training data: https://archive.ics.uci.edu/ml/datasets/census+income

Extraction was done by Barry Becker from the 1994 Census database.

Prediction task is to determine whether a person makes over 50K a year.

Features:

    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

For both training and evaluation, categorical features of the data are encoded using OneHotEncoder and the target is transformed using LabelBinarizer

## Evaluation Data

The original dataset is first preprocessed and then split into training and evaluation data with evaluation data size of 20%

## Metrics
Precision: 0.7152 | Recall: 0.6194 | F1: 0.6639

## Ethical Considerations

This model is trained on US census data and may not be accurate for other countries.

## Caveats and Recommendations

I recommend that review is included upstream of any decision-making points to ensure that bias is minimized.
