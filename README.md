import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import os
import joblib

model_file="model.pkl"
pipeline_file="pipeline.pkl"
input_file="input.csv"
def build_pipeline(num_name,cat_name):
    num_pipeline=Pipeline([
    ("Imputer",SimpleImputer(strategy="median")),
    ("Scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
    ("One_Hot",OneHotEncoder(handle_unknown="ignore"))
    ])
    combine=ColumnTransformer([
    ("num",num_pipeline,num_name),
    ("cat",cat_pipeline,cat_name)
    ])  
    return combine


if not os.path.exists(model_file) and not os.path.exists(pipeline_file):
    housing=pd.read_csv("housing.csv")

    #create training dataset and test dataset
    housing["income_cat"]=pd.cut(housing["median_income"],
                                bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                                labels=[1,2,3,4,5])
    #slipt of training dataset and testing dataset from housing csv
    sp=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for training_index,testing_index in sp.split(housing,housing["income_cat"]):
        stat_training_dataset=housing.iloc[training_index]
        stat_testing_dataset=housing.iloc[testing_index]
    
    #Graph of training_dataset and test_dataset
    plt.subplot(1,3,1)
    plt.hist(housing["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="Income_cat")
    plt.legend()
    plt.subplot(1,3,2)
    plt.hist(stat_training_dataset["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="stat_training_dataset",color="yellow")
    plt.legend()
    plt.subplot(1,3,3)
    plt.hist(stat_testing_dataset["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="stat_testing_dataset",color="red")
    plt.legend()
    plt.show()

    housing_train=stat_training_dataset.drop("income_cat",axis=1).drop("median_house_value",axis=1).copy()
    housing_labels=stat_training_dataset["median_house_value"].copy()
    housing_testing=stat_testing_dataset.drop("income_cat",axis=1).copy()

    #About Pipeline
    num_name=housing_train.drop("ocean_proximity",axis=1).columns.tolist()
    cat_name=["ocean_proximity"] 
    pipeline=build_pipeline(num_name,cat_name)  
    proper_training_dataset=pipeline.fit_transform(housing_train)

    #Model RandomForestRegressor
    model=RandomForestRegressor(random_state=42)
    model.fit(proper_training_dataset,housing_labels)

    #model store model_file  and  pipeline store in pipeline_file
    joblib.dump(model,model_file)
    joblib.dump(pipeline,pipeline_file)
    housing_testing.to_csv(input_file, index=False)

    print("Model Trained And Saved")

else:
    #Reload The Model And Pipeline
    model=joblib.load(model_file)
    pipeline=joblib.load(pipeline_file)
    input_data = pd.read_csv(input_file).drop("median_house_value",axis=1)


    transformed_input=pipeline.transform(input_data)
    prediction=model.predict(transformed_input)
    prediction_df = pd.DataFrame(
        prediction,
        columns=["Predicted House Value"]
    )
    prediction_df.to_csv("output.csv",index=False)

    print("Result saved to output.csv file")
