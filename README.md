import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer

housing=pd.read_csv("housing.csv")

#create training dataset and test dataset
housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                             labels=[1,2,3,4,5])
#graph of income_cat
plt.subplot(1,3,1)
plt.hist(housing["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="Income_cat")
plt.legend()

#slipt of training dataset and testing dataset from housing csv
sp=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for training_index,testing_index in sp.split(housing,housing["income_cat"]):
  stat_training_dataset=housing.iloc[training_index]
  stat_testing_dataset=housing.iloc[testing_index]

#Graph of training_dataset and test_dataset
plt.subplot(1,3,2)
plt.hist(stat_training_dataset["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="Income_cat",color="yellow")
plt.legend()
plt.subplot(1,3,3)
plt.hist(stat_testing_dataset["income_cat"],bins=[0,1,3,4,5,6,7],edgecolor="Black",label="Income_cat",color="red")
plt.legend()
plt.show()

# droping the income_cat
stat_training_dataset=stat_training_dataset.drop("income_cat",axis=1).copy()
stat_testing_dataset=stat_testing_dataset.drop("income_cat",axis=1).copy()
housing_train=stat_training_dataset.drop("median_house_value",axis=1)
housing_label=stat_training_dataset["median_house_value"]
housing_test=stat_testing_dataset.drop("median_house_value",axis=1)
housing_test_label=stat_testing_dataset["median_house_value"]

# convert the housing_train into proper trainingdataset
num_name=housing_train.drop("ocean_proximity",axis=1).columns.tolist()
cat_name=["ocean_proximity"]

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
proper_trainingdataset=combine.fit_transform(housing_train)
proper_testingdataset=combine.transform(housing_test)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
#model training and find rmse
lin_reg=LinearRegression()
lin_reg.fit(proper_trainingdataset,housing_label)
lin_preds=lin_reg.predict(proper_testingdataset)
lin_rmse=root_mean_squared_error(housing_test_label,lin_preds)
print(lin_rmse)

dec_tree=DecisionTreeRegressor()
dec_tree.fit(proper_trainingdataset,housing_label)
dec_preds=dec_tree.predict(proper_testingdataset)
dec_rmse=root_mean_squared_error(housing_test_label,dec_preds)
print(dec_rmse)

random_tree=RandomForestRegressor()
random_tree.fit(proper_trainingdataset,housing_label)
random_tree_preds=random_tree.predict(proper_testingdataset)
random_tree_rmse=root_mean_squared_error(housing_test_label,random_tree_preds)
print(random_tree_rmse)
