import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import lightgbm
#==============================================================================
#GENERATING A SIMPLE DUMMY DATASET
ml_df = pd.read_csv('abalone_data.csv')

ring_bins = [0, 5, 10, 15, 20, 25, 30]
labels = ['0-5','6-10','11-15','16-20','21-25','26-30']
ml_df['Rings_category'] = pd.cut(ml_df['Rings'], bins=ring_bins, labels=labels)
ml_df['Class'] = ml_df['Sex']
ml_df.drop(columns=['Sex'],inplace=True)
#==============================================================================


#==============================================================================
#Define the numerical and categorical columns of the features in the dataset
category_list = ['Rings_category','Class']
for i in category_list:
    ml_df[i] = ml_df[i].astype('category')


numerical_list = ['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']
for i in numerical_list:
    ml_df[i] = ml_df[i].astype('float')
#==============================================================================


#==============================================================================
#Minmax scaling all numerical columns so that they are between 0 and 1
df_scaler = MinMaxScaler()

numerical_variables_list = []
for col in numerical_list:
    if col in ml_df.columns:
        numerical_variables_list.append(col)

columns_to_preprocess = numerical_variables_list
x = ml_df[numerical_variables_list].values
x_scaled = df_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=columns_to_preprocess, index = ml_df.index)
ml_df[columns_to_preprocess] = df_temp
#==============================================================================


#==============================================================================
#Generating X and y (Input features and label)
X = ml_df.drop(columns=['Class'])
y = ml_df[['Class']]
#==============================================================================





#==============================================================================
#The real part begins here - LightGBM

#Optional: Based on train_test_split, I adapted the code so that I can allocate and split validation data too
def train_val_test_split(X, y, train_size, val_size, test_size, random_state=0, shuffle=True):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state, shuffle=shuffle)
    relative_train_size = train_size / (val_size + train_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      train_size = relative_train_size, test_size = 1-relative_train_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test



#Split data into train, validation, test
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X,y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=0)

'''
Number of estimators is inversely correlated with learning rate. Therefore, we can ignore learning rate and purely tune the estimators
The model will validate on X_val by incrementally adding 1 to n_estimators and making a prediction each time
The tuned model will be returned after there are no improvements in performances after 200 iterations (early stopping round)
'''
classifier = lightgbm.LGBMClassifier(n_estimators=99999, objective='multi_logloss',metric='multi_logloss', num_class=3
                          ,early_stopping_round=200)
classifier.fit(X_train, y_train, eval_set=(X_val,y_val))


#Make prediction with the validated/tuned classifier
y_pred = classifier.predict(X_test)



#Feature Importance plot
feature_importance = classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12,6))
barlist = plt.barh(range(len(sorted_idx)),feature_importance[sorted_idx],align='center')
sorted_columns = np.array(X_test.columns)[sorted_idx]
plt.yticks(range(len(sorted_idx)),sorted_columns)
plt.title('Feature Importance Visualisation')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance')

ranked_features = pd.DataFrame(sorted_columns,columns=['Ranked_Feature'])

#Let's say the client wants to emphasise and highlight these parameters. They will coloured red in the plot.
highlighted_features = ['Height','Diameter']
for i in highlighted_features:
    extracted_index_temp = ranked_features[ranked_features['Ranked_Feature'] == i].index[0]
    barlist[extracted_index_temp].set_color('r')


#Let's say we highlighted these features. They will be coloured green in the plot.
key_features = ['Shell_weight','Viscera_weight']
for i in key_features:
    extracted_index_temp = ranked_features[ranked_features['Ranked_Feature'] == i].index[0]
    barlist[extracted_index_temp].set_color('g')




#Save the feature importances in an easily accessible tuple format
importance = classifier.feature_importances_
importances = list(zip(classifier.feature_importances_, X_test.columns))
importances.sort(reverse=True)
















