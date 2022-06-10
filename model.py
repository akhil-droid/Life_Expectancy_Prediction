import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# def primary_model(input_data):

#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#
#     Linear_model = LinearRegression()
#
#     Linear_model.fit(X_train, y_train)
#     return Linear_model.predict(input_data)
def inputs_for_pickle(country_input, yr, l):
    X = pd.read_csv("clean_life_expectancy_data1.csv")
    y = pd.read_csv("clean_life_expectancy_data2.csv")

    X_test1 = [0] * 193
    l1 = [yr]
    l1.extend(l)
    l1.extend(X_test1)
    col_index = X.columns.get_loc(country_input)

    l1[col_index] = 1
    X_test = np.array(l1)

    lr = pickle.load(open('Linear_model.pkl','rb'))
    y_pred = lr.predict([X_test])
    return y_pred


def generating_the_default_values(country_input, yr, features_selected, new_df):
    x = []
    y = []
    default_input_list = []
    default_input_list.append(yr)

    l = new_df.index[new_df['country'] == country_input].tolist()
    print(l)
    for col in features_selected:

        if col not in ('country', 'year', 'life_expectancy'):
            for i in l:
                x.append(new_df._get_value(i, 'year'))
                y.append(new_df._get_value(i, col))  # we will take the column names form the best selected column list
            x = np.array(x)
            y = np.array(y)
            print(x, y)

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import r2_score
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

            from sklearn.linear_model import LinearRegression
            inner_Linear_model = LinearRegression()

            inner_Linear_model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
            y_pred = inner_Linear_model.predict(X_test.reshape(-1, 1))
            X_test = np.array([yr])  # the year is given by the user

            pred = inner_Linear_model.predict(X_test.reshape(-1, 1))
            default_input_list.append(round(pred[0][0], 2))
            x = []
            y = []

    return default_input_list


def main(country, yr):
    features_selected = ['country', 'year', 'life_expectancy', 'adult_mortality', 'bmi', 'diphtheria',
                         'hiv_and_aids', 'thinness_10_19_years', 'income_composition_of_resources', 'schooling']
    new_df = pd.read_csv("clean_life_expectancy_data.csv")
    print(country, yr)
    return generating_the_default_values(country, yr, features_selected, new_df)












# country = 'Afghanistan'
# year = 2045
#
# X_test1 = [0] * 193
# X_test = generating_the_default_values(country, year, features_selected)
# X_test.extend(X_test1)
#
# col_index = X.columns.get_loc(country)
#
# X_test[col_index] = 1
# X_test = np.array(X_test)
# y_pred = Linear_model.predict([X_test])
#
# print(y_pred[0])

# pickle.dump(Linear_model, open('Linear_model.pkl', 'wb'))
# pickle.dump(inner_Linear_model, open('inner_Linear_model.pkl', 'wb'))
