from django.shortcuts import render,redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
def home(request):
    return render(request,'index.html')

def predict(request):
    return render(request,'predict.html')

def result(request):

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    res = normalize_user_input(val1,val2,val3,val4,val5)
    print(res[0])
    print(res[1])
    print(res[2])
    print(res[3])
    print(res[4])

    if request.GET['algo'] == 'ml' :

        # val1 = float(request.GET['n1'])
        # val2 = float(request.GET['n2'])
        # val3 = float(request.GET['n3'])
        # val4 = float(request.GET['n4'])
        # val5 = float(request.GET['n5'])
        # res = normalize_user_input(val1,val2,val3,val4,val5)
        # print(res[0])
        # print(res[1])
        # print(res[2])
        # print(res[3])
        # print(res[4])

        df = pd.read_csv('bnormalizedt20.csv')
        print(df.shape)
        df = df.drop(columns=['Unnamed: 0'],axis=1)
        print(df.shape)

        train_data = df.values
        Y = train_data[:,-1].reshape(train_data.shape[0],1)
        X = train_data[:,:-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print("Shape of X_train :", X_train.shape)
        print("Shape of Y_train :", Y_train.shape)
        print("Shape of X_test :", X_test.shape)
        print("Shape of y_test :",Y_test.shape)


        X_train = np.vstack((np.ones((X_train.shape[0], )), X_train.T)).T
        X_test = np.vstack((np.ones((X_test.shape[0], )), X_test.T)).T
        
        print("Shape of X_train :", X_train.shape)
        print("Shape of Y_train :", Y_train.shape)
        print("Shape of X_test :", X_test.shape)
        print("Shape of y_test :",Y_test.shape)
        
        # iteration = 100000
        # learning_rate = 0.001
        # theta = model(X_train, Y_train, learning_rate = learning_rate, iteration = iteration)
        input = np.array([[1,res[0],res[1],res[2],res[3],res[4]]])

        filename = 'linear_model.pkl'
        theta = pickle.load(open(filename,'rb'))
    
        y_pred = np.dot(input, theta)[0]

        total_runs = round((int(263-55) * y_pred[0]) + 55)
        
        print("total runs:",total_runs) 
        
        # return redirect('/predict/',{'data':total_runs})
        return render(request,'predict.html',{'data':total_runs})

    else:
        filename = 'forest.pkl'
        forest = pickle.load(open(filename,'rb'))
        # print(forest)
        print("im random forest")
        mydict = [{'current_score': res[0], 'balls_left': res[1], 'wickets_left': res[2], 'crr': res[3],'last_five' : res[4]}]
        input = pd.DataFrame(mydict)
        print(input)
        # print("values are:",input.iloc[0])
        prediction = random_forest_predictions(input,forest)
        print("output:",prediction[0])
        total_runs = round((int(263-55) * prediction[0]) + 55)
        print("total runs:",total_runs) 
        return render(request,'predict.html',{'data':total_runs})
            # return redirect('/predict/', data=total_runs)
def normalize_user_input(val1,val2,val3,val4,val5):
    current_score = (val1 - 8) / (263 - 8)
    balls_left = (val2- 0) / (98 - 0)
    wickets_left = (val3 - 0) / (10 - 0)
    crr = (val4 - 1.6) / (16.6 - 1.6)
    last_five = (val5 - 8) / (89 - 8)
    return current_score,balls_left,wickets_left,crr,last_five


# def model(X_train, Y_train, learning_rate, iteration):
    m = Y_train.size
    print("model size:",m)
    theta = np.zeros((X_train.shape[1], 1))
#     cost_list = []
    
    for i in range(iteration):
        y_pred = np.dot(X_train, theta)
#         cost = (1/(2*m))*np.sum(np.square(y_pred - Y_train))
        d_theta = (1/m)*np.dot(X_train.T, y_pred - Y_train)
        theta = theta - learning_rate*d_theta
#         cost_list.append(cost)
        # to print the cost for 10 times
#         if(i%(iteration/10) == 0):
#             print("Cost is :", cost)
            
    return theta

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mean(axis=1)
    
    return random_forest_predictions


def decision_tree_predictions(test_df, tree):
#     predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions

def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)




