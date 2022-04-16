import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

matplotlib.use("Agg")

import keras
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def file_selector():

    file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    
    if file is not None:
        file_name = file.name
        data = pd.read_csv(file)
        st.write("The selected database will be displayed below. The panel gives you the chance to make some adjustments to the Dataset. You can select target column and feature columns. Also you can **encode** some columns as well. But first you should select an algorithm to perform on.")
        st.write("And remember, **YOU** are entirely responsible for the adjustments you make to the dataset.")
        st.write("Remember your dataset looks like this:")
        st.text(f"{file_name}")
        st.table(data.head(5))
        return data
    
    else:
        st.write("You must upload a CSV file to continue.")
        st.write("Remember your  dataset must have columns name/id and should look like this:")
        penguins = sns.load_dataset('penguins')
        st.write("Penguins")
        st.table(penguins.head(5))
        st.write("Or this maybe:")
        tips = sns.load_dataset('tips')
        st.write("Tips")
        st.table(tips.head(5))

def prepare_data(data, algorithm_type, algorithm_name, params, le, ohe):
    
    complete_data = data.copy()
    complete_data = complete_data.replace('?', np.nan)
    complete_data = complete_data.replace('', np.nan)
    complete_data = complete_data.dropna(axis=0, how="any")

    X = complete_data[params["chosen_features"]]
    y= complete_data[[params["chosen_target"]]]

    X_inverse = X.copy()
    y_inverse = y.copy()
    lenco = preprocessing.LabelEncoder()

    #encoding for X
    if le:
        for column in params["label_encode"]:
            X[column] = lenco.fit_transform(X[column])
            X_inverse[column] = lenco.inverse_transform(X[column])
    
    if ohe:
        for clm in params["one_hot_encode"]:
            dummy = pd.get_dummies(X[clm])
            X = pd.concat([X, dummy], axis = 1)
            X = X.drop([clm], axis=1)
    
    if algorithm_type == "Regression":

        if algorithm_name == 'Lineer Regression':
            return X, y, X_inverse, y_inverse

        if algorithm_name == 'DT Regression':
            return X, y, X_inverse, y_inverse

        if algorithm_name == 'Neural Networks Regression':
            return X, y, X_inverse, y_inverse

    elif algorithm_type == "Classification":

        if algorithm_name == "Logistic Regression":
            y = lenco.fit_transform(y)
            y_inverse = lenco.inverse_transform(y)
            return X, y, X_inverse, y_inverse

        if algorithm_name == "DT Classifier":
            y = lenco.fit_transform(y)
            y_inverse = lenco.inverse_transform(y)
            return X, y, X_inverse, y_inverse

        if algorithm_name == "Neural Networks Classification":
            y_inverse = y
            dummy = pd.get_dummies(y)
            y = pd.concat([y, dummy], axis = 1)
            y = y.drop([params["chosen_target"]], axis=1)
            return X, y, X_inverse, y_inverse

        if algorithm_name == "Naive Bayes (Gaussian)":
            y_inverse = np.unique(y)
            y = lenco.fit_transform(y)
            return X, y, X_inverse, y_inverse


    else: st.text("Wrong infos")

def algorithm_parameters(algorithm_name, data):

    params = dict()
    le = False
    ohe = False

    train_test_split = st.sidebar.slider("Train-Test Split", 1, 99, 66)
    params["train_test_split"] = train_test_split

    target_options = data.columns
    chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))
    params['chosen_target'] = chosen_target

    X_features = target_options.drop(chosen_target)
    chosen_features = st.sidebar.multiselect("Please choose feature columns", (X_features))
    all_features = st.sidebar.checkbox("Select all features")
    
    if all_features:
        params["chosen_features"] = X_features

    else:
        params["chosen_features"] = chosen_features

    if len(params["chosen_features"]) != 0:

        checkbox_encode = st.sidebar.checkbox("I want to encode some columns")

        if checkbox_encode:

            label_encode = st.sidebar.multiselect("Select the columns you wish to label encode", (params["chosen_features"]))
            one_hot_encode = st.sidebar.multiselect("select the columns you wish to one hot encode", (params["chosen_features"]))

            params["label_encode"] = label_encode
            params["one_hot_encode"] = one_hot_encode

            if len(label_encode) != 0:
                le = True
            else: le = False
            if len(one_hot_encode) != 0:
                ohe = True
            else: ohe = False

    if algorithm_name == 'Lineer Regression':
        pass       

    elif algorithm_name == 'Logistic Regression':
        pass

    elif algorithm_name == 'DT Regression':
        
        max_depth = st.sidebar.slider("Max Depth", 0, 10, 5)
        params["max_depth"] = max_depth

    elif algorithm_name == 'DT Classifier':

        max_depth = st.sidebar.slider("Max Depth", 0, 10, 5)
        params["max_depth"] = max_depth

    elif algorithm_name == 'Neural Networks Classification':

        epochs = st.sidebar.slider("number of epochs", 0 ,1000 ,100, 50)
        batch_size = st.sidebar.slider("Batch Size", 0,200,50)
        optimizer = st.sidebar.selectbox("Optimizer", ("Adam", "SGD", "RMSprop"))
        learning_rate = float(st.sidebar.text_input("learning rate:", "0.01"))
        hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 1)

        for hidden_layer in range(hidden_layers):

            params[f"units{hidden_layer}"] = st.sidebar.slider(f"Units of Hidden Layer {hidden_layer +1}", 1, 10, 5)
            params[f"Activation{hidden_layer}"] = st.sidebar.selectbox(f"Activation of Hidden Layer {hidden_layer +1}", ("relu", "tanh", "sigmoid", "softmax", "elu", "selu"))

        params["hidden_layers"] = hidden_layers
        params["epochs"] = epochs
        params["learning_rate"] = learning_rate
        params["optimizer"] = optimizer
        params["batch_size"] = batch_size
    
    elif algorithm_name == 'Neural Networks Regression':

        epochs = st.sidebar.slider("number of epochs", 0 ,1000 ,100, 50)
        batch_size = st.sidebar.slider("Batch Size", 0,200,150)
        optimizer = st.sidebar.selectbox("Optimizer", ("Adam", "SGD", "RMSprop"))
        learning_rate = float(st.sidebar.text_input("learning rate:", "0.01"))
        hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 1)

        for hidden_layer in range(hidden_layers):

            params[f"units{hidden_layer}"] = st.sidebar.slider(f"Units of Hidden Layer {hidden_layer +1}", 1, 10, 5)
            params[f"Activation{hidden_layer}"] = st.sidebar.selectbox(f"Activation of Hidden Layer {hidden_layer +1}", ("relu", "tanh", "sigmoid", "softmax", "elu", "selu"))

        params["hidden_layers"] = hidden_layers
        params["epochs"] = epochs
        params["learning_rate"] = learning_rate
        params["optimizer"] = optimizer
        params["batch_size"] = batch_size

    elif algorithm_name == "Naive Bayes (Gaussian)":
        pass
    else:
        pass
    
    return params, le, ohe

def get_available_dataset(name):
    data = None
    if name == 'Iris':
        data = sns.load_dataset("iris")
    elif name == 'Diamonds':
        data = sns.load_dataset("diamonds")
    elif name == 'Geyser':
        data = sns.load_dataset("geyser")
    elif name == 'Penguins':
        data = sns.load_dataset("penguins")
    elif name == 'Tips':
        data = sns.load_dataset("tips")

    return data

def run_algoirthms(X, y, X_inverse, y_inverse, algorithm_type, algorithm_name, params):

    if algorithm_type == "Regression":

        #* That is done
        if algorithm_name == 'Lineer Regression':

            PredictorScaler=MinMaxScaler()
            TargetVarScaler=MinMaxScaler()
            X=PredictorScaler.fit_transform(X)
            y=TargetVarScaler.fit_transform(y)

            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)

            lm = LinearRegression()

            model=lm.fit(X_train,y_train)

            st.subheader("**Training...**")
            r_sq = model.score(X_train, y_train)
            st.write('Coefficient of determination (model score) : ', r_sq)

            col1, col2 = st.columns(2)

            col1.write(" Model intercept (B0) = ")
            col1.table((model.intercept_))
            col2.write(" Model coef (B1) = ")
            col2.table((model.coef_))
            st.write("Train Error (RMSE) = ", np.sqrt(mean_squared_error(y_train,model.predict(X_train))))


            st.subheader("**Testing/Predicting...**")
            y_pred = model.predict(X_test)
            y_pred=TargetVarScaler.inverse_transform(y_pred)
            y_test_orig=TargetVarScaler.inverse_transform(y_test)

            st.write("Test Error (RMSE) = ", np.sqrt(mean_squared_error(y_test_orig,y_pred)))

            y_s = (pd.DataFrame({
                'y_pred    ' : y_pred.flatten(),
                'y_test    ' : y_test_orig.flatten()
            }))
            st.table(y_s.head(5))

        if algorithm_name == 'DT Regression':
            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)
            st.subheader("**Train & Test**")
            d_tree1 = DecisionTreeRegressor(max_depth = params["max_depth"], random_state=42)
            d_tree1.fit(X_train, y_train)
            st.write("Tain over")
            predictions = d_tree1.predict(X_test)
            st.write("predicted")

            y_test = np.array(y_test)
            predictions.flatten()
            y_test.flatten()
            errors = abs(predictions - y_test)
            col1, col2 = st.columns(2)
            col1.write('Mean Absolute Error:')
            col1.write(round(np.mean(errors), 2),)
            mape = 100 * (errors / y_test)
            accuracy = 100 - np.mean(mape)
            acc = round(accuracy, 3)

            col2.write('Accuracy: %')
            col2.write(acc)

            y_s = (pd.DataFrame({
                'y_pred' : predictions.flatten(),
                'y_test' : y_test.flatten()
            }))
            st.table(y_s.head(5))
            
            dot_data = tree.export_graphviz(d_tree1,
                                            feature_names= X.columns,
                                            rounded=True,
                                            filled=True)

            st.graphviz_chart(dot_data)


            fig, ax = plt.subplots()
            ranking = d_tree1.feature_importances_
            features = np.argsort(ranking)[::-1][:10]
            columns = X.columns

            plt.title("Feature importances based on Decision Tree Regressor", y = 1.03, size = 18)
            plt.bar(range(len(features)), ranking[features], color="lime", align="center")
            plt.xticks(range(len(features)), columns[features], rotation=80)

            st.pyplot(fig)

        #* That is done
        if algorithm_name == 'Neural Networks Regression':

            st.subheader("**Train & Test**")


            PredictorScaler=MinMaxScaler()
            TargetVarScaler=MinMaxScaler()
            X=PredictorScaler.fit_transform(X)
            y=TargetVarScaler.fit_transform(y)


            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)

            # create ANN model
            model = Sequential()
            for layer in range(params["hidden_layers"]):
                if layer == 1:
                    model.add(Dense(units=params[f"units{layer}"], input_dim=X_train.shape[1], kernel_initializer='normal', activation=params[f"Activation{layer}"]))
                else:
                    model.add(Dense(units=params[f"units{layer}"], kernel_initializer='normal', activation=params[f"Activation{layer}"]))

            if params["optimizer"] == "Adam":
                opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
            elif params["optimizer"] == "SGD":
                opt = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
            else:
                opt = tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])

            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))

            model.compile(loss='mean_squared_error', optimizer=opt)

            # Fitting the ANN to the Training set
            train = model.fit(X_train, y_train,batch_size = params["batch_size"], epochs = params["epochs"], verbose=1)
            # Generating Predictions on testing data
            Predictions=model.predict(X_test)

            # Scaling the predicted Price data back to original price scale
            Predictions=TargetVarScaler.inverse_transform(Predictions)

            # Scaling the y_test Price data back to original price scale
            y_test_orig=TargetVarScaler.inverse_transform(y_test)

            # Scaling the test data back to original scale
            # Test_Data=PredictorScaler.inverse_transform(X_test)

            y_s = (pd.DataFrame({
                'y_pred' : Predictions.flatten(),
                'y_test' : y_test_orig.flatten()
            }))
            st.table(y_s.head(5))

            APE=100*(abs(Predictions-y_test_orig)/y_test_orig)

            st.write('The Accuracy of ANN model is: %', 100-np.mean(APE))

            fig,ax = plt.subplots()
            plt.plot(train.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            st.pyplot(fig)
            pass

    elif algorithm_type == "Classification":

        #* That is done
        if algorithm_name == "Logistic Regression":

            st.subheader("**Train & Test**")

            scaler=MinMaxScaler()
            X=scaler.fit_transform(X)
            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)


            logisticRegr = LogisticRegression()
            sc_X=preprocessing.StandardScaler()
            X_train=sc_X.fit_transform(X_train)
            X_test=sc_X.transform(X_test)
            logisticRegr.fit(X_train,y_train)
            y_pred=logisticRegr.predict(X_test)
            cm=confusion_matrix(y_test,y_pred)
            y_s = (pd.DataFrame({
                'y_pred' : y_pred,
                'y_test' : y_test
            }))
            st.table(y_s.head(5))
            class_names = np.unique(y_inverse)
            st.write("Accuracy: % ",100*metrics.accuracy_score(y_test, y_pred))


            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            # create heatmap
            cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            sns.heatmap((cm), annot=True ,fmt='g')
            ax.xaxis.set_label_position("top")
            plt.tight_layout()
            plt.title('Confusion matrix', y=1.1)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)
            # st.write("Precision:",metrics.precision_score(y_test, y_pred))
            # st.write("Recall:",metrics.recall_score(y_test, y_pred))

        if algorithm_name == "DT Classifier":

            st.subheader("**Train & Test**")
            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)


            d_tree1 = DecisionTreeClassifier(max_depth = params["max_depth"], random_state=42)
            d_tree1.fit(X_train, y_train) 

            predictions = d_tree1.predict(X_test)
            score = round(accuracy_score(y_test, predictions), 3)
            cm1 = confusion_matrix(y_test, predictions)
            st.write("Accuracy score: %" , 100*score)
            y_s = (pd.DataFrame({
                'y_pred' : predictions,
                'y_test' : y_test
            }))
            st.table(y_s.head(5))
            class_names = np.unique(y_inverse)

            fig, ax = plt.subplots()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            # create heatmap
            cm = pd.DataFrame(cm1, index=class_names, columns=class_names)
            sns.heatmap((cm), annot=True ,fmt='g')
            ax.xaxis.set_label_position("top")
            plt.tight_layout()
            plt.title('Accuracy Score: {0}'.format(score), size = 15)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)


            # st.write(pd.DataFrame(classification_report(y_test, predictions, target_names=['0', '1'])))
                
            dot_data = tree.export_graphviz(d_tree1,
                                feature_names= X.columns,
                                rounded=True,
                                filled=True)
            
            st.graphviz_chart(dot_data)
            fig, ax = plt.subplots()
            ranking = d_tree1.feature_importances_
            features = np.argsort(ranking)[::-1][:10]
            columns = X.columns

            plt.title("Feature importances based on Decision Tree Classifier", y = 1.03, size = 18)
            plt.bar(range(len(features)), ranking[features], color="lime", align="center")
            plt.xticks(range(len(features)), columns[features], rotation=80)

            st.pyplot(fig)

        #* That is done
        if algorithm_name == "Neural Networks Classification":

            st.subheader("**Train & Test**")
            st.write("Please wait, it may take some time.")
            scaler=MinMaxScaler()
            X=scaler.fit_transform(X)

            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)
            # create ANN model
            model = Sequential()

            for layer in range(params["hidden_layers"]):
                if layer == 1:
                    model.add(Dense(units=params[f"units{layer}"], input_dim=X_train.shape[1], kernel_initializer='normal', activation=params[f"Activation{layer}"]))
                else:
                    model.add(Dense(units=params[f"units{layer}"], kernel_initializer='normal', activation=params[f"Activation{layer}"]))

            if params["optimizer"] == "Adam":
                opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
            elif params["optimizer"] == "SGD":
                opt = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
            else:
                opt = tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])

            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(len(y.columns), kernel_initializer='normal', activation="softmax"))

            model.compile(loss='mean_squared_error', optimizer=opt)

            train = model.fit(X_train, y_train,batch_size = params["batch_size"], epochs = params["epochs"], verbose=1)

            y_test_org = y_test.gt(0).dot(y_test.columns)
            y_test_org =list(y_test_org)
            Predictions=model.predict(X_test)
            classes = np.argmax(Predictions, axis=1)

            y_pred = []

            for label in classes:
                y_pred.append(y.columns[label])
            
            col1,col2 = st.columns(2)
            col1.write("Predicted Labels")
            col1.write(y_pred[0:10])
            col2.write("Original Labels")
            col2.write(y_test_org[:10])

            APE=100*(abs(np.sum(Predictions-y_test))/np.sum(y_test))

            st.write('Here the outputs are probabilistic. So the accuracy of ANN model is: %', 100-np.mean(APE))

            fig,ax = plt.subplots()
            plt.plot(train.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            st.pyplot(fig)

            class_names = np.unique(y_inverse)


            cm = confusion_matrix(y_test_org, y_pred)
            fig, ax = plt.subplots()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            # create heatmap
            cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            sns.heatmap((cm), annot=True ,fmt='g')
            ax.xaxis.set_label_position("top")
            plt.tight_layout()
            plt.title('Confusion matrix', y=1.1)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)

        #* That is done
        if algorithm_name == "Naive Bayes (Gaussian)":

            st.subheader("**Train & Test**")
            scaler=MinMaxScaler()
            X=scaler.fit_transform(X)

            train_size = params["train_test_split"]/100
            X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=(train_size),random_state=0)
            from sklearn.naive_bayes import GaussianNB
            siniflandirici=GaussianNB()
            siniflandirici.fit(X_train, y_train)
            y_pred=siniflandirici.predict(X_test)

            y_pred_labels = []
            y_test_labels = []

            APE=100*(abs(np.sum(y_pred-y_test))/np.sum(y_test))
            st.write('Here the outputs are probabilistic. So the accuracy of ANN model is: %', 100-np.mean(APE))

            for prediction in y_pred:
                y_pred_labels.append(y_inverse[prediction])

            y_pred_labels =np.array(y_pred_labels)

            for test in y_test:
                y_test_labels.append(y_inverse[test])

            y_test_labels = np.array(y_test_labels)

            col1,col2 = st.columns(2)
            col1.write("Predicted Labels")
            col1.write(list(y_pred_labels[0:10]))
            col2.write("Original Labels")
            col2.write(list(y_test_labels[:10]))

            cm = confusion_matrix(y_test, y_pred)

            class_names = (y_inverse)

            cm = pd.DataFrame(cm, index=class_names, columns=class_names)

            fig, ax = plt.subplots()
            sns.heatmap((cm), annot=True ,fmt='g')
            st.pyplot(fig)

def main():

    menu = ["Welcome!", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Welcome!":
        st.title("**Welcome, user!**")
        st.write("You can use this platform to train and test some machine learning algorithms on a dataset. I am here to help you to understand this GUI.")
        st.subheader("Our supported ML algorithms so far are:\n"
                "\n**_Lineer Regression, Decission Tree Regression, Neural Nerworks Regression_** - For Your Regression Problems\n"
                "\n**_Logistic Regression, Decision Tree Classification, Neural Networks Classification, Naive Bayes_** - For your Classification Problems")
        st.write("You can use the panel on the left to set your algorithm, dataset etc. But remember, we are still under a heavy development. If you want to see what you can do with this platform, go to **'Predict'** section in Menu.")

    elif choice == "Predict":

        radio_db = st.sidebar.radio("Pick one", ["Available Datasets", "I want to use my own dataset"])
        
        if radio_db == "Available Datasets":
            st.title("**We Have Avaible Datasets!**")
            st.write("You can use the data sets we imported from the **Seaborn** library. And don't worry, the selected database will be displayed below. If you want to use your own dataset, it's fine as well.")
            st.write("The panel gives you the chance to make some adjustments to the Dataset. But first you should select an algorithm to perform on.")
            st.write("And remember, **YOU** are entirely responsible for the adjustments you make to the dataset.")
            dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Diamonds", "Geyser", "Penguins", "Tips"))
            st.subheader(f"You currently selected {dataset_name} dataset")
            data = get_available_dataset(dataset_name)
            st.table(data.head(10))
            # st.table(np.array(data.info()))

            select_run(data)

        elif radio_db == "I want to use my own dataset":
            st.title("**Okay So You Want To Use Your Own Dataset!**")
            data = file_selector() 

            select_run(data)

def select_run(data):
    if data is not None:
        st.write("Data shape: " ,data.shape)
        st.sidebar.write("*I will remove all the rows with NaN/missing/? values")              
        algorithm_type = st.sidebar.selectbox("Select Algorithm Type", ("Classification", "Regression"))

        if algorithm_type == "Classification":
            algorithm_name = st.sidebar.selectbox("Select A Classification Algorithm to Run on", (" ", "Logistic Regression", "DT Classifier", "Neural Networks Classification", "Naive Bayes (Gaussian)"))
        else:
            algorithm_name = st.sidebar.selectbox("Select A Regression Algorithm to Run on", (" ", "Lineer Regression", "DT Regression", "Neural Networks Regression"))
        if algorithm_name != " ":
            params, le, ohe = algorithm_parameters(algorithm_name, data)
                
            button_pedict = st.sidebar.button('Predict / Get Results')
            if button_pedict:
                X, y, X_inverse, y_inverse = prepare_data(data, algorithm_type, algorithm_name, params,le, ohe)
                st.write("Now your features looks like this. Is this what you want?")
                st.table(X.head(5))
                run_algoirthms(X, y, X_inverse, y_inverse, algorithm_type, algorithm_name, params)

if __name__ == "__main__":
    main()