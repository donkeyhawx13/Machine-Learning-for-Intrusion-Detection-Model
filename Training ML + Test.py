import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle
import os

#immporting of dataset using a for loop to combine all csv files within a directory
print("Loading in Data...")

path = "C:\\Users\\liamb\\OneDrive - Staffordshire University\\Year 3\\FYP\\Dataset\\Processed Traffic Data for ML Algorithms"
extension = ".csv"
files = [file for file in os.listdir(path) if file.endswith(extension)]
dfs = []
for file in files:
    df = pd.read_csv(os.path.join(path, file))
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(df)

#removal of all useless columns within the dataset
labels = df
df.pop("Timestamp")
df.pop("Init Fwd Win Byts")
df.pop("Dst Port")
df.pop("Flow IAT Min")
df.pop("Fwd Pkts/s")
df.pop("Fwd Pkt Len Std")
df.pop("Fwd Seg Size Min")
df.pop("Flow Duration")
df.pop("Fwd IAT Min")
df.pop("ECE Flag Cnt")
df.pop("Fwd IAT Mean")
df.pop("Init Bwd Win Byts")
df.pop("Bwd Pkts/s")
df.pop("Idle Max")
df.pop("Fwd IAT Std")
df.pop("FIN Flag Cnt")
df.pop("Fwd Header Len")
df.pop("SYN Flag Cnt")
df.pop("Fwd Pkt Len Max")
df.pop("Flow Pkts/s")
df.pop("Fwd Byts/b Avg")
df.pop("Fwd Pkts/b Avg")
df.pop("Fwd Blk Rate Avg")
df.pop("Bwd Byts/b Avg")
df.pop("Bwd Pkts/b Avg")
df.pop("Bwd Blk Rate Avg")
df.pop("Fwd PSH Flags")
df.pop("Bwd PSH Flags")
df.pop("Fwd URG Flags")
df.pop("Bwd URG Flags")

print("Loading in Data Done!")
df0 = df.copy
df2 = df

print("Pre processing...")
#preprocessing the label collum to change the text from strings to intergers
label_encoder = preprocessing.LabelEncoder()
df2["Label"] = label_encoder.fit_transform(df2["Label"])
df2["Label"].factorize()
df2.replace([np.inf, -np.inf], np.nan, inplace=True)

#data reshaping to replace NaN and Inifintiy with actual values
imp = SimpleImputer(strategy="most_frequent")
df2["Flow Byts/s"] = imp.fit_transform(df2["Flow Byts/s"].values.reshape(-1, 1))[:,0]
df2["Flow Byts/s"].isna().sum()

#making a new dataframe by cloning the old one for easier modifications to columns
df3 = pd.DataFrame(df2)

#data normalising
norm = pd.DataFrame(preprocessing.normalize(df3))
df3 = norm
 
#scales down data to fit between 0-10 instead of 0-10000
scaled = pd.DataFrame(StandardScaler().fit_transform(df3))
df3 = scaled
print(df3)

#swapping the scaled and normalised "Label" column with the reshaped one for whole numbers 
df3.columns = df.columns
print("Pre processing Finished")
df3["Label"] = df["Label"]
print(df3)
df3.to_csv("DF", index=False)
#df0.to_csv("DFLabel", index=False)

print("Machine learning Setting up...")
#setting up the X and y varibles for machine learning
X = df3.drop("Label", axis=1)
y = df3["Label"]
X.head()

y.head(), y.value_counts()

#splitting data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

print("Machine learning Setting up Done!")

print("Machine learning!")
#using linearsvc comapred to svc because of better handeling at larger datasets reducing computing time
model = LinearSVC()#max_iter=10000, dual=False, C=1)
model.fit(X_train, y_train)

print("Saving Machine learning...")

#saving the SVM model
export_plk = "IDSSVM.plk"
with open(export_plk, 'wb') as file:  
    pickle.dump(model, file)

print("Loading Machine learning...")

#loading back in the SVM model
export_plk = "IDSSVM.plk"
with open(export_plk, 'rb') as file:  
    model = pickle.load(file)

print("Predicting...")

#model prediction
y_pred = model.predict(X_test)

print("Loading Metrics and charts...")

#metrics of the machine learning model outputting as a dict to be able to plot in graphs
print("Metrics score")
classrep = classification_report(y_test, y_pred, output_dict=True)

df4 = pd.DataFrame(classrep)

df4.pop("accuracy")
df4.pop("macro avg")
df4.pop("weighted avg")
df5 = df4.loc["support"]
print(df5)
df6 = labels["Label"].unique()
print(df6)
#outputing metrics as a pie chart
fig = plt.figure(figsize=(10, 10))
plt.pie(df5, labels = df6)
plt.show()
