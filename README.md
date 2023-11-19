# project_mini


## AIM:
The aim of Data Shield is to enhance IoT database maintenance and security through the integration of machine intelligence, ensuring data integrity, confidentiality, and availability in the Internet of Things ecosystem.

## THEOREY:


## PROCEDURE:
### STEP1: Data Collection and Preprocessing:
Gather necessary data, such as system activity logs, network traffic logs, and other security-related information.
Handle missing values, scale or normalize numerical features, and encode categorical variables as part of the preprocessing step of the data.

### STEP2: Feature Selection:
Determine and select features that are important for anomaly detection and security monitoring.

### STEP3: Training Data Preparation:
Make a training dataset that mostly consists of normal cases to illustrate typical system behavior.

### STEP4: Model Training:
a. Isolation Forest:
Utilizing the prepared training dataset, train the isolation forest model.

b. Local Outlier Factor (LOF):
Utilizing the training data, train the LOF model.

c. Autoencoders:
Utilizing the training dataset, train the autoencoder model.

### STEP5: Anomaly Detection:
### a. Isolation Forest:
Utilize the complete dataset with the learned isolation forest model.
Shorter travel lengths should be noted as possible anomalies.

### b. Local Outlier Factor (LOF):
Using the dataset, apply the learned LOF model.
Lower local density instances should be noted as potential anomalies.

### c. Autoencoders:
Reconstruct the input data using the trained autoencoder.
For every instance, determine the reconstruction error.
Determine whether cases have huge reconstruction mistakes and mark them as possible anomalies.

### STEP6: Threshold Setting:
Determine a threshold for reconstruction mistakes or anomaly scores by using domain expertise or validation data.
Outliers are those instances that go outside of the norm.

### STEP7: Integration with Security Infrastructure:
The outlier detection system should be incorporated into the larger security framework.
Initiate real-time monitoring to discover anomalies continuously.

### STEP8: Alerting and Response:
Implement a system that will send out warnings or alerts in the event that possible outliers are detected.
Establish a response strategy to look into and address security threats that the outlier detection system has found.

### STEP9: Regular Model Evaluation and Updating:
Analyze the models' performance on a regular basis with new data.
To adjust to modifications in the system and data patterns, update the models as necessary.

### STEP10: Documentation and Reporting:
Keep records of the processes, model configurations, and outcomes.
Provide periodical reports that highlight significant discoveries and the effectiveness of outlier detection.

## PROGRAM:
## PREPROCESSING:
```
import pandas as pd
url = "/content/trafficData158324.csv"
df = pd.read_csv(url)
pip install matplotlib
df.head
print(df['vehicleCount'])
monthi = df['TIMESTAMP'].head(40)
print(monthi)
vehicle_countsi = df['vehicleCount'].head(40)
print(vehicle_countsi)
avgi = df['avgSpeed'].head(5)
import matplotlib.pyplot as plt
import numpy as np
# Sample data: Number of vehicles per month
months = monthi
vehicle_counts = vehicle_countsi
    # Create a figure and axis for the bar chart
fig, ax = plt.subplots()

# Create the bar chart
ax.bar(vehicle_counts,months ,color='b')

# Set axis labels and title
ax.set_xlabel('vehicle count')
ax.set_ylabel('Timestamp')
ax.set_title('Monthly Vehicle Counts')

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.hist(df['vehicleCount'], bins=10)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

import matplotlib.pyplot as plt
plt.plot(vehicle_countsi,monthi)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot')
plt.show()
import matplotlib.pyplot as plt
plt.scatter(vehicle_countsi,monthi)
plt.xlabel('vehicle')
plt.ylabel('time')
plt.title('Scatter Plot')
plt.show()
df = data[["TIMESTAMP","vehicleCount"]]
print(df.describe())
pip install scikit-learn
```
### Isolation Forest:
```
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
data = pd.read_csv('/content/trafficData158324.csv')
 # Assuming the dataset has a 'timestamp' and a 'flow' column

timestamps = monthi
flow_data =vehicle_countsi
X = flow_data.values.reshape(-1, 1)
model = IsolationForest(contamination=0.05, random_state=42)  # Adjust the contamination parameter as needed
model.fit(X)
# Predict anomalies
anomaly_predictions = model.predict(X)
# Plot the traffic flow data with anomalies highlighted
plt.figure(figsize=(12, 6))
plt.plot(timestamps, flow_data, label='Traffic Flow', color='b')
plt.scatter(timestamps[anomaly_predictions == -1], flow_data[anomaly_predictions == -1], color='r', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Flow')
plt.title('Anomaly Detection in Traffic Flow Data')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
```

### Local Outlier Factor (LOF):
```
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
# Initialize and fit the LOF model
model = LocalOutlierFactor(contamination=0.05)
anomaly_scores = model.fit_predict(X)

# Plot the traffic flow data with anomalies highlighted
plt.figure(figsize=(12, 6))
plt.plot(timestamps, flow_data, label='Traffic Flow', color='black')
plt.scatter(timestamps[anomaly_scores == -1], flow_data[anomaly_scores == -1], color='blue', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Flow')
plt.title('Anomaly Detection in Traffic Flow Data (Local Outlier Factor)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
```
###  Autoencoders:
```
import pandas as pd
df = pd.read_csv('minids.csv',nrows=25)
df.head()
df.isnull().sum()
import matplotlib.pyplot as plt
# Assuming 'df' is your DataFrame with a 'traffic_volume' column
plt.hist(df['_id'], bins=10)
plt.xlabel('TIMESTAMP')
plt.ylabel('vechicleCount')
plt.title('Histogram of Traffic Volume')
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
# Assuming your data is in a DataFrame named 'df'
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']) # Convert 'TIMESTAMP
plt.figure(figsize=(12, 6))
plt.plot(df['TIMESTAMP'], df['avgMeasuredTime'], label='Average Measure
plt.xlabel('Timestamp')
plt.ylabel('Average Measured Time')
plt.title('Average Measured Time Over Time')
plt.legend()
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
# Assuming your data is in a DataFrame named 'df'
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']) # Convert 'TIMESTAMP
plt.figure(figsize=(12, 6))
plt.plot(df['TIMESTAMP'], df['avgMeasuredTime'], label='Average Measure
plt.xlabel('Timestamp')
plt.ylabel('vechicleCount')
plt.title('vechicle count Over Time')
plt.legend()
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
# Assuming your data is in a DataFrame named 'df'
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']) # Convert 'TIMESTAMP
plt.figure(figsize=(12, 6))
plt.plot(df['TIMESTAMP'], df['vehicleCount'], label='Average Measured T
plt.xlabel('Timestamp')
plt.ylabel('Average Measured Time')
plt.title('Average Measured Time Over Time')
plt.legend()
plt.grid(True)
plt.show()
 import matplotlib.pyplot as plt
# Assuming your data is in a DataFrame named 'df'
plt.scatter(df['avgMeasuredTime'], df['avgSpeed'])
plt.xlabel('Average Measured Time')
plt.ylabel('Average Speed')
plt.title('Scatter Plot of Average Measured Time vs. Average Speed')
plt.show()
import matplotlib.pyplot as plt
# Assuming your data is in a DataFrame named 'df'
plt.bar(df['vehicleCount'].value_counts().index, df['vehicleCount'].val
#plt.xlabel('Status')
#plt.ylabel('Count')
#plt.title('Distribution of Status')
plt.show()
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Load the dataset
#df = pd.read_csv('your_traffic_data.csv') # Replace with your dataset
# Preprocess the data
data = df[['avgMeasuredTime', 'avgSpeed', 'vehicleCount']] # Select nu
scaler = StandardScaler()
data = scaler.fit_transform(data)
# Build an autoencoder model
input_dim = data.shape[1]
encoding_dim = 2 # Adjust as needed
input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation="relu")(input_lay
decoder = keras.layers.Dense(input_dim, activation="relu")(encoder)
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
# Compile the model
autoencoder.compile(optimizer="adam", loss="mse")
# Train the model
autoencoder.fit(data, data, epochs=50, batch_size=32, verbose=0)
# Use the trained autoencoder to reconstruct data
reconstructed_data = autoencoder.predict(data)
# Calculate the mean squared error (MSE) for each data point
mse = np.mean(np.power(data - reconstructed_data, 2), axis=1)
# Define a threshold for anomaly detection (adjust as needed)
threshold = 0.1
# Detect anomalies by comparing MSE to the threshold
anomalies = df[mse > threshold]
# Visualize anomalies
plt.scatter(anomalies.index, mse[mse > threshold], color='red', marker=
plt.xlabel('Data Point Index')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Anomaly Detection using Autoencoder')
plt.legend()
plt.show()
# Print the detected anomalies
print("Detected Anomalies:")
print(anomalies)
```
### FEATURE GENERATION:
```
"""df = pd.read_csv(url)
vehicle_countsi = df['vehicleCount'].head(40)"""


# Convert 'timestamp' column to datetime if it's not already
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Extract hour from the 'timestamp' column
df['hour_of_day'] = df['TIMESTAMP'].dt.hour

# Define a threshold for high and low vehicle count (you can adjust these values)
high_vehicle_threshold = 10
low_vehicle_threshold = 7

# Create binary features indicating high and low vehicle count
df['high_vehicle_count'] = np.where(df['vehicleCount'] > high_vehicle_threshold, 1,0)
df['low_vehicle_count'] = np.where(df['vehicleCount'] < low_vehicle_threshold, 0,1)




# Display the updated DataFrame
df.head(50)
```
## OUTPUT:
## PREPROCESSING:
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/14b027c8-9904-4733-8ebc-a08e6fddb933)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/958fa8cd-e35a-4caa-ab5d-e9946ea2c4e4)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/a9279f0e-b2a2-4397-b0a2-aa204d3d98de)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/7cbfb242-2ea4-4ba3-8096-3ee26e02f281)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/d9b2a106-a532-4d98-b8a4-507ccdc095ed)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/d5338b4a-2d89-4338-b760-c3a94ebe24f8)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/2fa81698-d438-4214-8061-4cee6587944e)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/5f2f0ef7-ebb9-4a3d-9669-95dd601f0de4)
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/818ae100-f219-494f-a59e-64807e40a836)



### Isolation Forest:
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/262b3d7f-a040-4393-a6e9-022bc230b4d1)

### Local Outlier Factor (LOF):
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/0a2063ca-ca3d-4923-9936-46fdc618cb36)

###  Autoencoders:


### FEATURE GENERATION
![image](https://github.com/Evangelin-Ruth/project_mini/assets/94219798/9984cc8a-b892-4780-9901-cee5da316def)
