# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset (You would typically load your dataset here)
# For this example, I'll create a dummy dataset.
data = {
    'Clothing1': ['Red T-shirt', 'Blue Shirt', 'Grey Pants', 'Yellow Dress'],
    'Clothing2': ['Grey Pants', 'Black Jeans', 'Red T-shirt', 'Blue Shirt'],
    'Compatibility': [1, 0, 1, 0]  # 1 for compatible, 0 for not compatible
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing (This step would involve more complex preprocessing in practice)
# In this example, we'll convert clothing descriptions to numerical features.
df['Clothing1'] = df['Clothing1'].astype('category').cat.codes
df['Clothing2'] = df['Clothing2'].astype('category').cat.codes

# Step 3: Data Splitting
X = df[['Clothing1', 'Clothing2']]
y = df['Compatibility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 6: Make Predictions (You would typically use this to make compatibility predictions for new data)
new_data = {
    'Clothing1': ['Blue Jeans', 'Red T-shirt'],
    'Clothing2': ['Red T-shirt', 'Black Skirt']
}

new_df = pd.DataFrame(new_data)
new_df['Clothing1'] = new_df['Clothing1'].astype('category').cat.codes
new_df['Clothing2'] = new_df['Clothing2'].astype('category').cat.codes

predictions = clf.predict(new_df[['Clothing1', 'Clothing2']])
print(predictions)
