# model.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_and_evaluate(X_train, X_test, y_train, y_test, target_names, output_dir='outputs', random_state=42):
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Save evaluation metrics to a text file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    return clf, accuracy, report