# main.py
from data_loader import load_and_split_data
from model import train_and_evaluate
from visualize import plot_decision_tree, print_feature_importance

def main():
    # Define output directory
    output_dir = 'outputs'
    
    # Load and split data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_split_data()
    
    # Train and evaluate model
    clf, accuracy, report = train_and_evaluate(X_train, X_test, y_train, y_test, target_names, output_dir)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize results
    print_feature_importance(clf, feature_names)
    plot_decision_tree(clf, feature_names, target_names, output_dir)

if __name__ == "__main__":
    main()