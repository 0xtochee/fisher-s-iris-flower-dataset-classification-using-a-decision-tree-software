# visualize.py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def plot_decision_tree(clf, feature_names, target_names, output_dir='outputs'):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True)
    plt.title("Decision Tree for Iris Dataset")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_feature_importance(clf, feature_names):
    importances = clf.feature_importances_
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.4f}")