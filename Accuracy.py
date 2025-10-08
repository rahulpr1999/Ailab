Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=iris.target_names,
    columns=iris.target_names
)
print(" Confusion Matrix:")
print(cm_df, "\n")

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
