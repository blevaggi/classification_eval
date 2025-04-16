import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to display multiclass metrics - defined BEFORE it's called
def display_multiclass_metrics(y_true, y_pred, cm, label_map, num_classes):
    # Calculate multiclass metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Get per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, 
        index=[f'True: {label_map[i]}' for i in range(num_classes)],
        columns=[f'Pred: {label_map[i]}' for i in range(num_classes)])
    
    st.dataframe(cm_df)
    
    # Display overall metrics
    st.subheader("Overall Metrics (Macro Average)")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        # F1 Score
        st.metric("F1 Score (Macro)", f"{f1_macro:.4f}")
        st.markdown("""
        **F1 Score (Macro) Explanation:**  
        The F1 score calculated independently for each class and then averaged.
        This gives equal weight to each class, regardless of its frequency.
        
        Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
        """)
    
    with metrics_col2:
        # Precision and Recall
        st.metric("Precision (Macro)", f"{precision_macro:.4f}")
        st.metric("Recall (Macro)", f"{recall_macro:.4f}")
        st.markdown("""
        **Macro Average Explanation:**  
        The macro average calculates metrics for each class independently, then takes the average.
        This treats all classes equally regardless of how many samples each has.
        """)
    
    # Display per-class metrics
    st.subheader("Per-Class Metrics")
    
    # Create metrics table
    class_metrics = []
    
    for i in range(num_classes):
        class_name = label_map[i]
        if str(i) in report:
            class_data = report[str(i)]
            class_metrics.append({
                'Class': class_name,
                'Precision': f"{class_data['precision']:.4f}",
                'Recall': f"{class_data['recall']:.4f}",
                'F1 Score': f"{class_data['f1-score']:.4f}",
                'Support': int(class_data['support'])
            })
    
    class_metrics_df = pd.DataFrame(class_metrics)
    st.dataframe(class_metrics_df)
    
    # Explanation of per-class metrics
    st.markdown("""
    **Per-Class Metrics Explanation:**
    - **Precision**: For each class, what percentage of predictions for this class were correct. 
    - **Recall**: For each class, what percentage of actual instances of this class were correctly identified
    - **F1 Score**: The harmonic mean of precision and recall for each class
    - **Support**: The number of actual occurrences of the class in the dataset
    
    These metrics help you identify which classes your model struggles with the most.
    """)
    
    # Display confusion matrix explanation
    st.markdown("""
    **Reading the Confusion Matrix:**
    - Each row represents the actual class
    - Each column represents the predicted class
    - The diagonal elements show correct predictions
    - Off-diagonal elements show misclassifications
    
    For example, the value at row "True: low" and column "Pred: medium" shows how many "low" instances were incorrectly classified as "medium".
    """)

# Set page title and configuration
st.set_page_config(page_title="F1 Score Calculator", layout="wide")
st.title("Classification Metrics Calculator")
st.write("Upload your dataset with ground truth and predicted values to calculate F1 score and other metrics.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File successfully loaded!")
        
        # Display preview of the data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        columns = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            ground_truth_col = st.selectbox("Select ground truth column", columns)
        
        with col2:
            predicted_col = st.selectbox("Select predicted column", columns, index=min(1, len(columns)-1))
        
        # Button to calculate metrics
        if st.button("Calculate Metrics"):
            # Wrap everything in a try-except to catch any errors
            try:
                # Get the ground truth and predicted values
                # Handle missing values first
                st.info("Checking for missing values...")
                
                # Drop rows with missing values in either column
                missing_rows = df[df[ground_truth_col].isna() | df[predicted_col].isna()]
                if not missing_rows.empty:
                    st.warning(f"Dropped {len(missing_rows)} rows with missing values.")
                    df = df.dropna(subset=[ground_truth_col, predicted_col])
                    
                # Convert values to appropriate types
                try:
                    y_true = df[ground_truth_col].values
                    y_pred = df[predicted_col].values
                except Exception as e:
                    st.error(f"Error extracting values: {e}")
                    # return
                
                # Convert values to strings first to handle mixed types
                y_true_str = [str(val) for val in y_true]
                y_pred_str = [str(val) for val in y_pred]
                
                # Get unique values after string conversion
                unique_values = sorted(list(set(y_true_str) | set(y_pred_str)))
                num_classes = len(unique_values)
                
                # Create mapping using string values
                value_map = {val: i for i, val in enumerate(unique_values)}
                
                # Map all values to integers
                y_true_mapped = np.array([value_map[str(val)] for val in y_true])
                y_pred_mapped = np.array([value_map[str(val)] for val in y_pred])
                
                st.info(f"Values mapped to numbers: {value_map}")
                
                # Use the mapped values for all calculations
                y_true = y_true_mapped
                y_pred = y_pred_mapped
                
                # Create reverse mapping for display
                reverse_map = {i: val for val, i in value_map.items()}
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Calculate metrics
                if num_classes == 2:
                    # Binary classification case
                    binary_mode = st.radio("Choose metric calculation mode:", ["Macro Average", "Class 1 as Positive"])
                    
                    if binary_mode == "Class 1 as Positive":
                        # For binary case, calculate metrics treating class 1 as positive
                        tn, fp, fn, tp = cm.ravel()
                        
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        # Display confusion matrix visualization
                        st.subheader("Confusion Matrix")
                        
                        # Create DataFrame for better visualization
                        cm_df = pd.DataFrame(cm, 
                            index=[f'True: {reverse_map[i]}' for i in range(num_classes)],
                            columns=[f'Pred: {reverse_map[i]}' for i in range(num_classes)])
                        
                        st.dataframe(cm_df)
                        
                        # Display binary metrics
                        st.subheader("Binary Classification Metrics")
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            # F1 Score
                            st.metric("F1 Score", f"{f1:.4f}")
                            st.markdown("""
                            **F1 Score Explanation:**  
                            The F1 score is the harmonic mean of precision and recall, giving equal importance to both metrics.
                            It ranges from 0 (worst) to 1 (best) and is particularly useful when the dataset is imbalanced.
                            
                            Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
                            """)
                            
                            # True Positive Rate (Recall/Sensitivity)
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            st.metric("True Positive Rate (Recall/Sensitivity)", f"{tpr:.4f}")
                            st.markdown("""
                            **True Positive Rate Explanation:**  
                            The proportion of actual positives that were correctly identified.
                            This measures how good the model is at finding all positive cases.
                            
                            Formula: TPR = TP / (TP + FN)
                            """)
                            
                            # True Negative Rate (Specificity)
                            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                            st.metric("True Negative Rate (Specificity)", f"{tnr:.4f}")
                            st.markdown("""
                            **True Negative Rate Explanation:**  
                            The proportion of actual negatives that were correctly identified.
                            This measures how good the model is at avoiding false alarms.
                            
                            Formula: TNR = TN / (TN + FP)
                            """)
                        
                        with metrics_col2:
                            # True Positives
                            st.metric("True Positives (TP)", tp)
                            st.markdown("""
                            **True Positives Explanation:**  
                            Cases that were actually positive and predicted as positive.
                            Think of these as "correctly identified positives."
                            """)
                            
                            # False Positives
                            st.metric("False Positives (FP)", fp)
                            st.markdown("""
                            **False Positives Explanation:**  
                            Cases that were actually negative but predicted as positive.
                            These are also called "Type I errors" or "false alarms."
                            """)
                            
                            # False Negatives
                            st.metric("False Negatives (FN)", fn)
                            st.markdown("""
                            **False Negatives Explanation:**  
                            Cases that were actually positive but predicted as negative.
                            These are also called "Type II errors" or "misses."
                            """)
                            
                            # True Negatives
                            st.metric("True Negatives (TN)", tn)
                            st.markdown("""
                            **True Negatives Explanation:**  
                            Cases that were actually negative and predicted as negative.
                            Think of these as "correctly identified negatives."
                            """)
                        
                        # Additional metrics
                        st.subheader("Additional Metrics")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            # Precision
                            st.metric("Precision (Positive Predictive Value)", f"{precision:.4f}")
                            st.markdown("""
                            **Precision Explanation:**  
                            The proportion of predicted positives that were actually positive.
                            This measures how trustworthy the positive predictions are.
                            
                            Formula: Precision = TP / (TP + FP)
                            """)
                        
                        with col4:
                            # Accuracy
                            accuracy = (tp + tn) / (tp + tn + fp + fn)
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            st.markdown("""
                            **Accuracy Explanation:**  
                            The proportion of all predictions that were correct.
                            This measures overall correctness across both classes.
                            
                            Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
                            """)
                    else:
                        # Display macro average metrics
                        display_multiclass_metrics(y_true, y_pred, cm, reverse_map, num_classes)
                else:
                    # Multiclass classification case
                    display_multiclass_metrics(y_true, y_pred, cm, reverse_map, num_classes)
                
                # F1 Score explained with an analogy
                st.subheader("Understanding F1 Score with an Analogy")
                st.markdown("""
                **Fishing Analogy:**
                
                Imagine you're fishing in a lake with both fish and logs. Your task is to catch only fish:
                - **Precision** is the percentage of actual fish among everything you caught. If you caught 8 fish and 2 logs, your precision is 80%.
                - **Recall** is the percentage of all the fish in the lake that you managed to catch. If there were 10 fish in the lake and you caught 8, your recall is 80%.
                - **F1 Score** balances these two metrics. It will be high only if both precision and recall are reasonably high.
                
                A high F1 score means you're not only catching most of the fish (high recall) but also avoiding catching logs (high precision).
                
                **Extending to Multiple Classes:**
                
                In a multi-class setting, it's like fishing for different types of fish (trout, bass, etc.) while avoiding logs. The F1 score is calculated for each type of fish separately, then averaged, giving you an overall measure of how well you're identifying each type of fish.
                """)
                
                # Tips for improving the metrics
                st.subheader("Tips for Improving Classification Performance")
                st.markdown("""
                - **Low Precision (many false positives)**: Your model is too "eager" to predict certain classes. Try adjusting the classification thresholds.
                - **Low Recall (many false negatives)**: Your model is missing too many cases of certain classes. Try rebalancing your training data.
                - **Low F1 Score**: Consider feature engineering, trying different algorithms, or collecting more training data.
                - **Imbalanced Classes**: Use techniques like oversampling, undersampling, or SMOTE to address class imbalance.
                - **Confusion Between Specific Classes**: Look at the confusion matrix to identify which classes are getting mixed up, then focus on features that better distinguish between them.
                """)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
                st.info("Debug information: Please check your data types and ensure you have selected the correct columns.")
        
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    # Display sample data option
    if st.checkbox("Try with sample data"):
        data_type = st.radio("Choose sample data type:", ["Binary Classification", "Multi-class Classification"])
        
        if data_type == "Binary Classification":
            st.info("Loading sample binary classification data...")
            
            # Create sample binary data
            sample_data = {
                'actual': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'predicted': [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
            }
            
            sample_df = pd.DataFrame(sample_data)
        else:
            st.info("Loading sample multi-class classification data...")
            
            # Create sample multi-class data
            sample_data = {
                'actual': ['low', 'medium', 'high', 'low', 'medium', 'high', 'low', 'medium', 'high', 'low', 
                           'medium', 'high', 'low', 'medium', 'high', 'low', 'medium', 'high', 'low', 'medium'],
                'predicted': ['low', 'medium', 'medium', 'low', 'medium', 'high', 'medium', 'medium', 'high', 'low',
                              'low', 'high', 'low', 'high', 'high', 'low', 'medium', 'medium', 'low', 'high']
            }
            
            sample_df = pd.DataFrame(sample_data)
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(sample_df)
        
        # Set default columns for sample data
        ground_truth_col = 'actual'
        predicted_col = 'predicted'
        
        # Calculate and display metrics for sample data
        y_true = sample_df[ground_truth_col].values
        y_pred = sample_df[predicted_col].values
        
        # Convert to numeric if needed
        if data_type == "Multi-class Classification":
            # Create mapping
            unique_values = sorted(list(set(y_true) | set(y_pred)))
            value_map = {val: i for i, val in enumerate(unique_values)}
            reverse_map = {i: val for val, i in value_map.items()}
            
            y_true_mapped = np.array([value_map[val] for val in y_true])
            y_pred_mapped = np.array([value_map[val] for val in y_pred])
            
            y_true = y_true_mapped
            y_pred = y_pred_mapped
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Display confusion matrix visualization
            st.subheader("Sample Confusion Matrix")
            
            # Create DataFrame for better visualization
            cm_df = pd.DataFrame(cm, 
                index=[f'True: {reverse_map[i]}' for i in range(len(unique_values))],
                columns=[f'Pred: {reverse_map[i]}' for i in range(len(unique_values))])
            
            st.dataframe(cm_df)
            
            # Display metrics
            f1_macro = f1_score(y_true, y_pred, average='macro')
            st.metric("F1 Score (Macro)", f"{f1_macro:.4f}")
        else:
            # Calculate confusion matrix for binary case
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            f1 = f1_score(y_true, y_pred)
            
            # Display confusion matrix visualization
            st.subheader("Sample Confusion Matrix")
            cm_df = pd.DataFrame([
                [tp, fp],
                [fn, tn]
            ], 
            index=['Actual Positive', 'Actual Negative'],
            columns=['Predicted Positive', 'Predicted Negative'])
            
            st.dataframe(cm_df)
            
            # Display metrics
            st.metric("F1 Score", f"{f1:.4f}")
        
        st.write("This is just a sample. Upload your own data to calculate metrics for your specific classification problem.")

# Add instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use this app:
1. Upload a CSV or Excel file containing your classification data
2. Select the columns containing ground truth and predicted values
3. Click "Calculate Metrics" to see the results
4. Review the detailed metrics and explanations to understand your model's performance

Your data should have at least two columns:
- One column with the actual (ground truth) class labels
- One column with the predicted class labels

This app supports both binary classification (two classes) and multi-class classification (three or more classes).
For multi-class data, metrics are calculated using macro averaging by default.
""")
