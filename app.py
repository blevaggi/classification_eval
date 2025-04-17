import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to display multiclass metrics - defined BEFORE it's called
def display_multiclass_metrics(y_true, y_pred, cm, label_map, num_classes, threshold):
    # Calculate multiclass metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Get per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Display confusion matrix
    st.subheader("Results Table")
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, 
        index=[f'Actual: {label_map[i]}' for i in range(num_classes)],
        columns=[f'Predicted: {label_map[i]}' for i in range(num_classes)])
    
    st.dataframe(cm_df)
    
    # Display overall metrics
    st.subheader("Overall Accuracy Measures")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        # Overall Score
        st.metric("Overall Accuracy (F1) Score", f"{f1_macro:.4f}")
        st.markdown("""
        **What is the Overall Accuracy (F1) Score?**  
        This score combines how good our model is at correctly identifying items (precision) 
        and how good it is at finding all the items it should find (recall).
        
        A score of 1.0 means perfect performance, while 0.0 means the worst possible performance.
        """)
    
    with metrics_col2:
        # Success rates
        st.metric("Correct Classification Rate (Precision)", f"{precision_macro:.4f}")
        st.metric("Discovery Rate (Recall)", f"{recall_macro:.4f}")
        st.markdown("""
        **What do these rates mean?**  
        - **Correct Classification Rate**: When the model predicts something belongs to a category, how often is it right?
        - **Discovery Rate**: How many items that should be found in each category are actually found?
        """)
    
    # Display per-class metrics
    st.subheader("Performance by Category")
    
    # Create metrics table
    class_metrics = []
    
    for i in range(num_classes):
        class_name = label_map[i]
        if str(i) in report:
            class_data = report[str(i)]
            class_metrics.append({
                'Category': class_name,
                'Correct Classification Rate': f"{class_data['precision']:.4f}",
                'Discovery Rate': f"{class_data['recall']:.4f}",
                'Overall Score': f"{class_data['f1-score']:.4f}",
                'Number of Items': int(class_data['support'])
            })
    
    class_metrics_df = pd.DataFrame(class_metrics)
    st.dataframe(class_metrics_df)
    
    # Explanation of per-class metrics
    st.markdown("""
    **What do these numbers mean for each category?**
    - **Correct Classification Rate**: When the model says an item belongs to this category, how often is it right?
    - **Discovery Rate**: How many items that actually belong to this category did the model correctly find?
    - **Overall Score**: A balanced measure that combines the above two rates
    - **Number of Items**: How many items of this category were in your data
    
    Lower scores show which categories your model has trouble with.
    """)
    
    # Display confusion matrix explanation
    st.markdown("""
    **How to read the Results Table:**
    - Each row shows the actual category
    - Each column shows what the model predicted
    - The diagonal numbers (top-left to bottom-right) show correct predictions
    - All other numbers show mistakes
    
    For example, if you see a number "5" at the row "Actual: low" and column "Predicted: medium", 
    it means 5 items that were actually "low" were incorrectly classified as "medium".
    """)

    # Evaluate against risk tolerance threshold
    st.subheader("Risk Assessment")
    
    # Compare the F1 score with the threshold
    if f1_macro >= threshold:
        st.success(f"✅ Your model meets the required accuracy for your selected quality requirement! (Score: {f1_macro:.4f}, Required: {threshold:.2f})")
        st.markdown(f"""
        **Recommendation**: Your model's accuracy is acceptable for your selected quality requirement.
        
        You can:
        - Deploy this model with confidence
        - Monitor performance over time to ensure it maintains this level of accuracy
        - Consider saving this model as your baseline
        """)
    else:
        st.error(f"❌ Your model does not meet the required accuracy. (Score: {f1_macro:.4f}, Required: {threshold:.2f})")
        
        # Show targeted improvement suggestions based on issues detected
        if precision_macro < threshold:
            st.warning(f"🔍 **Issue detected**: Low Correct Classification Rate ({precision_macro:.4f})")
            st.markdown(f"""
            **Targeted fix**: Your model is saying "yes" too often, creating false alarms.
            
            Try:
            - Making your model more selective about what belongs to a category
            - Increasing the confidence threshold for positive predictions
            - Adding more examples of negative cases to your training data
            """)
        
        if recall_macro < threshold:
            st.warning(f"🔍 **Issue detected**: Low Discovery Rate ({recall_macro:.4f})")
            st.markdown(f"""
            **Targeted fix**: Your model is missing too many items it should find.
            
            Try:
            - Making your model less strict about what belongs to a category
            - Decreasing the confidence threshold for positive predictions
            - Adding more examples of positive cases to your training data
            """)
        
        # Check for class imbalance
        if len(class_metrics) > 1:
            min_support = min([item['Number of Items'] for item in class_metrics])
            max_support = max([item['Number of Items'] for item in class_metrics])
            if max_support > min_support * 3:  # If max class is 3x larger than min class
                st.warning("🔍 **Issue detected**: Uneven number of examples between categories")
                st.markdown("""
                **Targeted fix**: Some categories have too few examples compared to others.
                
                Try:
                - Collecting more examples of the rare categories
                - Using data balancing techniques like oversampling or undersampling
                - Using weighted loss functions during training
                """)
        
        # Find the most confused classes
        if num_classes > 2:
            confused_pairs = []
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j and cm[i, j] > 0:
                        confused_pairs.append((i, j, cm[i, j]))
            
            if confused_pairs:
                confused_pairs.sort(key=lambda x: x[2], reverse=True)
                top_confusion = confused_pairs[0]
                class_i, class_j, confusion_count = top_confusion
                
                st.warning(f"🔍 **Issue detected**: Confusion between categories")
                st.markdown(f"""
                **Targeted fix**: Your model often confuses **{label_map[class_i]}** with **{label_map[class_j]}** 
                ({confusion_count} times).
                
                Try:
                - Finding better features that distinguish between these specific categories
                - Adding more training examples of these categories
                - Creating a specialized model just for these hard-to-distinguish categories
                """)
    
    # Always show general tips
    with st.expander("General Tips for Improving Your Results", expanded=f1_macro < threshold):
        st.markdown("""
        - **Low Correct Classification Rate** (many false alarms): Your model is saying "yes" too often. 
          Try making it more selective about what it identifies as belonging to a category.
        
        - **Low Discovery Rate** (many misses): Your model is missing too many items it should find. 
          Try making it less strict about what it considers to belong to a category.
        
        - **Low Overall Score**: Consider adding more examples, finding better features that distinguish 
          between categories, or trying a different type of model.
        
        - **Uneven Number of Items**: If some categories have very few examples compared to others, 
          try to collect more examples of the rare categories.
        
        - **Confusion Between Specific Categories**: Look at the Results Table to see which categories 
          are getting mixed up. Focus on finding ways to better distinguish between those specific categories.
        """)
    
    # Return the F1 score for evaluation elsewhere
    return f1_macro

# Set page title and configuration
st.set_page_config(page_title="Classification Results Viewer", layout="wide")
st.title("Classification Results Viewer")
st.write("Upload your prediction results to see how well your model is performing.")

# Add risk tolerance slider
st.sidebar.subheader("Quality Requirements")
accuracy_requirement = st.sidebar.slider(
    "Select your accuracy requirement (%):",
    min_value=60,
    max_value=90,
    value=75,
    step=1,
    help="Higher values require better model performance"
)

# Map the accuracy percentage to a risk tolerance level
if 60 <= accuracy_requirement <= 70:
    risk_tolerance = "High"
    st.sidebar.markdown("**High Risk Tolerance (60-70%)**: Your application can accept more errors. Examples: Initial data exploration, non-critical filtering tasks, preliminary research.")
elif 71 <= accuracy_requirement <= 84:
    risk_tolerance = "Medium"
    st.sidebar.markdown("**Medium Risk Tolerance (71-84%)**: Your application requires good accuracy. Examples: Customer recommendations, content categorization, basic automation.")
else:  # 85-90
    risk_tolerance = "Low"
    st.sidebar.markdown("**Low Risk Tolerance (85-90%)**: Your application requires high accuracy. Examples: Medical diagnosis, safety systems, financial compliance.")

# Define threshold based on accuracy requirement
threshold = accuracy_requirement / 100.0

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
            ground_truth_col = st.selectbox("Select the column with actual values (ground truth)", columns)
        
        with col2:
            predicted_col = st.selectbox("Select the column with predicted values", columns, index=min(1, len(columns)-1))
        
        # Button to calculate metrics
        if st.button("Show Results"):
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
                
                st.info(f"Categories found in your data: {', '.join(unique_values)}")
                
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
                    binary_mode = st.radio("Choose how to calculate results:", ["Average across both categories", "Focus on one category"])
                    
                    if binary_mode == "Focus on one category":
                        # For binary case, calculate metrics treating class 1 as positive
                        tn, fp, fn, tp = cm.ravel()
                        
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        # Display confusion matrix visualization
                        st.subheader("Results Table")
                        
                        # Create DataFrame for better visualization
                        cm_df = pd.DataFrame(cm, 
                            index=[f'Actual: {reverse_map[i]}' for i in range(num_classes)],
                            columns=[f'Predicted: {reverse_map[i]}' for i in range(num_classes)])
                        
                        st.dataframe(cm_df)
                        
                        # Display binary metrics
                        st.subheader("Results Summary")
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            # Overall Score
                            st.metric("Overall Accuracy Score", f"{f1:.4f}")
                            st.markdown("""
                            **What is the Overall Accuracy Score?**  
                            This score combines how good our model is at correctly identifying items (precision) 
                            and how good it is at finding all the items it should find (recall).
                            
                            A score of 1.0 means perfect performance, while 0.0 means the worst possible performance.
                            """)
                            
                            # Discovery Rate (Recall/Sensitivity)
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            st.metric("Discovery Rate", f"{tpr:.4f}")
                            st.markdown("""
                            **What is the Discovery Rate?**  
                            Out of all the items that actually belong to your main category, 
                            what percentage did the model correctly identify?
                            
                            Higher is better. A score of 1.0 means the model found all the items it should have found.
                            """)
                            
                            # Rejection Rate (Specificity)
                            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                            st.metric("Rejection Rate", f"{tnr:.4f}")
                            st.markdown("""
                            **What is the Rejection Rate?**  
                            Out of all the items that don't belong to your main category, 
                            what percentage did the model correctly reject?
                            
                            Higher is better. A score of 1.0 means the model correctly rejected everything it should have.
                            """)
                        
                        with metrics_col2:
                            # True Positives
                            st.metric("Correct Identifications", tp)
                            st.markdown("""
                            **What are Correct Identifications?**  
                            Items that actually belong to your main category and were 
                            correctly identified as such by the model.
                            """)
                            
                            # False Positives
                            st.metric("False Alarms", fp)
                            st.markdown("""
                            **What are False Alarms?**  
                            Items that don't actually belong to your main category, 
                            but the model incorrectly said they do.
                            """)
                            
                            # False Negatives
                            st.metric("Misses", fn)
                            st.markdown("""
                            **What are Misses?**  
                            Items that actually belong to your main category, 
                            but the model failed to identify them.
                            """)
                            
                            # True Negatives
                            st.metric("Correct Rejections", tn)
                            st.markdown("""
                            **What are Correct Rejections?**  
                            Items that don't belong to your main category, 
                            and the model correctly kept them out.
                            """)
                        
                        # Additional metrics
                        st.subheader("Additional Information")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            # Precision
                            st.metric("Correct Classification Rate", f"{precision:.4f}")
                            st.markdown("""
                            **What is the Correct Classification Rate?**  
                            When the model says an item belongs to your main category, 
                            how often is it right?
                            
                            Higher is better. A score of 1.0 means every prediction was correct.
                            """)
                        
                        with col4:
                            # Accuracy
                            accuracy = (tp + tn) / (tp + tn + fp + fn)
                            st.metric("Simple Accuracy", f"{accuracy:.4f}")
                            st.markdown("""
                            **What is Simple Accuracy?**  
                            The percentage of all predictions (both categories) that were correct.
                            
                            A score of 1.0 means every single prediction was correct.
                            """)
                        
                        # Evaluate against risk tolerance threshold
                        st.subheader("Risk Assessment")
                        
                        # Compare the F1 score with the threshold
                        if f1 >= threshold:
                            st.success(f"✅ Your model meets the required accuracy for your selected quality requirement! (Score: {f1:.4f}, Required: {threshold:.2f})")
                            st.markdown(f"""
                            **Recommendation**: Your model's accuracy is acceptable for your selected quality requirement.
                            
                            You can:
                            - Deploy this model with confidence
                            - Monitor performance over time to ensure it maintains this level of accuracy
                            - Consider saving this model as your baseline
                            """)
                        else:
                            st.error(f"❌ Your model does not meet the required accuracy. (Score: {f1:.4f}, Required: {threshold:.2f})")
                            
                            # Show targeted improvement suggestions based on issues detected
                            if precision < threshold:
                                st.warning(f"🔍 **Issue detected**: Low Correct Classification Rate ({precision:.4f})")
                                st.markdown(f"""
                                **Targeted fix**: Your model is saying "yes" too often, creating false alarms.
                                
                                Try:
                                - Making your model more selective about what belongs to a category
                                - Increasing the confidence threshold for positive predictions
                                - Adding more examples of negative cases to your training data
                                """)
                            
                            if recall < threshold:
                                st.warning(f"🔍 **Issue detected**: Low Discovery Rate ({recall:.4f})")
                                st.markdown(f"""
                                **Targeted fix**: Your model is missing too many items it should find.
                                
                                Try:
                                - Making your model less strict about what belongs to a category
                                - Decreasing the confidence threshold for positive predictions
                                - Adding more examples of positive cases to your training data
                                """)
                        
                        # Always show general tips
                        with st.expander("General Tips for Improving Your Results", expanded=f1 < threshold):
                            st.markdown("""
                            - **Low Correct Classification Rate** (many false alarms): Your model is saying "yes" too often. 
                              Try making it more selective about what it identifies as belonging to a category.
                            
                            - **Low Discovery Rate** (many misses): Your model is missing too many items it should find. 
                              Try making it less strict about what it considers to belong to a category.
                            
                            - **Low Overall Score**: Consider adding more examples, finding better features that distinguish 
                              between categories, or trying a different type of model.
                            
                            - **Uneven Number of Items**: If some categories have very few examples compared to others, 
                              try to collect more examples of the rare categories.
                            
                            - **Confusion Between Specific Categories**: Look at the Results Table to see which categories 
                              are getting mixed up. Focus on finding ways to better distinguish between those specific categories.
                            """)
                    else:
                        # Display macro average metrics
                        _ = display_multiclass_metrics(y_true, y_pred, cm, reverse_map, num_classes, threshold)
                else:
                    # Multiclass classification case
                    _ = display_multiclass_metrics(y_true, y_pred, cm, reverse_map, num_classes, threshold)
                
                # Results explained with an analogy
                st.subheader("Understanding These Results: A Simple Analogy")
                st.markdown("""
                **The Fishing Analogy:**
                
                Imagine you're fishing in a lake with both fish and logs. Your task is to catch only fish:
                
                - **Correct Classification Rate** is like asking: "Of everything I caught, what percentage were actually fish?" 
                  If you caught 8 fish and 2 logs, your correct classification rate is 80%.
                
                - **Discovery Rate** is like asking: "What percentage of all fish in the lake did I manage to catch?"
                  If there were 10 fish in the lake and you caught 8, your discovery rate is 80%.
                
                - **Overall Accuracy Score** balances these two rates. It will be high only if both are reasonably high.
                  A high score means you're not only catching most of the fish (high discovery rate) but also avoiding catching logs (high correct classification rate).
                
                **For Multiple Categories:**
                
                If you're trying to catch different types of fish (trout, bass, etc.) while avoiding logs:
                - The app calculates how well you're doing for each type of fish
                - Then gives you an average score across all types
                """)
                
            except Exception as e:
                st.error(f"Error calculating results: {str(e)}")
                st.info("Debug information: Please check your data types and ensure you have selected the correct columns.")
        
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    # Display sample data option
    if st.checkbox("Try with sample data"):
        data_type = st.radio("Choose sample data type:", ["Two Categories", "Multiple Categories"])
        
        if data_type == "Two Categories":
            st.info("Loading sample data with two categories...")
            
            # Create sample binary data
            sample_data = {
                'actual': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'predicted': [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
            }
            
            sample_df = pd.DataFrame(sample_data)
        else:
            st.info("Loading sample data with multiple categories...")
            
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
        if data_type == "Multiple Categories":
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
            st.subheader("Sample Results Table")
            
            # Create DataFrame for better visualization
            cm_df = pd.DataFrame(cm, 
                index=[f'Actual: {reverse_map[i]}' for i in range(len(unique_values))],
                columns=[f'Predicted: {reverse_map[i]}' for i in range(len(unique_values))])
            
            st.dataframe(cm_df)
            
            # Calculate F1 score
            f1_macro = f1_score(y_true, y_pred, average='macro')
            
            # Display metrics
            st.metric("Overall Accuracy Score", f"{f1_macro:.4f}")
            
            # Evaluate against risk tolerance threshold
            st.subheader("Risk Assessment")
            
            # Compare F1 score with threshold
            if f1_macro >= threshold:
                st.success(f"✅ Your model meets the required accuracy for your selected quality requirement! (Score: {f1_macro:.4f}, Required: {threshold:.2f})")
            else:
                st.error(f"❌ Your model does not meet the required accuracy. (Score: {f1_macro:.4f}, Required: {threshold:.2f})")
                
            st.write("This is just a sample. Upload your own data to see results for your specific prediction problem.")
        else:
            # Calculate confusion matrix for binary case
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            f1 = f1_score(y_true, y_pred)
            
            # Display confusion matrix visualization
            st.subheader("Sample Results Table")
            cm_df = pd.DataFrame(cm,
                index=['Actual: 1', 'Actual: 0'],
                columns=['Predicted: 1', 'Predicted: 0'])
            
            st.dataframe(cm_df)
            
            # Display metrics
            st.metric("Overall Accuracy Score", f"{f1:.4f}")
            
            # Evaluate against risk tolerance threshold
            st.subheader("Risk Assessment")
            
            # Compare F1 score with threshold
            if f1 >= threshold:
                st.success(f"✅ Your model meets the required accuracy for your selected quality requirement! (Score: {f1:.4f}, Required: {threshold:.2f})")
            else:
                st.error(f"❌ Your model does not meet the required accuracy. (Score: {f1:.4f}, Required: {threshold:.2f})")
                
            st.write("This is just a sample. Upload your own data to see results for your specific prediction problem.")

# Add instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use this app:
1. Upload a CSV or Excel file containing your prediction results
2. Select the columns containing actual values and predicted values
3. Click "Show Results" to see how well your predictions performed
4. Review the detailed results and explanations to understand your model's performance

Your data should have at least two columns:
- One column with the actual (correct) category labels
- One column with the predicted category labels from your model

This app works with both two-category data (like yes/no, spam/not spam) and 
multi-category data (like low/medium/high, or different product categories).
""")
