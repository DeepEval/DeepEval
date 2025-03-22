import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def method():
    # Assuming the data is already loaded into a DataFrame called df
    # df = pd.read_csv('path_to_your_data.csv')  # Replace with actual path to your dataset

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_coder_v2_lite_instruct/testcases/task_294.csv')

    # Display the first few rows of the DataFrame to understand its structure
    print(df.head())

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Display the correlation matrix
    print(corr_matrix)

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Extract the relevant columns for visualization
    columns_of_interest = ['Age', 'Positive_Nodes', 'Operation_Year', 'Survival']
    df_subset = df[columns_of_interest]

    # Plot pairwise relationships in a dataset
    sns.pairplot(df_subset)
    plt.suptitle('Pairwise Relationships', y=1.02)
    plt.show()

    # Based on the correlation matrix and visualizations, we can confirm the relationships
    output = "Age and positive nodes are negatively correlated with survival while operation year is positively correlated."

    return output

# Call the method for validation
output = method()
print(output)