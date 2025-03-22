import pandas as pd
from wordcloud import WordCloud

# Load the dataset
# df = pd.read_csv('data.csv')

# 修改为本地数据文件
df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_22.csv')

# Create a new dataframe with only the question pairs
question_pairs = df[['question1', 'question2']].copy()

# Create a new dataframe with only the duplicate question pairs
duplicates = question_pairs[question_pairs.duplicated()]

# Create a new dataframe with only the non-duplicate question pairs
non_duplicates = question_pairs[~question_pairs.duplicated()]

# Create a word cloud of the duplicate question pairs
duplicate_word_cloud = WordCloud(
    width=800,
    height=600,
    margin=10,
    background_color='white',
    max_words=500,
    stopwords=['the', 'of', 'and', 'a', 'an', 'in'],
    min_font_size=10
).generate(duplicates['question1'] + duplicates['question2'])

# Create a word cloud of the non-duplicate question pairs
non_duplicate_word_cloud = WordCloud(
    width=800,
    height=600,
    margin=10,
    background_color='white',
    max_words=500,
    stopwords=['the', 'of', 'and', 'a', 'an', 'in'],
    min_font_size=10
).generate(non_duplicates['question1'] + non_duplicates['question2'])

# Plot the word clouds
plt.figure(figsize=(15, 10))
plt.subplot(121)
plt.imshow(duplicate_word_cloud)
plt.title('Duplicate Question Pairs')
plt.axis('off')

plt.subplot(122)
plt.imshow(non_duplicate_word_cloud)
plt.title('Non-Duplicate Question Pairs')
plt.axis('off')

plt.show()