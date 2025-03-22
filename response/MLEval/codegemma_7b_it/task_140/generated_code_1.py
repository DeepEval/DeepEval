from pyspark.sql.functions import col, add

def method():
    # Create a DataFrame
    df = spark.createDataFrame([(1, 2), (3, 4), (5, 6)], ["x", "y"])

    # Add x and y element-wise
    df_with_sum = df.withColumn("sum", add(col("x"), col("y")))

    # Return the DataFrame with the added column
    return df_with_sum

# Call the method
df_with_sum = method()

# Print the DataFrame
df_with_sum.show()