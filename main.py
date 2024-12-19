from pyspark.sql.functions import col, when, avg
from pyspark.sql import SparkSession;
import numpy as np;
from pyspark.ml.feature import StringIndexer, VectorAssembler
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def Readrawdata(path):
    spark = SparkSession.builder \
        .appName("appName") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Airbnb.Airbnb_listings") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Airbnb.Airbnb_listings") \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:10.4.0') \
        .getOrCreate()
    df = spark.read.csv(path, header=True, inferSchema=True)
    df.write.format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "Airbnb_listings") \
        .mode("overwrite") \
        .save()


    df.printSchema()
    df.show()
    df.write.format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "Bronze") \
        .mode("overwrite") \
        .save()
    df_bronze = spark.read \
        .format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "Bronze") \
        .load()
    num_rows = df_bronze.count()  # Get number of rows
    num_columns = len(df_bronze.columns)  # Get number of columns

    print(f"Number of bronze Rows: {num_rows}")
    print(f"Number of bronze Columns: {num_columns}")
    return df_bronze, spark



# Press the green button in the gutter to run the script.


def Datacleaning(data_bronze):
    data_bronze=data_bronze.drop_duplicates(['Listings id']);
    print(type(data_bronze))
    df_cleaned = data_bronze.fillna({
        'Last year reviews': 0,
        'Host number of listings': 0,
        'Beds number': 0,
        'Bedrooms number': 0,
        'Maximum allowed guests': 0,
        'Price': 0,
        'Total reviews': 0,
        'Rating score': 0,
        'Accuracy score': 0,
        'Cleanliness score': 0,
        'Checkin score': 0,
        'Communication score': 0,
        'Location score': 0,
        'Value for money score': 0,
        'Reviews per month': 0,
        'Bathrooms number': 0
    })

    # For categorical fields, fill missing values with 'Unknown'
    df_cleaned = df_cleaned.fillna({
        'Host is superhost': 'Unknown',
        'Neighbourhood': 'Unknown',
        'Property type': 'Unknown',
        'City': 'Unknown',
        'Season': 'Unknown',
        'Bathrooms type': 'Unknown',
        'Coordinates': 'Unknown',
        'Date of scraping': '1970-01-01'  # Default date if missing
    })
    df_cleaned = df_cleaned.filter(col("Price") >= 0)

    df_cleaned.show(5)
    print(df_cleaned.columns)

    return df_cleaned


def writeSilver(df_cleaned, spark):
    df_cleaned.write.format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "silver") \
        .mode("overwrite") \
        .save()
    df_silver = spark.read \
        .format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "silver") \
        .load()
    num_rows = df_silver.count()  # Get number of rows
    num_columns = len(df_silver.columns)  # Get number of columns

    print(f"Number of Silver Rows: {num_rows}")
    print(f"Number of Silver Columns: {num_columns}")
    return df_silver


def dropcolumns(df_silver, spark):
    columns_to_drop = ['Date of scraping', 'Coordinates', 'Bathrooms type', 'Host since']
    df_dropped = df_silver.drop(*columns_to_drop)
    print(df_dropped.columns)
    df_dropped.write.format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "gold") \
        .mode("overwrite") \
        .save()
    df_gold = spark.read \
        .format("mongodb") \
        .option("uri", "mongodb://localhost:27017/") \
        .option("database", "Airbnb") \
        .option("collection", "gold") \
        .load()
    num_rows = df_gold.count()  # Get number of rows
    num_columns = len(df_gold.columns)  # Get number of columns

    print(f"Number of gold Rows: {num_rows}")
    print(f"Number of gold Columns: {num_columns}")

    return df_gold


def Analysis(df_gold):
    avg_rating_city = df_gold.groupBy("City").agg(avg("Rating score").alias("Average Rating"))
    avg_rating_city = avg_rating_city.toPandas();
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rating_city["City"], avg_rating_city["Average Rating"], marker='o', linestyle='-', color='green')
    plt.xlabel("City")
    plt.ylabel("Average Rating")
    plt.title("Average Rating by City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    property_analysis = df_gold.groupBy("Property type").agg(
       avg("Price").alias("Average Price"),
       avg("Rating score").alias("Average Rating")
    ).toPandas()

    # Sort by average price for consistent visualization
    property_analysis = property_analysis.sort_values(by="Average Price", ascending=False)

    # Bar Graph: Average Price by Property Type
    plt.figure(figsize=(12, 6))
    plt.bar(
       property_analysis["Property type"],
       property_analysis["Average Price"],
       color='skyblue',
       alpha=0.8
    )
    plt.title("Average Price by Property Type", fontsize=16)
    plt.xlabel("Property Type", fontsize=12)
    plt.ylabel("Average Price", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    city_analysis = df_gold.groupBy("City").agg(
       avg("Rating score").alias("Average Rating"),
       avg("Cleanliness score").alias("Average Cleanliness")
    ).toPandas()

    # Sort cities by average rating
    city_analysis = city_analysis.sort_values(by="Average Rating", ascending=False)

    # Define bar width for grouped bar chart
    bar_width = 0.35
    indices = np.arange(len(city_analysis["City"]))

    # Plot grouped bar chart
    plt.figure(figsize=(14, 7))
    plt.bar(
       indices,
       city_analysis["Average Rating"],
       bar_width,
       label="Average Rating",
       color="skyblue",
       alpha=0.8
    )
    plt.bar(
       indices + bar_width,
       city_analysis["Average Cleanliness"],
       bar_width,
       label="Average Cleanliness",
       color="orange",
       alpha=0.8
    )

    # Add labels and title
    plt.title("Average Rating vs. Cleanliness Score by City", fontsize=16)
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(indices + bar_width / 2, city_analysis["City"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()
    cluster_features = ['Rating score', 'Price', 'Reviews per month', 'Total reviews']
    df_cluster = df_gold.select(cluster_features).na.fill(0)

    # Assemble features into a single vector
    assembler_cluster = VectorAssembler(inputCols=cluster_features, outputCol="features")
    df_cluster_vector = assembler_cluster.transform(df_cluster)

    # Apply K-means
    kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="cluster")
    kmeans_model = kmeans.fit(df_cluster_vector)

    # Add cluster predictions
    df_cluster_result = kmeans_model.transform(df_cluster_vector)

    # Evaluate clustering
    evaluator = ClusteringEvaluator(predictionCol="cluster")
    silhouette = evaluator.evaluate(df_cluster_result)
    print(f"Silhouette Score: {silhouette}")
    # Convert cluster result to Pandas
    df_cluster_pd = df_cluster_result.select("Price", "Rating score", "cluster").toPandas()

    # Scatter plot
    plt.figure(figsize=(10, 6))
    for cluster_id in df_cluster_pd['cluster'].unique():
        cluster_data = df_cluster_pd[df_cluster_pd['cluster'] == cluster_id]
        plt.scatter(cluster_data['Price'], cluster_data['Rating score'], label=f"Cluster {cluster_id}")

    plt.title("Clusters of Listings based on Rating and Price")
    plt.xlabel("Price")
    plt.ylabel("Rating score")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    path = "/Users/chinmai/Downloads/airbnb.csv"
    data_bronze, spark = Readrawdata(path)
    df_cleaned = Datacleaning(data_bronze)
    df_silver = writeSilver(df_cleaned, spark)
    df_gold = dropcolumns(df_silver, spark)
    Analysis(df_gold)
    # d

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
