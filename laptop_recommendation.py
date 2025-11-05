import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LaptopRecommender:
    def __init__(self, data_path):
        # Load dataset
        self.data = pd.read_csv("laptop_data.csv")

        # Combine key text features into a single column
        self.data["combined_features"] = (
            self.data["Brand"].astype(str) + " " +
            self.data["Processor"].astype(str) + " " +
            self.data["RAM"].astype(str) + " " +
            self.data["Storage"].astype(str) + " " +
            self.data["GPU"].astype(str) + " " +
            self.data["Usage"].astype(str)
        )

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data["combined_features"])

    def recommend(self, brand, usage, min_price, max_price, top_n=5):
        """Return top N recommended laptops based on similarity"""
        # Filter by price
        filtered = self.data[(self.data["Price"] >= min_price) & (self.data["Price"] <= max_price)]
        if filtered.empty:
            return pd.DataFrame({"Message": ["No laptops found in this price range."]})

        # Create query text
        query = f"{brand} {usage}"
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query and filtered dataset
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix[filtered.index])

        # Get top N matches
        top_indices = similarity_scores[0].argsort()[-top_n:][::-1]

        # Return relevant details
        return filtered.iloc[top_indices][["Name", "Brand", "Processor", "RAM", "Storage", "GPU", "Price", "Usage"]]


if __name__ == "__main__":
    print(" Welcome to the Laptop Recommendation System \n")

    data_path = "laptops.csv"  # Make sure your CSV file is in the same folder
    recommender = LaptopRecommender(data_path)

    # Take user inputs
    brand = input("Enter preferred brand (e.g., HP, Dell, Lenovo): ").strip()
    usage = input("Enter your usage type (e.g., gaming, office, student): ").strip()
    min_price = int(input("Enter your minimum budget (e.g., 40000): "))
    max_price = int(input("Enter your maximum budget (e.g., 100000): "))

    print("\n Finding best laptops for you...\n")

    # Get recommendations
    results = recommender.recommend(brand, usage, min_price, max_price, top_n=5)

    # Display results
    if "Message" in results.columns:
        print(results["Message"].values[0])
    else:
        print("Top Recommended Laptops:\n")
        print(results.to_string(index=False))
