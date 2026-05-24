"""
Cyberbullying Detection System - Complete Analysis
SafeNet Analytics Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Define stop words manually (avoiding NLTK network issues)
STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "might",
    "more",
    "most",
    "must",
    "my",
    "myself",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

# ============================================================================
# PART A: DATA PREPARATION AND MODEL CHOICE
# ============================================================================


class CyberbullyingDetector:
    """
    A responsible cyberbullying detection system with emphasis on
    transparency, fairness, and interpretability.
    """

    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

    def explore_data(self):
        """Initial data exploration"""
        print("=" * 70)
        print("DATASET OVERVIEW")
        print("=" * 70)
        print(f"Total samples: {len(self.df)}")
        print(f"\nClass Distribution:")
        print(self.df["cyberbullying_type"].value_counts())
        print(f"\nClass Balance Analysis:")
        class_counts = self.df["cyberbullying_type"].value_counts()
        print(
            f"Min class: {class_counts.min()} ({(class_counts.min()/len(self.df)*100):.2f}%)"
        )
        print(
            f"Max class: {class_counts.max()} ({(class_counts.max()/len(self.df)*100):.2f}%)"
        )

        # Sample tweets from each category
        print("\n" + "=" * 70)
        print("SAMPLE TWEETS BY CATEGORY")
        print("=" * 70)
        for category in self.df["cyberbullying_type"].unique():
            print(f"\n{category.upper()}:")
            samples = self.df[self.df["cyberbullying_type"] == category].sample(2)
            for idx, tweet in enumerate(samples["tweet_text"].values, 1):
                print(f"  {idx}. {tweet[:100]}...")

    def clean_text(self, text):
        """
        Text preprocessing with careful consideration of context
        - Preserves some emotionally significant words
        - Anonymizes usernames
        - Handles URLs and special characters
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Anonymize usernames (privacy protection)
        text = re.sub(r"@\w+", "<user>", text)

        # Remove URLs (while preserving context)
        text = re.sub(r"http\S+|www\S+", "<url>", text)

        # Remove hashtag symbol but keep the word (context preservation)
        text = re.sub(r"#", "", text)

        # Remove special characters but keep emotionally relevant punctuation
        text = re.sub(r"[^\w\s!?.]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def advanced_preprocessing(self, text):
        """
        Advanced preprocessing with selective stop word removal
        - Keeps negations (not, never, no) as they're crucial for sentiment
        - Keeps intensifiers (very, really, so)
        """
        # Simple tokenization by splitting on whitespace
        tokens = text.split()

        # Custom stop words list (excluding sentiment-critical words)
        stop_words = STOP_WORDS.copy()
        # Keep important words for context
        keep_words = {
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nor",
            "very",
            "really",
            "so",
            "too",
        }
        stop_words = stop_words - keep_words

        # Remove stop words while preserving context
        tokens = [word for word in tokens if word not in stop_words or len(word) > 2]

        return " ".join(tokens)

    def prepare_data(self):
        """
        Complete data preparation pipeline with bias mitigation
        """
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)

        # 1. Create a copy to preserve original data
        self.df_processed = self.df.copy()

        # 2. Check for duplicates
        duplicates = self.df_processed.duplicated().sum()
        print(f"Duplicate tweets found: {duplicates}")
        if duplicates > 0:
            self.df_processed = self.df_processed.drop_duplicates()
            print(f"Removed {duplicates} duplicates")

        # 3. Text cleaning
        print("\nApplying text preprocessing...")
        self.df_processed["cleaned_text"] = self.df_processed["tweet_text"].apply(
            self.clean_text
        )
        self.df_processed["processed_text"] = self.df_processed["cleaned_text"].apply(
            self.advanced_preprocessing
        )

        # 4. Remove very short tweets (likely not meaningful)
        min_length = 3
        before_filter = len(self.df_processed)
        self.df_processed = self.df_processed[
            self.df_processed["processed_text"].str.split().str.len() >= min_length
        ]
        print(
            f"Removed {before_filter - len(self.df_processed)} tweets with < {min_length} words"
        )

        # 5. Check class balance after preprocessing
        print("\nClass distribution after preprocessing:")
        print(self.df_processed["cyberbullying_type"].value_counts())

        # 6. Bias Analysis: Check for representative language distribution
        print("\nBias Check - Top words by category:")
        self.analyze_category_vocabulary()

        return self.df_processed

    def analyze_category_vocabulary(self):
        """
        Analyze vocabulary distribution to identify potential biases
        """
        for category in self.df_processed["cyberbullying_type"].unique()[
            :3
        ]:  # Show 3 categories
            category_text = " ".join(
                self.df_processed[self.df_processed["cyberbullying_type"] == category][
                    "processed_text"
                ].values
            )
            words = category_text.split()
            common_words = Counter(words).most_common(10)
            print(f"\n  {category}: {[word for word, _ in common_words]}")

    def create_binary_classifier(self):
        """
        Create binary classification (cyberbullying vs not)
        More interpretable for initial deployment
        """
        print("\n" + "=" * 70)
        print("BINARY CLASSIFICATION SETUP")
        print("=" * 70)

        # Create binary labels
        self.df_processed["is_cyberbullying"] = (
            self.df_processed["cyberbullying_type"] != "not_cyberbullying"
        ).astype(int)

        print("Binary distribution:")
        print(self.df_processed["is_cyberbullying"].value_counts())
        print(
            f"\nPercentage of cyberbullying: {self.df_processed['is_cyberbullying'].mean()*100:.2f}%"
        )

    def train_model(self, model_type="logistic"):
        """
        Train model with justification for choice

        Model Choice Justification:
        - Logistic Regression: Highly interpretable, fast, good baseline
        - Can examine feature weights to understand what the model learns
        - Suitable for ethical contexts where explainability is crucial
        """
        print("\n" + "=" * 70)
        print(f"MODEL TRAINING - {model_type.upper()}")
        print("=" * 70)

        # Prepare features and labels
        X = self.df_processed["processed_text"]
        y = self.df_processed["is_cyberbullying"]

        # Split data (stratified to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # TF-IDF Vectorization
        # Using character and word n-grams to capture context
        print("\nVectorizing text with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams for context
            min_df=5,  # Word must appear in at least 5 documents
            max_df=0.8,  # Ignore very common words
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Train Logistic Regression
        print("\nTraining Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Handle any remaining class imbalance
            random_state=42,
            C=1.0,  # Regularization to prevent overfitting
        )

        self.model.fit(X_train_vec, y_train)

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_vec = X_test_vec

        print("✓ Model training completed")

        return X_train_vec, X_test_vec, y_train, y_test

    def evaluate_model(self):
        """
        Comprehensive model evaluation with ethical considerations
        """
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # Predictions
        y_pred = self.model.predict(self.X_test_vec)
        y_pred_proba = self.model.predict_proba(self.X_test_vec)

        # Overall metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="binary"
        )

        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(
            f"  Precision: {precision:.4f} (of flagged tweets, how many are actually bullying)"
        )
        print(f"  Recall:    {recall:.4f} (of actual bullying, how many we catch)")
        print(f"  F1-Score:  {f1:.4f} (harmonic mean of precision and recall)")

        # Detailed classification report
        print("\n" + "-" * 70)
        print("Detailed Classification Report:")
        print("-" * 70)
        print(
            classification_report(
                self.y_test,
                y_pred,
                target_names=["Not Cyberbullying", "Cyberbullying"],
                digits=4,
            )
        )

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)

        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("                  Not CB  |  CB")
        print(f"Actual Not CB    {cm[0,0]:6d}  | {cm[0,1]:6d}")
        print(f"Actual CB        {cm[1,0]:6d}  | {cm[1,1]:6d}")

        # Ethical Analysis of Errors
        print("\n" + "=" * 70)
        print("ETHICAL IMPLICATIONS OF MISCLASSIFICATION")
        print("=" * 70)

        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]

        print(f"\nFalse Positives (FP): {false_positives}")
        print("  → Harmless content flagged as bullying")
        print("  → Could lead to unfair content removal or user penalties")
        print("  → May suppress free speech or legitimate discourse")

        print(f"\nFalse Negatives (FN): {false_negatives}")
        print("  → Actual bullying not detected")
        print("  → Allows harmful content to remain visible")
        print("  → Victims continue to experience harm")

        print("\nRecommendation:")
        print("  Given the ethical trade-offs, this system should be used as a")
        print("  FLAGGING TOOL for human review, not for automated decisions.")

        # Feature importance (most predictive words)
        print("\n" + "=" * 70)
        print("MODEL INTERPRETABILITY - Top Predictive Features")
        print("=" * 70)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coef = self.model.coef_[0]

        # Top indicators of cyberbullying
        top_bullying_idx = np.argsort(coef)[-15:]
        print("\nTop 15 indicators of CYBERBULLYING:")
        for idx in reversed(top_bullying_idx):
            print(f"  {feature_names[idx]:20s} (weight: {coef[idx]:6.3f})")

        # Top indicators of non-bullying
        top_non_bullying_idx = np.argsort(coef)[:15]
        print("\nTop 15 indicators of NON-CYBERBULLYING:")
        for idx in top_non_bullying_idx:
            print(f"  {feature_names[idx]:20s} (weight: {coef[idx]:6.3f})")

        return cm, y_pred, y_pred_proba

    def analyze_ambiguous_cases(self, num_cases=10):
        """
        Analyze cases where the model is uncertain
        These require human review
        """
        print("\n" + "=" * 70)
        print("AMBIGUOUS CASE ANALYSIS")
        print("=" * 70)

        y_pred_proba = self.model.predict_proba(self.X_test_vec)

        # Find cases with probability close to 0.5 (uncertain)
        uncertainty = np.abs(y_pred_proba[:, 1] - 0.5)
        uncertain_idx = np.argsort(uncertainty)[:num_cases]

        print(f"\nTop {num_cases} Most Ambiguous Cases:")
        print("(These require human review and context consideration)\n")

        for i, idx in enumerate(uncertain_idx, 1):
            tweet = self.X_test.iloc[idx]
            actual = (
                "CYBERBULLYING" if self.y_test.iloc[idx] == 1 else "NOT CYBERBULLYING"
            )
            prob = y_pred_proba[idx, 1]

            print(f"{i}. Probability: {prob:.3f} | Actual: {actual}")
            print(f"   Tweet: {tweet[:120]}...")
            print()

    def test_on_examples(self):
        """
        Test model on various example cases
        """
        print("\n" + "=" * 70)
        print("TESTING ON EXAMPLE CASES")
        print("=" * 70)

        test_cases = [
            "You're so stupid, nobody likes you",
            "Great job on the presentation today!",
            "lol you're such a loser haha",
            "I disagree with your political opinion",
            "Women belong in the kitchen",
            "That's an interesting perspective, thanks for sharing",
            "Go kill yourself nobody wants you here",
            "Your cooking skills need improvement",
        ]

        for i, text in enumerate(test_cases, 1):
            cleaned = self.advanced_preprocessing(self.clean_text(text))
            vectorized = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(vectorized)[0]
            probability = self.model.predict_proba(vectorized)[0]

            result = "⚠️ CYBERBULLYING" if prediction == 1 else "✓ NOT CYBERBULLYING"
            confidence = probability[1] if prediction == 1 else probability[0]

            print(f'\n{i}. "{text}"')
            print(f"   → {result} (confidence: {confidence:.2%})")

    def generate_evaluation_report(self):
        """
        Generate comprehensive evaluation metrics for report
        """
        results = {
            "dataset_size": len(self.df),
            "processed_size": len(self.df_processed),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "test_accuracy": accuracy_score(
                self.y_test, self.model.predict(self.X_test_vec)
            ),
        }

        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("CYBERBULLYING DETECTION SYSTEM - SafeNet Analytics")
    print("Responsible AI for Digital Safety")
    print("=" * 70)

    # Initialize detector
    # detector = CyberbullyingDetector('/mnt/user-data/uploads/cyberbullying_tweets.csv')

    detector = CyberbullyingDetector("cyberbullying_tweets.csv")

    # Step 1: Explore data
    detector.explore_data()

    # Step 2: Prepare data
    detector.prepare_data()

    # Step 3: Create binary classification
    detector.create_binary_classifier()

    # Step 4: Train model
    detector.train_model()

    # Step 5: Evaluate model
    cm, y_pred, y_pred_proba = detector.evaluate_model()

    # Step 6: Analyze ambiguous cases
    detector.analyze_ambiguous_cases()

    # Step 7: Test on examples
    detector.test_on_examples()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Recommendations:")
    print("1. Use this system as a FLAGGING tool, not automated decision-making")
    print("2. Always involve human moderators for final decisions")
    print("3. Regularly audit for bias and update with diverse training data")
    print("4. Provide transparency to users about automated detection")
    print("5. Allow users to appeal automated flags")


if __name__ == "__main__":
    main()
