import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Create a large figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# 1. MODEL PERFORMANCE METRICS
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [81.01, 96.46, 80.24, 87.61]
colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]

bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
ax1.set_ylim([0, 100])
ax1.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")
ax1.set_title("Model Performance Metrics", fontsize=15, fontweight="bold", pad=15)
ax1.axhline(y=80, color="red", linestyle="--", alpha=0.3, linewidth=2)
ax1.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

# ============================================================================
# 2. CONFUSION MATRIX
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
cm = np.array([[1300, 231], [1548, 6287]])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="RdYlGn_r",
    cbar=False,
    square=True,
    linewidths=2,
    linecolor="black",
    annot_kws={"size": 16, "weight": "bold"},
    ax=ax2,
)

ax2.set_xlabel("Predicted", fontsize=13, fontweight="bold")
ax2.set_ylabel("Actual", fontsize=13, fontweight="bold")
ax2.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
ax2.set_xticklabels(["Not CB", "CB"], fontsize=11)
ax2.set_yticklabels(["Not CB", "CB"], fontsize=11, rotation=0)

# Add text annotations for errors
ax2.text(
    1,
    0.2,
    "False Positives\n(231)",
    ha="center",
    fontsize=9,
    color="red",
    weight="bold",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
ax2.text(
    0,
    1.2,
    "False Negatives\n(1,548)",
    ha="center",
    fontsize=9,
    color="red",
    weight="bold",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# ============================================================================
# 3. CLASS DISTRIBUTION
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
categories = ["Religion", "Age", "Gender", "Ethnicity", "Not CB", "Other CB"]
counts = [7995, 7988, 7875, 7955, 7657, 7358]
colors_cat = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#6ab04c", "#c44569"]

wedges, texts, autotexts = ax3.pie(
    counts,
    labels=categories,
    autopct="%1.1f%%",
    colors=colors_cat,
    startangle=90,
    textprops={"fontsize": 11, "weight": "bold"},
    explode=[0.02] * 6,
)

ax3.set_title(
    "Dataset Class Distribution\n(46,828 tweets after preprocessing)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)

# ============================================================================
# 4. TOP CYBERBULLYING INDICATORS
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
words = [
    "rape",
    "dumb",
    "bullies",
    "nigger",
    "feminazi",
    "idiot",
    "school bully",
    "muslims",
    "idiots",
    "gay",
]
weights = [8.384, 8.264, 8.099, 8.097, 7.867, 6.692, 6.477, 6.403, 6.366, 6.058]

y_pos = np.arange(len(words))
bars = ax4.barh(
    y_pos, weights, color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=1.5
)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(words, fontsize=11)
ax4.set_xlabel("Feature Weight", fontsize=13, fontweight="bold")
ax4.set_title("Top 10 Cyberbullying Indicators", fontsize=15, fontweight="bold", pad=15)
ax4.invert_yaxis()
ax4.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, weight) in enumerate(zip(bars, weights)):
    ax4.text(
        weight + 0.15, i, f"{weight:.2f}", va="center", fontweight="bold", fontsize=10
    )

# ============================================================================
# 5. TOP NON-CYBERBULLYING INDICATORS
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
words_safe = [
    "mkr",
    "daesh",
    "class",
    "mosul",
    "bullying",
    "college",
    "yesallwomen",
    "kat and",
    "andre",
    "user also",
]
weights_safe = [5.392, 3.433, 3.420, 3.397, 3.228, 2.696, 2.446, 2.379, 2.354, 2.168]

y_pos = np.arange(len(words_safe))
bars = ax5.barh(
    y_pos, weights_safe, color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=1.5
)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(words_safe, fontsize=11)
ax5.set_xlabel("Feature Weight (Absolute Value)", fontsize=13, fontweight="bold")
ax5.set_title(
    "Top 10 Non-Cyberbullying Indicators", fontsize=15, fontweight="bold", pad=15
)
ax5.invert_yaxis()
ax5.grid(axis="x", alpha=0.3)

# Add value labels
for i, (bar, weight) in enumerate(zip(bars, weights_safe)):
    ax5.text(
        weight + 0.08, i, f"{weight:.2f}", va="center", fontweight="bold", fontsize=10
    )

# ============================================================================
# 6. ERROR ANALYSIS
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
error_types = [
    "False\nPositives",
    "False\nNegatives",
    "True\nPositives",
    "True\nNegatives",
]
error_counts = [231, 1548, 6287, 1300]
colors_errors = ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1"]

bars = ax6.bar(
    range(len(error_types)),
    error_counts,
    color=colors_errors,
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)
ax6.set_xticks(range(len(error_types)))
ax6.set_xticklabels(error_types, fontsize=11, fontweight="bold")
ax6.set_ylabel("Count", fontsize=13, fontweight="bold")
ax6.set_title(
    "Classification Results Breakdown", fontsize=15, fontweight="bold", pad=15
)
ax6.set_yscale("log")
ax6.grid(axis="y", alpha=0.3)

# Add value labels
for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    ax6.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.15,
        f"{count:,}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

# ============================================================================
# FINAL FORMATTING
# ============================================================================
plt.suptitle(
    "Cyberbullying Detection System - Comprehensive Analysis\nSafeNet Analytics Project",
    fontsize=20,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Display the plot
plt.show()

print("\n✅ Visualizations generated successfully!")
print("📊 6 charts created showing:")
print("   1. Performance metrics")
print("   2. Confusion matrix")
print("   3. Class distribution")
print("   4. Top cyberbullying indicators")
print("   5. Top non-cyberbullying indicators")
print("   6. Error analysis breakdown")
