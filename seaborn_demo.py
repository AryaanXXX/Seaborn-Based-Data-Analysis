import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Initialize PDF
with PdfPages("seaborn_visualization.pdf") as pdf:
    # Set a Seaborn style for better aesthetics from the start
    sns.set_theme(style="darkgrid")

    # --- 1. Loading Data ---

    tips = sns.load_dataset("tips")
    print("--- 1. Data Preview (from pandas DataFrame) ---")
    print(tips.head())
    print("\n")
    print(tips.info())
    print("\n")




    # --- 2. Univariate Distributions ---

    fig1 = plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(data=tips, x="total_bill", kde=True, bins=20, color="skyblue")
    plt.title("Distribution of Total Bill (Histogram + KDE)")
    plt.xlabel("Total_bill($)")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    sns.kdeplot(data=tips, x="tip", fill=True, color="lightcoral")
    plt.title("Distribution of Tip Amount (KDE)")
    plt.xlabel("Tip Amount ($)")
    plt.ylabel("Density")

    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    print("\n--- 2. Univariate Distributions Explained ---")
    print("`histplot` shows the frequency of values in bins, `kde=True` overlays a smooth density curve.")
    print("`kdeplot` directly estimates the probability density function (PDF).")
    print("These help us see the shape, central tendency, and spread of a single variable.")
    print("Notice how `sns.set_theme()` already makes these plots look clean without extra Matplotlib styling.")
    print("\n")




    # --- 3. Bivariate Relationships ---

    fig2 = plt.figure(figsize=(8,6))
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex", size="size", sizes=(20,200), alpha=0.7)
    plt.title("Total Bill vs Tip (Colored by Sex, Sized by Party Size)")
    plt.xlabel("Total Bill ($)")
    plt.ylabel("Tip ($)")
    pdf.savefig(fig2)
    plt.close(fig2)

    print("\n--- 3. Bivariate Relationships Explained ---")
    print("`scatterplot` is fundamental for two continuous variables. Here, `hue` maps 'sex' to color,")
    print("and `size` maps 'size' (party size) to the marker size, adding more dimensions to the plot easily.")
    print("This allows us to see if different sexes tip differently or if larger parties tip more.")
    print("\n")

    # Relplot (Faceted)

    g = sns.relplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker",
                   size="size", style="sex", kind="scatter", height=4, aspect=1.2)
    g.set_axis_labels("Total Bill ($)", "Tip ($)")
    g.set_titles("Time: {col_name}")
    g.fig.suptitle("Total Bill vs Tip by Time of Day and Smoker Status", fontsize=14, fontweight='bold')
    g.fig.subplots_adjust(top=0.85)
    pdf.savefig(g.fig)
    plt.close(g.fig)

    print("`relplot` (Relational Plot) is a 'figure-level' function that creates a `FacetGrid` internally.")
    print("`col='time'` creates separate columns for 'Dinner' and 'Lunch'.")
    print("`hue='smoker'` colors points based on smoker status.")
    print("`style='sex'` changes marker style based on sex. This showcases Seaborn's power for multi-variate analysis.")
    print("\n")




    # --- 4. Categorical Data Plots ---

    fig3 = plt.figure(figsize=(15,6))
    plt.subplot(1,3,1)
    sns.boxplot(data=tips, x="day", y="total_bill", hue="sex", palette="viridis")
    plt.title("Total Bill by Day and Sex (Box Plot)")
    plt.xlabel("Day of week")
    plt.ylabel("Total Bill ($)")

    plt.subplot(1,3,2)
    sns.violinplot(data=tips, x="day", y="tip", hue="smoker", split=True, palette="rocket")
    plt.title("Tip amount by day and smoker (Violin Plot)")
    plt.xlabel("Day of week")
    plt.ylabel("Tip ($)")

    plt.subplot(1,3,3)
    sns.countplot(data=tips, x="day", hue="time", palette="pastel")
    plt.title("Number of Observations per Day by Time")
    plt.xlabel("Day of week")
    plt.ylabel("Count")

    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

    print("\n--- 4. Categorical Data Plots Explained ---")
    print("`boxplot` shows central tendency (median), spread (IQR), and outliers for numerical data across categories.")
    print("`violinplot` combines box plot with KDE, showing the full distribution shape.")
    print("`countplot` shows the number of observations in each category.")
    print("The `hue` parameter is used consistently across these plots to add another categorical dimension for comparison.")
    print("`palette` allows easy selection of color schemes.")
    print("\n")




    # --- 5. Matrix Plots (Correlation Heatmap) ---

    numerical_tips = tips.select_dtypes(include=np.number)
    fig4 = plt.figure(figsize=(8,7))
    sns.heatmap(numerical_tips.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Numerical Features in Tips Dataset")
    pdf.savefig(fig4)
    plt.close(fig4)

    print("\n--- 5. Matrix Plots (Correlation Heatmap) Explained ---")
    print("`heatmap` is excellent for visualizing matrices, such as correlation matrices.")
    print("`annot=True` displays the correlation values on the map.")
    print("`cmap` sets the color map, and `fmt` formats the annotation numbers.")
    print("This quickly reveals strong positive (red) or negative (blue) correlations between variables.")
    print("\n")




    # --- 6. Regression Plots ---

    g2 = sns.lmplot(data=tips, x="total_bill", y="tip", col="sex", row="smoker",
                    height=4, aspect=1.2, scatter_kws={'alpha':0.6})
    g2.set_axis_labels("Total Bill ($)", "Tip ($)")
    g2.set_titles(col_template="Sex: {col_name}", row_template="Smoker: {row_name}")
    g2.tight_layout()
    g2.fig.subplots_adjust(top=0.85)
    g2.fig.suptitle("Regression of Tip vs Total Bill by Sex and Smoker Status", y=0.98)
    pdf.savefig(g2.fig)
    plt.close(g2.fig)

    print("\n--- 6. Regression Plots Explained ---")
    print("`lmplot` (Linear Model Plot) draws a scatter plot with a regression line and a confidence interval.")
    print("Here, we use `col` and `row` to create a 2x2 grid, showing the regression for each combination of sex and smoker status.")
    print("This helps identify if the linear relationship between bill and tip varies by these categories.")
    print("\n")




    # --- 7. Combining Seaborn with Matplotlib ---

    fig5 = plt.figure(figsize=(10,6))
    plot = sns.barplot(data=tips, x="day", y="total_bill", estimator=np.sum, errorbar=None, hue="day", palette="crest") #plot
    plt.title("Total Bill Sum by Day (Customized with Matplotlib)", fontsize=16, color='darkblue')
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Total Bill Sum ($)", fontsize=12)
    plt.ylim(0,1500)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for container in plot.containers: # plot.containers is a list of bar containers created by sns.barplot. Each container holds a group of bars.
        plt.bar_label(container, fmt='%.2f') # adds labels on top of each bar to show the exact numeric value.

    pdf.savefig(fig5)
    plt.close(fig5)

    print("\n--- 7. Combining Seaborn with Matplotlib Explained ---")
    print("This demonstrates how Matplotlib functions (`plt.title`, `plt.xlabel`, `plt.ylim`, `plt.grid`, `plt.bar_label`)")
    print("can be used directly after a Seaborn call to add further customization.")
    print("\n")

print("All visualizations saved to 'seaborn_visualization.pdf'")

