import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Portfolio': [
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    ],
    'Portfolio NPV': [
        19779.06,
        19735.48,
        19734.01,
        19690.43,
        19676.38,
        19673.29,
        19629.71,
        19626.93
    ],
    'Strategic Score': [
        85,
        90,
        88,
        87,
        92,
        89,
        90,
        85
    ],
    'Justification': [
        'The proposed portfolio of projects demonstrate...',
        'The revised portfolio of projects demonstrates...',
        'The portfolio presents a robust alignment with...',
        'The portfolio presents a strong alignment with...',
        'The portfolio of projects exhibits a high degr...',
        'The portfolio of projects demonstrates a very ...',
        'The selected portfolio of projects aligns well...',
        'The portfolio of projects presents a strong al...'
    ]
}

# Extract the pareto set from all portfolio alternatives taking into account two variables: Portfolio NPV and Strategic Score
# Pareto set is a set of portfolios that are not dominated by any other portfolio
# A portfolio is dominated if there is another portfolio that has a better NPV and a better Strategic Score
# The resulting dataframe should contain the portfolios that are part of the pareto set, including their Portfolio, Portfolio NPV, Strategic Score and Justification
# The dataframe should be sorted by Portfolio NPV in descending order
# The resulting dataframe should have the following columns: 'Portfolio', 'Portfolio NPV', 'Strategic Score' and 'Justification'

# Generate a new dataframe that keeps only the portfolios that can be considered as part of the pareto set
# The dataframe should be sorted by Strategic Score in descending order
# The resulting dataframe should have the following columns: 'Portfolio', 'Portfolio NPV', 'Strategic Score' and 'Justification'

# Generate a new dataframe that keeps only the portfolios that can be considered as part of the pareto set

finalsol_df = pd.DataFrame(data)

# Function to identify pareto-efficient points
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Compare all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

# Use function to find pareto efficient points
pareto_points = identify_pareto(finalsol_df[['Portfolio NPV', 'Strategic Score']].values)
pareto_set = finalsol_df.iloc[pareto_points]

# Sort by 'Portfolio NPV' in descending order
pareto_set = pareto_set.sort_values('Portfolio NPV', ascending=False)

# Print Initial Dataframe
print("Top 8 Candidates from Simheuristic: ")
print(finalsol_df)

# For the second dataframe sorted by 'Strategic Score'
pareto_set_score = pareto_set.sort_values('Strategic Score', ascending=False)
print(pareto_set_score)

# extract the npv results and strategic scores from the finalsol_df dataframe
npv_results = finalsol_df['Portfolio NPV']
strategic_score = finalsol_df['Strategic Score']
# convert npv_results and strategic_score to 1d arrays
npv_results = np.array(npv_results)
strategic_score = np.array(strategic_score)
print("npv_results 1D: ", npv_results)
print("strategic_score 1D: ", strategic_score)

# strategic_score has numbers as text. Convert them to numbers
strategic_score = strategic_score.astype(int)

# # convert npv_results and strategic_score to 1d arrays
# npv_results = np.array(npv_results)
# strategic_score = np.array(strategic_score)
# print("npv_results 1D: ", npv_results)
# print("strategic_score 1D: ", strategic_score)

print("Top strategic scores and their justifications, and the portfolio to which it corresponds")
for i in range(len(pareto_set_score)):
    print(f"Portfolio {pareto_set_score.iloc[i, 0]}")
    print(f"Strategic Score: {pareto_set_score.iloc[i, 2]}")
    print(f"Portfolio NPV: {pareto_set_score.iloc[i, 1]}")
    print(f"Justification: {pareto_set_score.iloc[i, 3]}")
    print("\n")

# scatterplot of the npv results and strategic scores
plt.figure(1)
plt.scatter(npv_results, strategic_score)
plt.xlabel('NPV')
plt.ylabel('Strategic Score')
plt.title('NPV vs Strategic Score')
# Adjust the x and y axis limits
plt.xlim(19600, 19850)
plt.ylim(84, 94)
# Add grid
plt.grid(True)
# Add labels to the points, taking the dataframe id as label, writing the text "Portfolio " before the id
for i in range(len(npv_results)):
    plt.text(npv_results[i], strategic_score[i], "Portfolio " + str(finalsol_df.index[i]))
# Show the plot
plt.show()