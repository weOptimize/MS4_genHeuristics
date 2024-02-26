#!/home/pinoystat/Documents/python/mymachine/bin/python

#* get execution time 
import time
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_ods_reader import read_ods
from operator import itemgetter
import matplotlib.pyplot as plt 
from scipy import stats as st
from deap import base, creator, tools, algorithms
import sys

import os
import openai
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview" 
openai.api_base = "https://gpt-4-uks.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value .
openai.api_key = "removed_for_security_purposes"

# Save the original stdout
original_stdout = sys.stdout 
# Open your file in write mode ('w') and store its reference in a variable


# Redirect standard output to a file
sys.stdout = open('output.txt', 'w', encoding='utf-8')

# Write an array with the project summaries you want to evaluate
# The summaries are inside folder "Proj_Proposals" and named Proj_01, Proj_02, etc.
# The array is called project_summaries
project_summaries = []
for i in range(1, 21):
    try:
        if i < 10:
            project_summary = open(f"Proj_Proposals/Proj_0{i}.txt", "r", encoding='utf-8')
        else:
            project_summary = open(f"Proj_Proposals/Proj_{i}.txt", "r", encoding='utf-8')
        project_summaries.append(project_summary.read())
        project_summary.close()
    except UnicodeDecodeError:
        if i < 10:
            project_summary = open(f"Proj_Proposals/Proj_0{i}.txt", "r", encoding='utf-8', errors='ignore')
        else:
            project_summary = open(f"Proj_Proposals/Proj_{i}.txt", "r", encoding='utf-8', errors='ignore')
        project_summaries.append(project_summary.read())
        project_summary.close()

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from functions_for_simheuristic_v12 import *

# create an empty list to store the timestamps and labels
timestamps = []

start_time = time.time()
timestamps.append(('t = 0', time.time()))

#get budgetting confidence policy
#budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
budgetting_confidence_policies = [0.75]
#array to store all budgeted durations linked to the budgetting confidence policy
budgeteddurations = []
stdevs = []
#array to store all found solutions
solutions = []
#arrays to store all results of the monte carlo simulation
mcs_results = []
mcs_results1 = []
mcs_results2 = []
#defining a global array that stores all portfolios generated (and another one for the ones that entail a solution)
tested_portfolios = []
solution_portfolios = []
npv_results = []
budgets = []

#defining the correlation matrix to be used in the monte carlo simulation (and as check when the correlations are expected to be 0)
correlation_matrix = []


#*****


#I define the number of candidates to be considered and the number of iterations for the MCS
nrcandidates = 20
# iterations = 300
# iterations_finalMCS = 5000
iterations = 100
iterations_finalMCS = 500


#I define the budget constraint (in k€) and the minimum confidence level for the portfolio
maxbdgt = 10800
# maxbdgt = 7560
min_pf_conf = 0.90

#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))

#I define a candidate array of size nr candidates with all ones
candidatearray = np.ones(nrcandidates)
#I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
initial_projection_indexes = np.arange(nrcandidates)

#first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
mcs_results1 = simulate(candidatearray,iterations)

#print("mcs results1: ", mcs_results1[0])

# mcs_results1[0] corresponds to the project costs and mcs_results1[1] to the project benefits (NPV)
x_perproj_matrix1 = pointestimate(mcs_results1[0], mcs_results1[1], budgetting_confidence_policies, nrcandidates)
print ("x_perproj_matrix1: ", x_perproj_matrix1)

# write the first timestamp and label to the list
timestamps.append(('First MCS with point estimate of budgets and NPV for each project', time.time()))

# extract first column of the matrix to get the budgeted costs of each project and store it in bdgtperproject_matrix
bdgtperproject_matrix = x_perproj_matrix1[0]
# extract second column of the matrix to get the NPV of each project and store it in npvperproject_matrix
npvperproject_matrix = x_perproj_matrix1[1]
# print("bdgtperproject_matrix at MAIN: ", bdgtperproject_matrix)
# print("npvperproject_matrix at MAIN: ", npvperproject_matrix)
# print("x_perproj_matrix1: ", x_perproj_matrix1)
# sum the costs of all projects to get the total cost of the portfolio if choosing all projects
totalcost = np.sum(x_perproj_matrix1[0])


# print("total portfolio cost allocation request (without correlations because it is a request):")
# print(totalcost)

df10r = correlatedMCS(mcs_results1, iterations, nrcandidates, initial_projection_indexes)
# print("df10r: ", df10r)

# write the second timestamp (substract the current time minus the previously stored timestamp) and label to the list
timestamps.append(('First MCS with correlated cost and NPV for each project', time.time()))

# Defining the fitness function
def evaluate(individual, bdgtperproject, npvperproject, maxbdgt):
    total_cost = 0
    total_npv = 0
    #multiply dataframe 10r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df10r = df10r * individual
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost10r = pf_df10r.sum(axis=1)
    #extract the maximum of the resulting costs
    maxcost10r = max(pf_cost10r)
    #print("max cost:")
    #print(maxcost10r)
    #count how many results were higher than maxbdgt
    count = 0
    for i in range(pf_cost10r.__len__()):
        if pf_cost10r[i] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    portfolio_confidence = 1-count/iterations
    #print("portfolio confidence:")
    #print(portfolio_confidence)
    for i in range(nrcandidates):
        #print(total_cost)
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]
    if total_cost > maxbdgt or portfolio_confidence < min_pf_conf:
        return 0, 0
    return total_npv, portfolio_confidence

# Define the genetic algorithm parameters
# POPULATION_SIZE = 180 #was 100 #was 50
POPULATION_SIZE = 100    #was 30
P_CROSSOVER = 0.4
P_MUTATION = 0.6
# MAX_GENERATIONS = 300 #was 500 #was 200 #was 100
MAX_GENERATIONS = 300 #was 100
HALL_OF_FAME_SIZE = 8

# Create the individual and population classes based on the list of attributes and the fitness function # was weights=(1.0,) returning only one var at fitness function
creator.create("FitnessMax", base.Fitness, weights=(100000.0, 1.0))
# create the Individual class based on list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()
# register a function to generate random integers (0 or 1) for each attribute/gene in an individual
toolbox.register("attr_bool", random.randint, 0, 1)
# register a function to generate individuals (which are lists of several -nrcandidates- 0s and 1s -genes-
# that represent the projects to be included in the portfolio)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, nrcandidates)
# register a function to generate a population (a list of individuals -candidate portfolios-)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the goal / fitness function
toolbox.register("evaluate", evaluate, bdgtperproject=bdgtperproject_matrix, npvperproject=npvperproject_matrix, maxbdgt=maxbdgt)
# register the crossover operator (cxTwoPoint) with a probability of 0.9 (defined above)
toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to flip each attribute/gene of 0.05.
# indpb is the independent probability for each gene to be flipped and P_MUTATION is the probability of mutating an individual
# The difference between P_MUTATION and indpb is that P_MUTATION determines whether an individual will be mutated or not,
# while indpb determines how much an individual will be mutated if it is selected for mutation.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the hall of fame
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define the statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)

# defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_npv():
    # Empty the hall of fame
    hall_of_fame.clear()
    # print("****************new policy iteration****************")
    # Initialize the population
    population = toolbox.population(n=POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        # Vary the population
        offspring = algorithms.varAnd(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION)
        # Evaluate the new individuals fitnesses
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hall_of_fame.update(offspring)
        # reorder the hall of fame so that the highest fitness individual is first
        # hall_of_fame.sort(key=itemgetter(0), reverse=True)
        population = toolbox.select(offspring, k=len(population))    
        record = stats.compile(population)
        print(f"Generation {generation}: Max NPV = {record['max']}")

    #de momento me dejo de complicarme con el hall of fame y me quedo con el último individuo de la última generación
    # return the optimal portfolio from the hall of fame, their fitness and the total budget
    # print(hall_of_fame)
    #return hall_of_fame
    print("Hall of Fame:")
    for i in range(HALL_OF_FAME_SIZE):
        print(hall_of_fame[i], hall_of_fame[i].fitness.values[0], hall_of_fame[i].fitness.values[1], portfolio_totalbudget(hall_of_fame[i], bdgtperproject_matrix))
    #print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], hall_of_fame[0].fitness.values[1], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix))
    #print(hall_of_fame[1], hall_of_fame[1].fitness.values[0], hall_of_fame[1].fitness.values[1], portfolio_totalbudget(hall_of_fame[1], bdgtperproject_matrix))
    #print(hall_of_fame[2], hall_of_fame[2].fitness.values[0], hall_of_fame[2].fitness.values[1], portfolio_totalbudget(hall_of_fame[2], bdgtperproject_matrix))
    #return hall_of_fame[0], hall_of_fame[0].fitness.values[0][0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix)
    return hall_of_fame

# this function calculates the npv of each project and then uses the maximizer function to obtain and return portfolio, npv and bdgt in a matrix (solutions)
for i in range(len(budgetting_confidence_policies)):
    # I take the column of bdgtperproject_matrix that corresponds to the budgetting confidence policy
    bdgtperproject=bdgtperproject_matrix[:,i]
    # print(bdgtperproject)
    npvperproject=npvperproject_matrix[:,i]
    # print(npvperproject)
    # execute the maximizer function to obtain the portfolio, and its npv and bdgt
    projectselection = maximize_npv()
    # assign the result from projectselection to the variable solutions
    solutions.append(projectselection)
    #print(solutions)
# lately I only had one BCP, si it has performed the append only once, however as the solution is a hall of fame, it has appended a list of 3 individuals

#store the npv results, portfolio results, portfolio confidence levels and budgets taken in different lists
npv_results = [0] * len(projectselection)
portfolio_results = [0] * len(projectselection)
portfolio_confidence_levels = [0] * len(projectselection)
pf_conf2 = [0] * len(projectselection)
budgets = [0] * len(projectselection)
for i in range(nrcandidates):
    npv_results = [[x[i].fitness.values[0][0] for x in solutions] for i in range(len(projectselection))]
    #portfolio_results = [[x[i] for x in solutions] for i in range(len(projectselection))]
    portfolio_confidence_levels = [[x[i].fitness.values[1] for x in solutions] for i in range(len(projectselection))]
    budgets = [[portfolio_totalbudget(x[i], bdgtperproject_matrix)[0] for x in solutions] for i in range(len(projectselection))]

# take all arrays inside portfolio_results and sum all of them
portfolio_projection = [sum(x) for x in zip(*projectselection)]

# convert portfolio_projection array into a binary array, where 1 means that the project is selected and 0 means that it is not
portfolio_projection = [1 if x > 0 else 0 for x in portfolio_projection]

# calculate the amount of projects in "portfolio_projection"
projected_candidates = sum(portfolio_projection)

# store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
zipped_projection_indexes = [i for i, x in enumerate(portfolio_projection) if x == 1]

# convert portfolio_projection in a full ones array
# portfolio_projection = [1] * len(portfolio_projection)

# write the third timestamp (substract the current time minus the previously stored timestamp) and label to the list
timestamps.append(('Optimization step (GA algorithm)', time.time()))


print ("************ SUMMARY STAGE 1 **********")
print ("npv_results: ", npv_results)
print ("portfolio_results: ", projectselection)
print ("portfolio_confidence_levels: ", portfolio_confidence_levels)
print ("budgets: ", budgets)
print ("portfolio_projection: ", portfolio_projection)
print ("Indexes of selected projects: ", zipped_projection_indexes)
print ("Number of candidate projects for stage 2: ", projected_candidates)

# Load text from strategic plan as plain code so that I can concatenate it at the prompt following the conversation
# with the user
strategic_plan = open("EU_LIFE_Valid_Statement.txt", "r")

consolidated_summaries = []

# for each portfolio in the dataframe, extract the identificator of the projects that have been selected under "projectselection" and stere in an array called chosenprojects (remind that 0 means project 1, etc.)
# sum +1 to each of the values in zipped_projection_indexes to get the real number of the project
chosenprojects = [x+1 for x in zipped_projection_indexes]
print("Chosen projects: ", chosenprojects)
# Pick from project_summaries only the ones that belong to the chosen portfolios, before each append add the number of the project
Summary_Promising_Projects = ""
print("length of project summaries", len(project_summaries))
# print("project summaries", project_summaries)
for chosenproject in chosenprojects:
    # print("Chosenproject: ", chosenproject)
    Summary_Promising_Projects += f"Project {chosenproject}: {project_summaries[chosenproject-1]}\n"
    # print("Consolidated summary: ", consolidated_summary)
print("Summary_Promising_Projects: ", Summary_Promising_Projects)

# concatenate an initial instruction with the strategic plan in the same string
initialization_prompt = "You are a reliable and ojective portfolio manager, who only writes statements that \
can be reasonably justified based only on the information you have available. You are provided with a proposal of a portfolio \
consisting of several project for which you will be provided with an executive summary. You will compare the portfolio choice respect to \
a strategic plan I am providing at this message. Your evaluation should be on a scale from 1 to 100, with 100 being the maximum alignment.\
provide first the rating (only the value without saying '/100') and then the justification for the rating, detailing how well the combination of projects aligns with the\
strategic plan and the key factors that influenced your assessment. The strategic plan is the following:" + strategic_plan.read() + \
"And the promising projects that you must evaluate are the following (take into account that each project summary is preceded by its project number \
so that you remember which projects are proposed at each potential portfolio):\n" + Summary_Promising_Projects

print ("************ STARTING STAGE 2 (long MCS) **********")
#second simulation to get all cdfs for cost & benefits after optimization step (may_update: was 1000)
mcs_results2 = simulate(portfolio_projection,iterations_finalMCS)


# mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (NPV)
x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies, projected_candidates)
# print ("x_perproj_matrix2: ", x_perproj_matrix2)

# write the fourth timestamp and label to the list
timestamps.append(('Second MCS, also including point estimate of budgets and NPV for shortlisted projects', time.time()))

# we assume correlations at the cost side, not at the benefits side (conservative approach)
# update x_perproj_matrix2 with the correlation effect registered inside df20r
# print("x_perproj_matrix2: ", x_perproj_matrix2)
# separate the budget and npv results from the x_perproj_matrix
bdgtperproject_matrix = x_perproj_matrix2[0]
npvperproject_matrix = x_perproj_matrix2[1]
# print(type(bdgtperproject_matrix))
# print(type(npvperproject_matrix))
bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
npvperproject_matrix = np.squeeze(npvperproject_matrix)

# remove all data that has zeroes from bdgtperproject_matrix and npvperproject_matrix
# bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
# npvperproject_matrix = npvperproject_matrix[np.nonzero(npvperproject_matrix.flatten())]

# print("bdgtperproject_matrix: ", bdgtperproject_matrix)
# print("npvperproject_matrix: ", npvperproject_matrix)
print("size of bdgtperproject_matrix", len(bdgtperproject_matrix))
print("size of npvperproject_matrix", len(npvperproject_matrix))
print("size of mcs_results2", len(mcs_results2))

# print("mcs_results2 (input para correlacionar): ", mcs_results2)

# for each of the options obtained in projectselection, calculate the total portfolio npv and the portfolio budget based on the information from x_perproj_matrix
npv_results = [0] * len(projectselection) # as many as len(projectselection) because we have one npv per item in HoF
budgets = [0] * len(projectselection)
pf_conf2 = [0] * len(projectselection)
widened_bdgtperproject_matrix = [0] * nrcandidates # as many as initial amount of project candidates
widened_npvperproject_matrix = [0] * nrcandidates
# initialize dataframe called widened_df20r as a copy of df10r
widened_df20r = df10r.copy()
# enlarge the dataframe to the size of iterations_finalMCS
widened_df20r = widened_df20r.reindex(range(iterations_finalMCS))
# fill the dataframe with zeroes
widened_df20r.iloc[:, :] = 0

df20r = correlatedMCS(mcs_results2, iterations_finalMCS, projected_candidates, zipped_projection_indexes)
# print("df20r: ", df20r)

# pick in order the values from bdgtperproject_matrix and npvperproject_matrix and store them in widened_bdgtperproject_matrix and widened_npvperproject_matrix
# The location of the values to be picked is available in zipped_projection_indexes
j=0
for i in range(nrcandidates):
    if i in zipped_projection_indexes:
        widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix [j],3)
        widened_npvperproject_matrix [i] = round(npvperproject_matrix [j],3)
        j+=1
    else:
        pass
# print("widened_bdgtperproject_matrix: ", widened_bdgtperproject_matrix)
# print("widened_npvperproject_matrix: ", widened_npvperproject_matrix)

# pick in order the values from df20r and store them in widened_df20r (to be used in the next step)
i=0
j=0
k=0
for i in range(nrcandidates):
    if i in zipped_projection_indexes:
        for j in range(iterations_finalMCS):
            widened_df20r.loc[j, widened_df20r.columns[i]] = df20r.loc[j, df20r.columns[k]]
        k += 1
    else:
        pass

print("widened_df20r: ", widened_df20r)

for i in range(len(projectselection)):
    #calculate the total portfolio budget by multiplying the budget of each project by the binary array obtained in projectselection    
    print(projectselection[i])
    budgets[i] = np.sum(np.multiply(widened_bdgtperproject_matrix,projectselection[i]))
    #calculate the total portfolio npv by multiplying the npv of each project by the binary array obtained in projectselection
    npv_results[i] = np.sum(np.multiply(widened_npvperproject_matrix,projectselection[i]))
    #multiply dataframe 20r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df20r = widened_df20r * projectselection[i]
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost20r = pf_df20r.sum(axis=1)
    #extract the maximum of the resulting costs
    maxcost20r = max(pf_cost20r)
    print("max cost:")
    print(maxcost20r)
    #count how many results were higher than maxbdgt
    count = 0
    for j in range(pf_cost20r.__len__()):
        if pf_cost20r[j] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    pf_conf2[i] = 1-count/iterations_finalMCS

# create a dataframe with the results
finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio NPV': npv_results}) #, 'Portfolio Budget': budgets, 'Portfolio Confidence': pf_conf2})
# order the dataframe by the portfolio npv, starting with the highest npv
finalsol_df = finalsol_df.sort_values(by=['Portfolio NPV'], ascending=False)

# Initialize array to store the consolidation of "project selections" belonging to each portfolio that is in the HoF
chosenprojects = []

# for each portfolio in the dataframe, extract the identificator of the projects that have been selected under "projectselection" and store in an array called chosenprojects (remind that 0 means project 1, etc.)
# sum +1 to each of the values in projectselection to get the real number of the project
for i in range(len(finalsol_df)):
    # start with a text saying that <<the projects belonging to this portfolio are: ">>
    chosenprojects.append("The projects belonging to this portfolio are: ")
    # for each project in the portfolio, add the project number to the text, separated by commas
    for j in range(nrcandidates):
        if projectselection[i][j] == 1:
            chosenprojects[i] += str(j+1) + ", "
    # remove the last comma from the text
    chosenprojects[i] = chosenprojects[i][:-2]
    # print the text
    print("Chosen projects: ", chosenprojects)


# *********************  GPT-4  ************************

# The conversation is initialized with a message to the user, which is the first message in the conversation list
# First there is an instruction under "content", and then - also inside "content" - the strategic plan is concatenated
conversation=[{"role": "system", "content": initialization_prompt}]

# Loop through the consolidated_summaries and ask the AI(GPT4) to evaluate them
# Store the AI's response in a new array called "evaluations", where the topic is in the first column
# and the AI's response is in another column
evaluations = []
for choice in chosenprojects:
    conversation.append({"role": "user", "content": choice})
    try:
        response = openai.ChatCompletion.create(
            engine="GPT4_turbo_128k", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages = conversation,
            temperature=0.1,
            max_tokens=1250,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    except Exception as e:
        print("An error occurred: ", e)

    # response = openai.Completion.create(
    #     engine="weO_vs00_gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    #     messages = conversation
    # )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    # print("\n" + response['choices'][0]['message']['content'] + "\n")
    evaluations.append([choice, response['choices'][0]['message']['content']])

# Modify the evaluations array so that the last column is splitted into two columns. The first column should contain the number
# and the second column should contain the justification
for evaluation in evaluations:
    # Split the last column by the first "." symbol
    split = evaluation[1].split("\n", 1)
    # Remove the first period from the second column
    split[1] = split[1][1:]
    # Insert the split columns into the array
    evaluation[1] = split[0]
    evaluation.insert(2, split[1])

# Add the evaluations (only columns 2 and 3, forget 1) to the finalsol_df dataframe
for i in range(len(evaluations)):
    finalsol_df.loc[i, 'Strategic Score'] = evaluations[i][1]
    finalsol_df.loc[i, 'Justification'] = evaluations[i][2]

print ("Final Solution: ", finalsol_df)

# write the fifth timestamp and label to the list. Calculation FINALIZED
timestamps.append(('Application of correlation effect to final options', time.time()))

segments = [0] * (len(timestamps)-1)

# calculate the difference between each pair of timestamps
for i in range(0, len(timestamps)-1):
    segments[i] = (timestamps[i+1][0], round(timestamps[i+1][1] - timestamps[i][1], 2))
    print(segments)
    
# create a dataframe from the list of timestamps
crono_frame = pd.DataFrame(segments, columns=['Checkpoint', 'Execution time (s)'])

# add a final register with the total execution time
crono_frame.loc['Total'] = ['Total', crono_frame['Execution time (s)'].sum()]

# print the dataframe
print(crono_frame)

npv_results = []
strategic_score = []
budgets = []
pf_cost20r = []


# # reorder the dataframe by the Strategic Score, starting with the highest Strategic Score
# finalsol_df = finalsol_df.sort_values(by=['Strategic Score'], ascending=False)

# # extract the npv results and strategic scores from the finalsol_df dataframe
# npv_results = finalsol_df['Portfolio NPV']
# strategic_score = finalsol_df['Strategic Score']
# # print the results
# print("npv_results: ", npv_results)
# print("strategic_score: ", strategic_score)

# ************************* Extracting the Pareto Set for Strategic Analysis *************************

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
pareto_set_df = finalsol_df.iloc[pareto_points]

# Sort by 'Strategic Score' in descending order
pareto_set_df = pareto_set_df.sort_values('Strategic Score', ascending=False)

# Print Initial Dataframe
print("Top 8 Candidates from Simheuristic: ")
print(finalsol_df)

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


# Redirect standard output to a file
sys.stdout = open('GPT4_128k_Evaluation.txt', 'w', encoding='utf-8')

# Print the top three strategic scores from the pareto set and their justifications, and the portfolio to which it corresponds
print("Top strategic scores and their justifications, and the portfolio to which it corresponds")
for i in range(len(pareto_set_df)):
    print(f"Portfolio {pareto_set_df.iloc[i, 0]}")
    print(f"Strategic Score: {pareto_set_df.iloc[i, 2]}")
    print(f"Portfolio NPV: {pareto_set_df.iloc[i, 1]}")
    print(f"Justification: {pareto_set_df.iloc[i, 3]}")
    print("\n")

sys.stdout = original_stdout
print("opening file GPT4_128k_Evaluation")
top_pf_assessment = open("GPT4_128k_Evaluation.txt", "r")

exec_summary_prompt = "I have stored your assessment and I provide it to you it now again, so that you reword it in a summarized way. It is sequential, in decreasing order\
of strategic score, and includes the NPV (in k€). Start by highlighting (maximum 8 short sentences) the advantages of the portfolio with highest strategic score, and then \
continue with the other portfolios in order, highlisting what advantages are lost if choosing the next options, all in comparison respect to the one with highest strategic score. \
The assessment for your reference is the following:"+ top_pf_assessment.read()
sys.stdout = original_stdout

#*** Total execution time
print("Total execution time: %s seconds" %((time.time() - start_time)))

sys.stdout = open('GPT4_128k_Top_4_exec_summary.txt', 'w', encoding='utf-8')

# user_input = input()
conversation.append({"role": "user", "content": exec_summary_prompt})
try:
    response = openai.ChatCompletion.create(
        engine="GPT4_turbo_128k", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages = conversation,
        temperature=0.1,
        max_tokens=1250,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
except Exception as e:
    print("An error occurred: ", e)

# response = openai.Completion.create(
#     engine="weO_vs00_gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
#     messages = conversation
# )

conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
print("\n" + response['choices'][0]['message']['content'] + "\n")

# convert npv_results and strategic_score to 1d arrays
npv_results = np.array(npv_results)
strategic_score = np.array(strategic_score)
print("npv_results 1D: ", npv_results)
print("strategic_score 1D: ", strategic_score)

# strategic_score has numbers as text. Convert them to numbers
strategic_score = strategic_score.astype(int)



# # scatterplot of the npv results and strategic scores
# plt.figure(1)
# plt.scatter(npv_results, strategic_score)
# plt.xlabel('NPV (in k€)')
# plt.ylabel('Strategic Score')
# plt.title('NPV vs Strategic Score')
# # Adjust the x and y axis limits
# # plt.xlim(19600, 19850)
# plt.xlim(13600, 15000)
# #plt.ylim(84, 94)
# plt.ylim(82, 92)# Add grid
# plt.grid(True)
# # Add labels to the points, taking the dataframe id as label, writing the text "Portfolio " before the id
# for i in range(len(npv_results)):
#     plt.text(npv_results[i], strategic_score[i], "Portfolio " + str(finalsol_df.index[i]))
# # Show the plot
# plt.show()


# Scatterplot of NPV results and strategic scores
plt.figure(1)
plt.scatter(npv_results, strategic_score, zorder=1)
plt.xlabel('NPV (in k€)', fontsize=16)  # Increase font size
plt.ylabel('Strategic Score', fontsize=16)  # Increase font size
# plt.title('NPV vs Strategic Score', fontsize=16)  # Increase font size

# Adjust the x and y axis limits
plt.xlim(19650, 19850)
plt.ylim(82, 92)
plt.grid(True, color='grey', linestyle='--', linewidth=0.5, zorder=2)

# Set background color to white
plt.gca().set_facecolor('white')

# Set scatter plot points color to grey
# plt.scatter(npv_results, strategic_score, color='grey')

# Add labels to the points
for i in range(len(npv_results)):
    plt.text(npv_results[i], strategic_score[i], "Portfolio " + str(finalsol_df.index[i]), fontsize=16)

# Set plot outline color to black
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['top'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.savefig('Figure_3.png')

# Show the plot
plt.show()
