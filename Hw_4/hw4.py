# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday September 23 13:36:53 2021                               #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Wealth calculator using PySimpleGUI to project wealth over 70 years          #
################################################################################


import PySimpleGUI as sg                            # User for drawing the UI
import numpy as np                                  # Used for numerical calcualtions, sum, formatting
import matplotlib.pyplot as plt                     # get matplotlib for plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     # getFigureCanvas for plotting


# Maximum years defined - more so as maximum life expectancy
MAX_YEARS = 70

# Number of times each year has to be calculated
CALCULATION_TIMES = 10

# String values for labels in our UX
MEAN_RETURN = 'Mean Return (%)'
STD_DEV_RETURN = 'Std Dev Return (%)'
YEARLY_CONTRIBUTION = 'Yearly Contribution ($)'
CONTRIBUTION_YEARS = 'No. of Years of Contribution'
YEARS_TO_RETIREMENT = 'No. of Years to Retirement'
POST_RETIREMENT_EXPENDITURE = 'Annual Spend in Retirement'

# String values for our buttons in the UX
CALCULATE = 'Calculate'
QUIT = 'Quit'

# Defining the constant layout here.
# This includes 5 input parameters for calculating wealth
# Then an empty lable to display the calculated wealth
# Two buttons - Calculate and Quit - for performing those respective actions
# Finally a canvas where we'll plot the wealth projections.
LAYOUT = [[sg.Text(MEAN_RETURN), sg.InputText()],
          [sg.Text(STD_DEV_RETURN), sg.InputText()],
          [sg.Text(YEARLY_CONTRIBUTION), sg.InputText()],
          [sg.Text(CONTRIBUTION_YEARS), sg.InputText()],
          [sg.Text(YEARS_TO_RETIREMENT), sg.InputText()],
          [sg.Text(POST_RETIREMENT_EXPENDITURE), sg.InputText()],
          [sg.Text('', key='-wealth_label-')],
          [sg.Button(CALCULATE), sg.Button(QUIT)],
          [sg.Canvas(key='canvas')]]

# Header for the window
WINDOW_HEADER = 'Wealth Calculator'

# Actual window that shows up to take input, uses the layout and header defined above
# We start from 20,20 somewhere top left - to be able to have enough space to plot the graph in UX
WINDOW = sg.Window(WINDOW_HEADER,
                    LAYOUT,
                    finalize=True,
                    resizable=True,
                    location=(20, 20),
                    element_justification="right")

# Required to manage whether there is an graph being shown currently, if yes, we clear the canvas
# when the user clicks on Calculate again
CANVAS_PLOT = {'canvas_yearly': False}
    
# Utility function that takes in canvas and figure objects
# creates a canvas object using the figure and returns the aggregated object
def draw_figure(canvas, figure):
    
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return figure_canvas_agg


# Utility function to calculate the wealth using the current_wealth, rate of mean return and noise (calcualted below)
def get_wealth(current_wealth, rate, noise):
    
    return current_wealth * (1 + (rate / 100.0) + noise) 

# Utility function to calculate noise based on the given formula in the problem.
# Takes in the user input of Std rate of return as sigma
def get_noise(sigma):
    
    return (sigma / 100.0) * np.random.randn(MAX_YEARS)


# Validation function, takes in the user inputs and validates 2 conditions:
# - Input should not be negative
# - Input years should not exceed MAX_YEARS - otherwise the program will error out.
def validate_input(sigma, rate, yearly_contribution, contribution_years, retirement_years, post_retirement_expenditure):
    should_plot = False           # A boolean flag to check if conditions are true, initialized with false
    
    # Analysis would be incorrect if the retirement_years or contribution_years exceed MAX_YEARS
    if (retirement_years > MAX_YEARS or contribution_years > MAX_YEARS):
        WINDOW['-wealth_label-'].update("Should not exceed maximum years")   # Show a user error saying that maximum years is exceeded.
    
    # Validate that all input are >= 0
    elif (sigma < 0 or rate < 0 or yearly_contribution < 0 or retirement_years < 0 or post_retirement_expenditure < 0):
        WINDOW['-wealth_label-'].update("All values should be numeric and > 0")
    
    # If all looks good, go for calculation. We could also add conditions to
    # All values are numeric, max limits on rates etc. 
    else:
        WINDOW['-wealth_label-'].update("")
        should_plot = True            # if none of the above conditions are true
    
    return should_plot                # return flag
    

# Utility function to calculate the wealth at the year of retirement
# Row = retirement_years - 1 because ideally a person can retire at 70 if MAX_YEARS is 70, so index would be 69
# traverses all columns for the number of analysis (10) in our case
# returns the total sum
def calculate_total_wealth(wealth_data, retirement_years):
    return np.sum( wealth_data[ (int ( retirement_years ) - 1), :], axis=0)


# Function to plot the graph and show the toal wealth in the UX
# Input - figure to show on the UX, total sum to show on the UX
def plot_and_update_user_interface(total_sum, figure):
    if (CANVAS_PLOT['canvas_yearly']):                                      # Check if yearly canvas is already published, if yes, then clean it
        CANVAS_PLOT['canvas_yearly'].get_tk_widget().forget()            
    
    CANVAS_PLOT['canvas_yearly'] = draw_figure(WINDOW['canvas'].TKCanvas, figure)  # Create a new canvas using the figure passed in above
    wealth_string = 'Wealth at retirement: ${:,}'.format(int(total_sum))
    
    WINDOW['-wealth_label-'].update(wealth_string)                          # Update the value on the UX


# Function to run the analysis
# We take in inputs as all the user inputs from the UX
# We loop CALCULATION_TIMES (10) default for the MAX_YEARS (70) times.
# It does the wealth calculation based on 3 cases - 1 contributing, 2 non-contributing before retirement 3 post retirement
# It finally plots the projections on a canvas in pysimplegui
def do_analysis(sigma, rate, yearly_contribution, contribution_years, retirement_years, post_retirement_expenditure):         
    wealth_data =  np.zeros((MAX_YEARS, CALCULATION_TIMES), dtype=float)      # create a wealth store, for each year for each analysis (MAX_YEAS x CALCULATION_TIMES)
    figure = plt.figure(figsize=(10, 8), dpi=100)                             # Initialize a 10 x 8 figure to add the plot.  

    for calculation_value in range(CALCULATION_TIMES):                     # We have to do 10 simulations, it iterates for each simulation
        noise = get_noise(sigma)                                           # Call get noise above to get the noise for the sigma user entered
        
        current_wealth = 0;                                                 # Initialize a temporary variable, wealth is zero initially for us all (unless inherited)
        last_year_you_had_money = 0;                                        # Initialize a temp variable to store the last year you have money, you're bankrupt after this. 
        
        for year in range(MAX_YEARS):                                       # Iterate for each year until the MAX_YEARS
            
        
            # Case 1 - the time you're saving money, contributions are coming in    
            if (year <= contribution_years):
                current_wealth = get_wealth(current_wealth, rate, noise[year]) + yearly_contribution # Formula provided in question

            # Case 2 - you've stopped contributing money, but you've not retired yet (not a good situation to be in)
            elif (year > contribution_years and year <= retirement_years):
                current_wealth = get_wealth(current_wealth, rate, noise[year])   # Formula provided in the question
                
            # Case 3 - you have retired, now you survive on your savings, until MAX_YEARS
            else:                
                current_wealth = get_wealth(current_wealth, rate, noise[year]) - post_retirement_expenditure   # Formula provided in question

            if (current_wealth >= 0):  # If current wealth is > = 0 we store the value, otherwise we initialized to zero already
                
                wealth_data[year, calculation_value] = current_wealth    # Store wealth corresponding to the year , calculation value of the simulation
                last_year_you_had_money = year    # track the last year where you had money
            # Do nothing otherwise
            else:
                break
        
        # To the plot add until the last year you had money, with the wealth data you had for the simulation value (calculation_value)
        plt.plot(range(last_year_you_had_money + 1), wealth_data[0 : last_year_you_had_money + 1, calculation_value], '-x')  # plot
        plt.title('Wealth Over 70 Years ')  # add a label for the plot
        plt.ylabel('Wealth')  # y axis label for the wealth
        plt.xlabel('Years')  # x axis label for the year
    
    
    # Invoke the calculate_total_wealth function above to ge the total wealth on the retirement year
    total_sum = calculate_total_wealth(wealth_data, retirement_years)
    
    # Calcualte the average : sum / count of simultations
    average_value = total_sum / CALCULATION_TIMES
    
    # Call the plot function to create a plot using the figure we initialized above and the average sum value
    plot_and_update_user_interface(average_value, figure)


# This is the way to create a window as per pySimpleGui docs
# Create the Window
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = WINDOW.read()              # read the WINDOW object
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks Quit, we break out of the infinite true loop
        break
    
    # Extract user input
    rate = float(values[0])  # User input for mean rate
    sigma = float(values[1])  # User input for std dev return
    yearly_contribution = float(values[2]) # User input for yearly contribution
    contribution_years = float(values[3])  # User input for contribution years (how many years)
    retirement_years = float(values[4])  # user input for retirement years (how many years)
    post_retirement_expenditure = float(values[5])  # User input for per year retirement expenditure
    
    # Input validations - basic (true or false will decide whether to plot or not)
    should_plot = validate_input(sigma, rate, yearly_contribution, contribution_years, retirement_years, post_retirement_expenditure)

    # If validations succeed
    if (should_plot): # call do analysis function below with user inputs
        do_analysis(sigma, rate, yearly_contribution, contribution_years, retirement_years, post_retirement_expenditure)
        

# Need to close the window when the infinite loop breaks on Quit click
WINDOW.close()