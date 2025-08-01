# FFDataAnalysis24
This repository should serve as a sandbox for doing some data analysis between the 2024 and 2025 NFL season


### Contributions
Stats are collected from a myraid of places, mainly from FantasyPros
More detailed urls can be found at https://github.com/jmandzak/FFDataCollection 


### Analysis Scripts
`correlation.py` - Runs a correlation analysis between Final PPG and each stat recorded before the season started.
Images saved when script is run.

`pos_analysis.py` - Run with a single position as command line argument [QB, RB, WR, TE, DEF, K]. Plots each feature against final PPG and adds a line of best fit to show correlation.

`sos_analysis.py` - Performs two tasks:

1. Creates a 3d plot for each position of final PPG vs position rank vs strength of schedule. This is meant to show how strength of schedule impacts PPG while also keeping in mind that players very far apart from each other in position rank are less likely to have SoS play as large a role

2. For each position, creates pairs of players within a certain range of position rank of each other and a minimum SoS difference. These pairs are then analyzed to see if the player with the better strength of schedule ended with more PPG, and at the end of each position the number of times SoS was "correct" vs "incorrect" is shown based on PPG comparisons from the pairs.
 