<img align="left" width="220" src="https://raw.githubusercontent.com/levirabreu/DSS_ConstructiveHeuristics_OSSP/main/software_icon.ico" />

# A decision support system based on constructive heuristics for Open shop scheduling problem with makespan minimization

Open shop scheduling problem solver - makespan minimization

## Software Overview

<img align="center" width="800" src="https://github.com/levirabreu/DSS_ConstructiveHeuristics_OSSP/blob/main/OSSP_solver_software_overview.gif?raw=true" />

## Description

A solver for the production scheduling problem in an open shop environment with makespan minimization. 

The algorithm used to solve the problem is the Minimal Idleness Heuristic with cheap InSerTion procedure [MIH-IST] (see the article by [Abreu et al. (2022)](https://doi.org/10.1016/j.cor.2022.105744)).

## Getting Started

### Dependencies

* python 3.10
* cython
* numpy
* matplotlib
* pandas
* docplex
* tabulate
* PySimpleGUI 4.60.5
* pyinstaller

### Installing

* You can download the latest installable version of Open shop scheduling problem solver - makespan minimization for Windows on releases page.
* You need just unpack the .zip file and execute DSS_ConstructiveHeuristics_OSSP.exe file.

### Executing program

* First, you need to define the number of jobs and machines in the production environment.
* Then, you need to define the start date of the planning horizon.
* Then click on **[Create processing time]** button to show the processing time matrix with random times between 0 and 1.
* Finally, click on **[Create scheduling]** button to show the generated schedule in text and on a Gantt chart.

## Help

Some of the decision support system's features:

* You can select the time horizon in numerical time format or in date and time format.
* You can edit the processing times for each machine-job operation in the processing time matrix created. The new processing times must be integers, with a single space delimiter.

## Authors

Contributors names and contact info.
  
* [@Levi R. Abreu](https://scholar.google.com.br/citations?user=hbm0KAoAAAAJ)
* [@Marcelo S. Nagano](https://scholar.google.com.br/citations?user=3BFXZQoAAAAJ)

## Version History

* 0.1
    * Initial Release.
* 0.2
	* Fix icon image in window icon and title bar.
	
## Intellectual Property Rights

This project is registered with Brazil's national intellectual property institute under the code BR512024000879-9.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

* [Lab LAOR from EESC-USP](http://www.laor.prod.eesc.usp.br/)
* [Lab OPL from UFC](http://www.opl.ufc.br/en/)