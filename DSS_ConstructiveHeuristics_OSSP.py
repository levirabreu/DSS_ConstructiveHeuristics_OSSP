from ConstrutiveHeuristics import MIH
from Tools import decoding, makespan
from insertion_heuristic import insert_heuristic

import docplex.cp.utils_visu as visu
from tabulate import tabulate
import numpy as np
import pandas as pd 

from datetime import datetime, timedelta
import io

import webbrowser

import PySimpleGUI as sg

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1918)

def plot_gantt(s, p):
    
    m, n = p.shape
    
    M = range(m)
    N = range(n)
    
    operations = [(machine, job) for machine in M
        for job in N]
    
    interval_list = [[] for i in M]
    
    Mac = [0 for i in M]
    Job = [0 for j in N]
    
    for idx in s:
        i, j = operations[idx]
        
        start = max(Mac[i], Job[j])
        end = max(Mac[i], Job[j]) + p[i][j]
        
        interval_list[i].append(
            (start, end, j, f'$J_{{{j+1}}}$')
            )
        
        Mac[i] = max(Mac[i], Job[j]) + p[i][j]
        Job[j] = Mac[i]
        

    for i in M:

        visu.sequence(name=f"$M_{{{i+1}}}$", intervals=interval_list[i])
        
    return visu.show(pngfile='gantt_ossp.png')
        

def print_table_machine_numeric(s, p):
    

    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    
    table = []
    
    m, n = p.shape
    
    M = range(m)
    N = range(n)
    
    operations = [(machine, job) for machine in M
        for job in N]
        
    Mac = [0 for i in M]
    Job = [0 for j in N]
    
    for idx in s:
        i, j = operations[idx]
        
        start = max(Mac[i], Job[j])
        end = max(Mac[i], Job[j]) + p[i][j]
        duration = p[i][j]
        
        Mac[i] = max(Mac[i], Job[j]) + p[i][j]
        Job[j] = Mac[i]
        
        Machine = f'M{i+1}'.translate(SUB)
        Task = f'J{j+1}'.translate(SUB)
        
        table.append([Machine, Task, start, end, duration])
        
    headers = ['Machine', 'Job', 'Start (h)', 'End (h)', 'Duration (h)']
    
    table_fmt = tabulate(table, headers=headers, tablefmt="grid")
    
    #print(table_fmt)
    
    return table_fmt

def print_table_machine_date(s, p, initial_date):
    
    initial_date = datetime.strptime(initial_date, "%d-%m-%Y %H:%M")

    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    
    table = []
    
    m, n = p.shape
    
    M = range(m)
    N = range(n)
    
    operations = [(machine, job) for machine in M
        for job in N]
        
    Mac = [initial_date for i in M]
    Job = [initial_date for j in N]
    
    for idx in s:
        i, j = operations[idx]
        
        start = max(Mac[i], Job[j])
        end = max(Mac[i], Job[j]) + timedelta(hours=p[i][j].item())
        duration = p[i][j]
        
        Mac[i] = max(Mac[i], Job[j]) + timedelta(hours=p[i][j].item())
        Job[j] = Mac[i]
        
        Machine = f'M{i+1}'.translate(SUB)
        Task = f'J{j+1}'.translate(SUB)
        
        table.append([Machine, Task, start.strftime("%d-%m-%Y %H:%M"), end.strftime("%d-%m-%Y %H:%M"), duration])
        
    headers = ['Machine', 'Job', 'Start (date)', 'End (date)', 'Duration (h)']
    
    table_fmt = tabulate(table, headers=headers, tablefmt="grid")
    makespan_date = max(Job)
    
    #print(table_fmt)
    
    return table_fmt, makespan_date


layout = [
    [sg.Text("Number of jobs/batches:            "), sg.Input(key='n',default_text=3), sg.Text("Start date:            "), sg.Input(default_text=datetime.now().strftime("%d-%m-%Y %H:%M"), key='initial_date', size=(67,10))],
    [sg.Text("Number of machines/resources: "), sg.Input(key='m',default_text=3), sg.Text("Time horizon:       "), sg.Radio('Numeric', "time", key='Numeric', default=True), sg.Radio('Date and hour', "time", key='Date'), sg.Button('Create processing times', key='-p-'), sg.Button('Create scheduling', key='-OK-')],
    [sg.Multiline(key='-PT-', write_only=False, size=(10,10), reroute_cprint=True)],
    [sg.Text("", size=(0, 1), key='OUTPUT'), sg.Text("", size=(0, 1), key='gantt_title')],
    [sg.Multiline(key='-ML-', write_only=True, size=(100,18), reroute_cprint=True), sg.Image(size=(1000, 300), key='gantt', expand_x=True, expand_y=True)],
    [sg.Text('See more information about the problem and the constructive heuristic used to solve it at the LINK.', enable_events=True, key='url', tooltip='https://doi.org/10.1016/j.cor.2022.105744')]
]
 
window = sg.Window("Open shop scheduling problem solver - makespan minimization - developed by Levi R. Abreu", layout, resizable=True, finalize=True, icon='software_icon.ico')

window['-PT-'].expand(True, True)
window['-ML-'].expand(True, True)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    
    elif event == '-p-':
        
        n = int(values['n'])
        m = int(values['m'])
        
        df = pd.DataFrame(np.zeros((m, n)), columns=[f"J_{i}" for i in range(1, n+1)],
                          index=[f"M_{i}" for i in range(1, m+1)])

        edited_df = df.iloc[:, :].astype(np.int64).copy()

        edited_df.iloc[:, :] = np.random.randint(1,100, size=(m, n), dtype=np.int64)
        
        window['-PT-'].update(edited_df.to_string(), font='Courier 11')
        
    elif event == '-OK-' and values['-PT-']:
        
        df_str = values['-PT-']
        edited_df = pd.read_csv(io.StringIO(df_str), sep='\s+')
        p = edited_df.to_numpy().astype(np.int64)

        window['-ML-'].update('')
        
        
        s = MIH(p, ls=False)
        s = insert_heuristic(s, p)
        
        s = np.array(s, dtype=np.int64)
        p = np.array(p, dtype=np.int64)
        s = decoding(s, p)
        
        obj_makespan = str(makespan(s, p))
    
        plot_gantt(s, p)    
        window['gantt'].update(filename='gantt_ossp.png', visible=True)
        window['gantt_title'].update(' '*227 + f'Gantt chart for OSSP - Makespan {obj_makespan} (h)')
        
        if values['Numeric'] == True:

            table_fmt = print_table_machine_numeric(s, p)
                    
            title = 'Makespan provided by MIH-IST heuristic: ' + obj_makespan + ' (h)'
            
            sg.cprint(' '*9, font='Courier 11', end=' ')
            sg.cprint(title, font='Courier 11', end = ' ')
            sg.cprint(table_fmt, font='Courier 11')
            
        else:
            initial_date = values['initial_date']
             
            table_fmt, makespan_date = print_table_machine_date(s, p, initial_date)
                    
            title = 'Makespan provided by MIH-IST heuristic: ' + makespan_date.strftime("%d-%m-%Y %H:%M") + ' (h)'
            
            sg.cprint(' '*9, font='Courier 10', end=' ')
            sg.cprint(title, font='Courier 10', end = ' ')
            sg.cprint(table_fmt, font='Courier 11')
            
    elif event.startswith("url"):

        webbrowser.open('https://doi.org/10.1016/j.cor.2022.105744')
        

        
window.close()