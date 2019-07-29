import os, sys, socket
import datetime
import pandas as pd
import csv, re
from termcolor import colored
import numpy as np
############################################################################################################
################################################ USER SPACE ################################################
############################################################################################################
####### General Comment section #######
ExperimentName = ''
ShortName = '' # short name in the log file: ShortName-Time.log
configpath = ['config_classification.py'] # Put here all configuration files to be saved
dataset_name = '' # Database Name, could be used inside config files
####### work only for one space texts #######
output_path = 'logs/'
GeneralComments = ''

############################################################################################################
############################################## END USER SPACE ##############################################
############################################################################################################

server = socket.gethostname()
user = os.environ['USER']
date = datetime.datetime.now()
try:configpath
except:configpath = '';print(colored('Warning: A configuration files (variable name: configpath) was not set for log file generation', 'red'))
if type(configpath)==str and configpath!='':
    with open(configpath, 'r') as configurations:
        config_text_version=configurations.read()
    config_text_version = configpath+'\n'+config_text_version
elif type(configpath)==list:
    config_text_version = '\n'
    for conf in configpath:
        with open(conf, 'r') as configurations:
            config_text=configurations.read()
        config_text_version = config_text_version+conf+'\n'+config_text+'\n'
else:
    config_text_version=''

def save_log(dicts='', 
                dataset_name_in=None, 
                GeneralComments_in=None, 
                ExperimentName_in=None, 
                ShortName_in=None, 
                output_path_in=None, 
                file=None,
                extra_name=None):
    global dataset_name, GeneralComments, ExperimentName, ShortName, configpath, output_path, date, server, user, config_text_version
    enddate = datetime.datetime.now()
    deltat = (enddate - date).total_seconds()
    deltat = str(int(deltat//3600)).zfill(4)+':'+str(int(deltat%3600//60)).zfill(2)+':'+str(int(deltat%3600%60)).zfill(2)
    #Replace Global names for local ones defined by main.
    if dataset_name_in: dataset_name = dataset_name_in
    if GeneralComments_in: GeneralComments = GeneralComments_in
    if ExperimentName_in: ExperimentName = ExperimentName_in
    if ShortName_in: ShortName = '-'+ShortName_in
    if output_path_in: output_path = output_path_in
    # WarningSection
    try:ExperimentName
    except:ExperimentName = '';print(colored('Warning: An Experiment name (variable name: ExperimentName) was not set for log file generation', 'red'))
    try:ShortName
    except:ShortName = '';print(colored('Warning: A short name (variable name: ShortName) was not set for log file generation', 'red'))
    try:dataset_name
    except:dataset_name = '';print(colored('Warning: A database name (variable name: dataset_name) was not set for log file generation', 'red'))
    try:GeneralComments
    except:GeneralComments = '';print(colored('Warning: A General comment section (variable name: GeneralComments) was not set for log file generation', 'red'))
    try:output_path
    except:output_path = './';print(colored('Warning: An Output path (variable name: output_path) was not set for log file generation', 'red'))

    experiment_file = str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)+'-'\
                     +str(date.hour)+str(date.minute).zfill(2)+str(date.second).zfill(2)
    experiment_date = str(date.year)+'/'+str(date.month).zfill(2)+'/'+str(date.day).zfill(2)+' - '\
                     +str(date.hour)+':'+str(date.minute).zfill(2)+':'+str(date.second).zfill(2)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.chmod(output_path, 0o777)
    if dicts:
        keysfromdict = list(dicts.keys())
        name = ShortName if ShortName else 'noname'
        if not os.path.isfile(output_path+name+'.csv'):
            df = pd.DataFrame({x:[] for x in keysfromdict})
            df['Processing Time'] = []
            df['Log Name'] = []
            df.to_csv(output_path+ShortName+'.csv', sep=',',index=False)
            os.chmod(output_path+ShortName+'.csv', 0o777)

        update_df = pd.read_csv(output_path+ShortName+'.csv')
        keysfromcsv = list(update_df.keys())
        update_df = pd.DataFrame({x:list(update_df[x]) for x in keysfromcsv})
        currentlenght = len(update_df[keysfromcsv[0]])
        for key in keysfromdict:
            update_df.loc[currentlenght,key] = dicts[key]
        update_df.loc[currentlenght,'Processing Time'] = deltat
        update_df.loc[currentlenght,'Log Name'] = output_path+ShortName+'-'+experiment_file
        update_df.to_csv(output_path+ShortName+'.csv', sep=',',index=False)

    # SAVE PROCESS FILE
    if file:
        df = pd.DataFrame(file)
        if not os.path.isfile(output_path+name+'_process.csv'):
            df.to_csv(output_path+ShortName+'_process.csv', sep=',',index=False)
            os.chmod(output_path+ShortName+'_process.csv', 0o777)

        update_df = pd.read_csv(output_path+ShortName+'_process.csv')
        keysfromcsv = list(update_df.keys())
        newkeys = list(df.keys())
        update_df = pd.DataFrame({x:list(update_df[x]) for x in keysfromcsv})
        large_size = np.max([len(update_df),len(df)])
        for key in newkeys:
            for c in range(large_size):
                if c<large_size and c<len(file[key]): update_df.at[c,key] = file[key][c]

        update_df.to_csv(output_path+ShortName+'_process.csv', sep=',',index=False)
    # cuttextin = 80
    # GeneralComments = re.sub(r'\s+', ' ',GeneralComments)
    # if len(GeneralComments)>cuttextin:
    #     chunks, chunk_size = len(GeneralComments)//cuttextin+1, cuttextin
    #     tempcomment = [ GeneralComments[i:i+chunk_size] for i in range(0, len(GeneralComments), chunk_size) ]
    #     GeneralComments = '\n' if GeneralComments[0] != '\n' else ''
    #     for i in tempcomment: GeneralComments += i + '\n' if i[-1] != '\n' else ''
    # else:
    #     GeneralComments = '\n'+GeneralComments
    name = ShortName+'-'+experiment_file if ShortName else experiment_file
    with open(output_path+name+'.log', "w") as text_file:
        print(f"\
Server: {server}\n\
User: {user}\n\
Experiment date: {experiment_date}\n\
Experiment Name: {ExperimentName}\n\
Specific Name: {ShortName}\n\
\n\
Database: {dataset_name}\n\
\n\
Time Processing: {deltat}\n\
\n\
General Comments: {GeneralComments}\n\
\n\
\n\
Configuration files:\n\n{config_text_version}\n\
", file=text_file)
