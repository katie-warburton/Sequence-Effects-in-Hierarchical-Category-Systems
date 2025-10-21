library(stringr) 
library(lmtest) 
library(dplyr) 
library(parameters) 
library(mclogit) 

df <- read.csv('Results/level2_trial_data.csv',  na.strings=c("","NA"))  
items <- sprintf("I%02d", 9:23)

df_1a <-  select(df, c(LOC, ORDER, NUM_A, NUM_X, NUM_B, P_ID, STIMULI, POOL))
df_1a <- df_1a[df_1a$ORDER != 'a',]

print(head(df_1a))