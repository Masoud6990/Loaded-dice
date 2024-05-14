# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

# Defining needed functions
def word_fre(Set_of_sums,List_of_sums):
    frequency_word=[]
    for i in Set_of_sums:
        frequency_word.append(List_of_sums.count(i))
    return frequency_word
    print(frequency_word)

def CDF_fn(Set_of_sums_list,Prob):
    Prob_CDF=[]
    sum=0
    for j in range(len(Set_of_sums_list)):
        sum+=Prob[j]
        Prob_CDF.append(sum)
    return Prob_CDF

def index_finder(CDF,limit):
    for i in range(len(CDF)):
        if CDF[i]>limit:
            break
    return CDF[i-1]


dice = [1,2,3,4,5,6]
sum_results = []
number_iterations = 1000
# Setting a random seed  for reproducibility
np.random.seed(42)

for i in range(number_iterations):
    # Throw the first dice
    throw_1 = np.random.choice(dice)
    # Throw the second dice
    throw_2 = np.random.choice(dice)
    # Sum the result
    sum_throw = throw_1 + throw_2
    # Append to the sum_result list
    sum_results.append(sum_throw)

greater_5_count = 0

for x in sum_results:
    if x > 5:
        greater_5_count += 1

probability = greater_5_count/len(sum_results)
print(f"The probability by this simulation is: {probability}")

print("Mean:",np.mean([1/6,1/6,1/6,1/6,1/6,1/6]))
print("Variance:",np.var([1/6,1/6,1/6,1/6,1/6,1/6]))

dice=[1,2,3,4,5,6]
np.random.seed(42)
List_of_sumss=[]
num_iteration=100000
for i in range(num_iteration):
    throw1=np.random.choice(dice)
    throw2=np.random.choice(dice)
    sum_dice=throw1+throw2
    List_of_sumss.append(sum_dice)
Set_of_sumss=set(List_of_sumss)


Set_of_sumss_list=list(Set_of_sumss)
print(Set_of_sumss_list)

frequency_word=[]
for i in Set_of_sumss:

    frequency_word.append(List_of_sumss.count(i))

print(frequency_word)

plt.bar(Set_of_sumss_list, frequency_word, color='blue', label='Set Points')

dice=[1,2,3,4]
np.random.seed(42)
List_of_sums=[]
List_of_throw1=[]
List_of_throw2=[]
num_iteration=100000
for i in range(num_iteration):
    throw1=np.random.choice(dice)
    List_of_throw1.append(throw1)
    throw2=np.random.choice(dice)
    List_of_throw2.append(throw2)
    sum_dice=throw1+throw2
    List_of_sums.append(sum_dice)
Set_of_sums=set(List_of_sums)
word_fre2=word_fre(Set_of_sums,List_of_sums)

Prob=[]
for i in range(len(word_fre2)):
    Prob.append(word_fre2[i]/iterations)
print(Set_of_sums)
print(Prob)

print("Mean:",np.mean(list(Set_of_sums)))
print("Variance:",np.var(list(Set_of_sums)))
print("Covariance between first and second row:",np.cov(List_of_throw1,List_of_throw2))

dice2=[1,2,2,3,4]
np.random.seed(42)
List_of_sums=[]
iterations=100000
for i in range(iterations):
    throw1=np.random.choice(dice2)
    throw2=np.random.choice(dice2)
    List_of_sums.append(throw1+throw2)
Set_of_sums=set(List_of_sums)
Set_of_sums_list=list(Set_of_sums)
word_fre2=word_fre(Set_of_sums,List_of_sums)

plt.bar(Set_of_sums_list,word_fre2)

dice2=[1,2,3,3,4,5,6]
np.random.seed(42)
List_of_sums=[]
iterations=100000
for i in range(iterations):
    throw1=np.random.choice(dice2)
    throw2=np.random.choice(dice2)
    List_of_sums.append(throw1+throw2)
Set_of_sums=set(List_of_sums)
Set_of_sums_list=list(Set_of_sums)
word_fre2=word_fre(Set_of_sums,List_of_sums)

Prob=[]
for i in range(len(word_fre2)):
    Prob.append(word_fre2[i]/iterations)

print(Prob)

CDF=CDF_fn(Set_of_sums_list,Prob)

index_finder(CDF,0.5)

CDF_fn(Set_of_sums_list,Prob).index(0.4474)

Set_of_sums_list[CDF_fn(Set_of_sums_list,Prob).index(index_finder(CDF,0.5))]

dice=[1,2,3,4,5,6]
np.random.seed(42)
List_of_sums=[]
iterations=1000
for i in range(iterations):
    throw1=np.random.choice(dice2)
    if throw1 >3:
        throw2=0
    else:
        throw2=np.random.choice(dice2)
    List_of_sums.append(throw1+throw2)
Set_of_sums=set(List_of_sums)
Set_of_sums_list=list(Set_of_sums)
word_fre2=word_fre(Set_of_sums,List_of_sums)

plt.bar(Set_of_sums_list,word_fre2)

dice=[1,2,3,4,5,6]
np.random.seed(42)
List_of_sums=[]
iterations=100000
for i in range(iterations):
    throw1=np.random.choice(dice)
    if throw1 >=3:
        throw2=np.random.choice(dice)
    else:
        throw2=0
    List_of_sums.append(throw1+throw2)
Set_of_sums=set(List_of_sums)
Set_of_sums_list=list(Set_of_sums)
word_fre3=word_fre(Set_of_sums,List_of_sums)
print(Set_of_sums_list)

plt.bar(Set_of_sums_list,word_fre3)


dice=[1,2,3,4,5,6,7]
np.random.seed(42)
List_of_sums=[]
List_of_throw1=[]
List_of_throw2=[]
num_iteration=100000
for i in range(num_iteration):
    throw1=np.random.choice(dice)
    List_of_throw1.append(throw1)
    throw2=np.random.choice(dice)
    List_of_throw2.append(throw2)
    sum_dice=throw1+throw2
    List_of_sums.append(sum_dice)
Set_of_sums=set(List_of_sums)
word_fre(Set_of_sums,List_of_sums)
print("Mean:",np.mean(list(Set_of_sums)))
print("Variance:",np.var(list(Set_of_sums)))
print("Covariance between first and second row:",np.cov(List_of_throw1,List_of_throw2))

dice=[1,2,3,4,5,6,6]
np.random.seed(42)
List_of_sums=[]
List_of_throw1=[]
List_of_throw2=[]
num_iteration=100000
for i in range(num_iteration):
    throw1=np.random.choice(dice)
    List_of_throw1.append(throw1)
    throw2=np.random.choice(dice)
    List_of_throw2.append(throw2)
    sum_dice=throw1+throw2
    List_of_sums.append(sum_dice)
Set_of_sums=set(List_of_sums)
word_fre2=word_fre(Set_of_sums,List_of_sums)
Set_of_sums_list=list(Set_of_sums)
Prob=[]
for i in range(len(word_fre2)):
    Prob.append(word_fre2[i]/iterations)
print(Set_of_sums)
print(Prob)
print("Mean:",np.mean(list(Prob)))
print("Variance:",np.var(list(Prob)))
print("Covariance between first and second row:",np.cov(List_of_throw1,List_of_throw2))
#Set_of_sums_list
plt.bar(Set_of_sums_list,Prob)
