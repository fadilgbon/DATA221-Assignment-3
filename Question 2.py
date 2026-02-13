import pandas
import matplotlib.pyplot as plt

crime_dataframe = pandas.read_csv("crime1.csv") #reads the data from the csv and stores it in a dataframe

plt.hist(crime_dataframe["ViolentCrimesPerPop"]) #creates the histogram from the data
plt.title("Distribution of Violent Crimes per Population") #add a title to the graph
plt.xlabel("Violent Crimes per Population") #add a x-axis label to the graph
plt.ylabel("Frequency") #add a y-axis label to the graph
plt.show() #shows the plot

plt.boxplot(crime_dataframe["ViolentCrimesPerPop"]) #creates the boxplot from the data
plt.title("Distribution of Violent Crimes per Population") #add a title to the graph
plt.xlabel("Violent Crimes per Population") #add a x-axis label to the graph
plt.ylabel("Frequency")#add a y-axis label to the graph
plt.show() #shows the plot
#the data is right skewed meaning that most of teh data points lie below the 50th perctile or on the left side of the median with few larger values past the 50th percentile
#the median is the line under the 0.4 marker on the scale meaning that the mean is not centered so the graph is skewed.
#the scale is much larger compared to the IQR of the data therefore it shows that there are outliers in the data with the largest of thm ebing on the right of center.