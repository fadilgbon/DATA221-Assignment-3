import pandas

crime_dataframe = pandas.read_csv("crime1.csv") #reads the data from the csv and stores it in a dataframe
violent_crimes_per_pop_mean = crime_dataframe["ViolentCrimesPerPop"].mean() #computes the mean
violent_crimes_per_pop_median = crime_dataframe["ViolentCrimesPerPop"].median()#finds the median
violent_crimes_per_pop_standard_deviation = crime_dataframe["ViolentCrimesPerPop"].std() #computes the standard deviation
violent_crimes_per_pop_maximum_value = crime_dataframe["ViolentCrimesPerPop"].max() #finds the maximum value
violent_crimes_per_pop_minimum_value = crime_dataframe["ViolentCrimesPerPop"].min() #finds the minimum value
print(f'Mean: {violent_crimes_per_pop_mean}')
print(f'Median: {violent_crimes_per_pop_median}')
print(f'Standard Deviation: {violent_crimes_per_pop_standard_deviation}')
print(f'Maximum Value: {violent_crimes_per_pop_maximum_value}')
print(f'Minimum Value: {violent_crimes_per_pop_minimum_value}')

#The distribution is right skewed. As the mean value is higher than the median, there are larger data points moving the graph towards a more right skewed shap
#The mean is more affected by extremes because the mean if calculated using every data value so large outliers can largely skew the mean, while the median is the middle value in the data so outliers don't have an effect on it.