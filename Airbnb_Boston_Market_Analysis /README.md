#### The following are packages/libraries have been used in this project:
<br> -pandas
<br> -datetime
<br> -pgeocode
<br> -plotly.express
<br> -LinearRegression
<br> -train_test_split
<br> -r2_score
<br> -mean_squared_error
<br> -sklearn.linear_model
<br> -sklearn.model_selection
<br> -sklearn.metrics
#### Data sources for this project
<br> Calendar.csv: contains date, availability, and price
<br> listings.csv: contains all booking list info

##### Here is link to medium blog:https://medium.com/@zhosheng1128/airbnb-market-in-boston-7796af083279 

##### Acknowledgment: https://plotly.com/python/bubble-maps/

##### Proposed three questions:
1. For airbnb booking in the Boston area, is there any strong seasonality? If so, what is the seasonal pattern with regard to the number of booking changes?
2. What are the top five influential listing factors in the Boston market?
3. Is there any geographic preference in the Boston market?

  For the first question, there were two python packages that had been imported, pandas and datetime. The data set calendar.csv was used for this analysis. I calculated the average number of daily bookings and plotted on the line chart. As a result, there were some seasonalities in the Boston market. It seems that the peak season is in September and it falls sharply throughout the winter season until it hits the bottom in December. Then the number of booking starts slowly picks up as the temperature gets warmer and finally stabilizes throughout the whole summer.
  
  
  For the second question, pandas and sklearn were imported for this analysis. The data set was pulled from listing.csv. The methodology I applied was to calculate the correlation coefficients and then picked the top five attributes. I applied the selected attributes to the linear regression model. To verify the fitness of attribute selection, I splitted the data set to training and test data sets by 70% to 30% ratio. Finally, I calculated the r-square value to check the correctness of attribute selection. The 0.41 value confirmed my selection was fairly good.
  
  
  For the third question, pandas, pgeocode, and plotly were imported for this analysis. The data set was pulled from listing.csv. I calculate the total number of bookings for each zip code. Based on the zip code, we plot the dots on the map. The size and color of each dot is based on the total number of bookings in that particular zip code. The following heat map tells that the preferred location is on the south side of the Cambridge area 
