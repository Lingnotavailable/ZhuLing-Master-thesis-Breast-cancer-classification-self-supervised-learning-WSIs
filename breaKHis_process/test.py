# PPHA 30537
# Spring 2024
# Homework 4

# YOUR NAME HERE

# YOUR CANVAS NAME HERE
# YOUR GITHUB USER NAME HERE

# Due date: Sunday May 12th before midnight
# Write your answers in the space between the questions, and commit/push only
# this file to your repo. Note that there can be a difference between giving a
# "minimally" right answer, and a really good answer, so it can pay to put
# thought into your work.

##################

# Question 1: Explore the data APIs available from Pandas DataReader. Pick
# any two countries, and then 
#   a) Find two time series for each place
#      - The time series should have some overlap, though it does not have to
#        be perfectly aligned.
#      - At least one should be from the World Bank, and at least one should
#        not be from the World Bank.
#      - At least one should have a frequency that does not match the others,
#        e.g. annual, quarterly, monthly.
#      - You do not have to make four distinct downloads if it's more appropriate
#        to do a group of them, e.g. by passing two series titles to FRED.

import pandas as pd
from pandas_datareader import wb
import pandas_datareader as pdr
import datetime
#Setting absolute path
absolute_path = r'/Users/zhengcui/Desktop/Python/HW4'

# Define the time period
start_1 = datetime.datetime(2000, 1, 1)
end_1 = datetime.datetime(2020, 12, 31)
start_2 = datetime.datetime(2002, 1, 1)  # Align start date for simplicity
end_2 = datetime.datetime(2018, 12, 31)  # Align end date for simplicity

# Countries of interest
countries = ['USA', 'CAN']

# Fetch GDP data from World Bank
gdp_data = wb.download(indicator='NY.GDP.MKTP.CD', country=countries, start=start_1, end=end_1).reset_index()
# Fetch unemployment rates from FRED
cpiallminmei = pdr.DataReader(['USACPIALLMINMEI', 'CANCPIALLMINMEI'], 'fred', start_2, end_2)  # Updated with correct series IDs for unemployment

#   b) Adjust the data so that all four are at the same frequency (you'll have
#      to look this up), then do any necessary merge and reshaping to put
#      them together into one long (tidy) format dataframe.
cpiallminmei = cpiallminmei.resample('A').mean()
cpiallminmei.reset_index(inplace=True)

gdp_data.loc[gdp_data['country'] == 'Canada', 'country'] = 'CAN'
gdp_data.loc[gdp_data['country'] == 'United States', 'country'] = 'USA'

cpiallminmei = cpiallminmei.melt(id_vars='DATE', var_name='country', value_name='CPIALLMINMEI')
cpiallminmei['country'] = cpiallminmei['country'].str.replace('CPIALLMINMEI', '')
cpiallminmei['year'] = cpiallminmei['DATE'].dt.year.astype(str)
cpiallminmei = cpiallminmei.drop('DATE', axis=1)
merged_data = pd.merge(gdp_data, cpiallminmei, on=['country', 'year'], how='inner')

# Display the result
print(merged_data.head())


#   c) Finally, go back and change your earlier code so that the
#      countries and dates are set in variables at the top of the file. Your
#      final result for parts a and b should allow you to (hypothetically) 
#      modify these values easily so that your code would download the data
#      and merge for different countries and dates.
#      - You do not have to leave your code from any previous way you did it
#        in the file. If you did it this way from the start, congrats!
#      - You do not have to account for the validity of all the possible 
#        countries and dates, e.g. if you downloaded the US and Canada for 
#        1990-2000, you can ignore the fact that maybe this data for some
#        other two countries aren't available at these dates.

#   d) Clean up any column names and values so that the data is consistent
#      and clear, e.g. don't leave some columns named in all caps and others
#      in all lower-case, or some with unclear names, or a column of mixed 
#      strings and integers. Write the dataframe you've created out to a 
#      file named q1.csv, and commit it to your repo.
merged_data.to_csv('q1.csv', index=False)

# Question 2: On the following Harris School website:
# https://harris.uchicago.edu/academics/design-your-path/certificates/certificate-data-analytics
# There is a list of six bullet points under "Required courses" and 12
# bullet points under "Elective courses". Using requests and BeautifulSoup:
#   - Collect the text of each of these bullet points
#   - Add each bullet point to the csv_doc list below as strings (following
#     the columns already specified). The first string that gets added should be
#     approximately in the form of:
#     'required,PPHA 30535 or PPHA 30537 Data and Programming for Public Policy I'
#   - Hint: recall that \n is the new-line character in text
#   - You do not have to clean up the text of each bullet point, or split the details out
#     of it, like the course code and course description, but it's a good exercise to
#     think about.
#   - Using context management, write the data out to a file named q2.csv
#   - Finally, import Pandas and test loading q2.csv with the read_csv function.
#     Use asserts to test that the dataframe has 18 rows and two columns.

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fetch the webpage
url = "https://harris.uchicago.edu/academics/design-your-path/certificates/certificate-data-analytics"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Prepare the CSV document
csv_doc = ['Type,Description']

# Extract required courses
a = soup.find('required courses')
required_courses = soup.find('h3', string='Required courses').find_next_siblings('ul', limit=2)
for result in required_courses:
    lis = result.find_all('li')
    for li in lis:
        text = f"required,{li.text.strip()}"
        csv_doc.append(text)
# Extract elective courses
elective_courses = soup.find('h3', string='Elective courses').find_next_siblings('ul')
for couses in elective_courses:
    lis = couses.find_all('li')
    for li in lis:
        text = f"elective,{li.text.strip()}"
        csv_doc.append(text)

# df = pd.DataFrame([x.split(',') for x in csv_doc], columns=['Type', 'Description'])

path = 'q2.csv'


with open(path, 'w', encoding='utf-8') as file:
    for item in csv_doc:
        file.write(item + '\n')


#Testing data frame
data = pd.read_csv('q2.csv')
print(data.head())
assert data.shape[0] == 18
assert data.shape[1] == 2