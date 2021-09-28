#import pandas
import pandas

#load csv file using pandas read_csv() into a dataframe called ghge
csv_file = "GHG.csv"
ghge = pandas.read_csv( r"C:\Users\agiop\Downloads\GHG.csv" ) #whis works given that the working direcotry
                                                              #is correctly specified
                                                              #"r" in front of a normal string converts it into
                                                              #row path that can be handled by pandas.read_csv()
                                                              
#store a dataset explicitly with pandas.DataFrame() using a dictionary or list
d = { "a": [ 1, 2, 3 ], "b": [ 4, 5, 6 ], "c": [ 7, 8, 9 ] }
df_dataframe = pandas.DataFrame( d )

s_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
df_series_list = pandas.Series( s_list, name = "s" )

s_dict = { "a": 1, "b": 2, "c": 3 }
df_series_dict = pandas.Series( s_dict, name = "s" )

#select specific column(s)
col = "year"
year = ghge[ col ] #year is called a series

cols = [ "year", "data_value" ]
ghge[ cols ] #using a list to secify multiple column names
             #ghge[ cols ] is a dataframe. not a series

#select specific row(s)
value = 2010
condition = ( year >= value )
ghge[ condition ] #select only the rows where year is greater or equal to value
                  #the row filtering is performed using conditional statements
                  #any single-column boolean dataframe of the same length as
                  #ghge's would work

#select specific row(s) and column(s): loc and iloc
ghge.loc[ condition, cols ] #use loc when specification by names is required                      
ghge.iloc[ 0:10, ]          #use iloc when specification by index is required
                            #before comma: rows, after comma: columns
                            #when one is left blank no selection is applied

#assign new values to existent data
ghge.iloc[ 0, 5 ] = ...  #single value
ghge.iloc[ 1:3, 5 ] = ... #multiple values

#create new columns using existent columns
