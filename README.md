# Titanic 
Practice Pandas and NumPy techniques for later use in ML projects pipelines.

TASKS

1) Clean Dataset: Define what "clean dataset" means and figure out ways to execute it (e.g drop missing values or impute them):

    a. Create a function that counts every column's missing values, if any, and returns the names of the columns, for which missing values have been reported, the corresponding counts and percentages of their missing values, and a barplot that collectivelly illustrates these pieces of information in a clear and compact form,

    b. Create a function that imputes the missing values in column "Age" via Linear Reggression (check, [here](https://rstudio-pubs-static.s3.amazonaws.com/98715_fcd035c75a9b431a84efca8b091a185f.html)),

    c. Create a function that imputes the missing values in column "Cabin". Here, use a probabilistic[1] method to do so, and instead of the exact cabin that the person occupied, try to predict the deck that it was situated into, and store the new information in a new column titled "Deck". As to why, have a look [here](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial/notebook), under paragraph "1.2.4 Cabin." Actually, there is a clear connection between the ticket class of a passenger (Pclass) and the location of his/her cabin on the ship. Although this information has already been captured in "Pclass", thus adding little predictive power, it would be a useful way to exercise data cleaning techniques, using Pandas and NumPy.

    [1]: Technically, all passengers' cabins were located in Decks A, B, C, D, E, F, and G. Most of the 1st class passengers' cabins, were in the decks A, B, and C, which were exclusively allocated for them. The rest of the 1st class's cabins were situated in decks D and E, yet these decks had also cabins, which were ment for all passengers. The last two decks, namelly decks F and G, had been provided only to 2nd and 3rd class passengers. All these facts, and more, are collectivelly illustrated and quantified in the "cabins.csv"[2] file, which can be found in the files above. Given these pieces of information, and the total number of passengers in the dataset, try to solve the problem at hand.

    [2]: In "cabins.csv" file is given the number of passengers, in total (total) and per class (fc, sc, tc, fsc, stc, and fstc), who could be accomodated in every deck.
For example, take deck E that can accomodate a total number of 597 (total) passengers. Specifically, 16 (fc) of these passengers could be 1st class, 62 (sc) 2nd class,
and 212 (tc) 3rd class. Apart from these cabins, there were also a number of other cabins that could accomodate, either 1st, or 2nd class passengers. These passengers,
in total, were 319 (fsc, i.e. 1st and 2nd class). As you can imagine, although these cabins could accomodate both classes, 1st class passengers would enjoy higher priority,
in case of an overbooking of other 1st class cabins had happend.

2) Basic EDA: Create some basic plots about variables of interest and explore what relationship (bivariate) they have with the target variable. Experiment with matplotlib, Seaborn.
3) Explore multivariate relationships via graphs. Identify a few plots designed to summarize information from several variables simultaneously. Explain how they work.
4) Explore your data via groups and report any interesting findings. Only use basic plots here if necessary.
5) Preprocess your data for input into machine learning algorithms in the SCIKIT-LEARN library. What are the steps required to transform a raw dataset into one compatible with the       aforementioned library? 
6) Create a helper .py file with your helper functions for better organization. Use Colab, clone your git and figure out how to run your helpers in there. 

More ideas and helper code: https://www.kaggle.com/c/titanic/code 
