# DVA_proj
reddit-bot-detection

I cleaned the code and organized it so each part can be used. I implemented DGI so that features can be embedded with the graph structure. 
See the 'Implementation' notebook

How it works:
1) Load / clean data
2) Create dataframe with engineered features per user
3) Create dataframe with edges for users on the same post (limit by a threshold for weight of the edges)
4) Using the edge and feature dataframes, create graph as a pytorch geometric object
5) Run DGI with the graph
6) The output of DGI is an embedding that can be used in anomaly detection


"Joe_notebook" file:

I engineered some features, created a graph network, and combined them into DOMINANT https://arxiv.org/abs/2406.04690v1 
This is similar to DGI, which takes a graph as well as features, and makes an embedding, which can then be fed into anomaly detection. DOMINANT takes the graph and features and goes right into outputting anomaly detection scores.

I was having some trouble getting the imports for pytorch working. I included a yaml, though you might need to change the pytorch version based on your GPU. Also in the yaml I put two lines that have to be run on their own in pip.

Features: 
User features
Number of posts
Average length of posts
Average upvotes, downvotes, scores
Temporal features
Average time in between posts
Fraction of the hours of the day that the user makes posts
Diversity of posting times (entropy)
Text features
Word diversity (amount of different words used)
Average length of posts
Semantic similarity with language model
