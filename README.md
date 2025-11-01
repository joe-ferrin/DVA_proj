# DVA_proj
reddit-bot-detection

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
