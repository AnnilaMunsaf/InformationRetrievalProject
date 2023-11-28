import pandas as pd
import spacy as sp
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

if __name__ == '__main__':

    movies_df = pd.read_csv('./wiki_movie_plots_deduped.csv',
                           usecols=['Release Year', 'Title', 'Plot', 'Genre'])

    movies_df.info()







