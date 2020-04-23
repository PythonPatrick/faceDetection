import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


class correlationPlots:

    def __init__(self, df):
        self.df=df

    def history(self):
        self.df.hist()
        plt.show()

    def density(self):
        self.df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
        plt.show()

    def box(self):
        self.df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
        plt.show()

    def correlation(self):
        plt.matshow(self.df.corr())
        plt.xticks(range(len(self.df.columns)), self.df.columns)
        plt.yticks(range(len(self.df.columns)), self.df.columns)
        plt.colorbar()
        plt.show()

    def scatterplot(self):
        pd.scatter_matrix(self.df, figsize=(6, 6))
        plt.show()

