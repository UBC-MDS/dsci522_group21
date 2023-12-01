import click
import pandas as pd

@click.command()
@click.option('--cv-results', help='Path to the cross-validation results CSV file.')
@click.option('--output-cv-means', help='Path to save the mean cross-validation scores as a CSV file.')

def main(cv_results, output_cv_means):
    """ Process cross-validation results and save the mean values to a CSV file. """

    res = pd.read_csv(cv_results, header=[0, 1], index_col=0)

    res.xs(
        'mean',
        axis='columns',
        level=1
    ).to_csv(output_cv_means)


if __name__ == "__main__":
    main()
