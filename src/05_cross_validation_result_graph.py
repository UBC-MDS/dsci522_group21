# +
import click
import pandas as pd

def process_data(file_path):
    cross_val_results = pd.read_csv(file_path, header=[0, 1], index_col=0)

    Fig_6 = cross_val_results.xs(
        'mean', 
        axis='columns', 
        level=1
    )
    return Fig_6

@click.command(context_settings={'ignore_unknown_options': True})
@click.option('--file-path', default='../data/processed/cv_results.csv', help='Path to the cv_results.csv file.')
@click.option('--output-path', default='../data/processed/Fig_6.xlsx', help='Output path for the Excel file.')
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def main(file_path, output_path, args):
    Fig_6 = process_data(file_path)

    with pd.ExcelWriter(output_path) as writer:
        Fig_6.style.format(precision=2).background_gradient(
            cmap='coolwarm',
            axis=None
        ).set_caption("Figure 6 - Scores").to_excel(writer)

if __name__ == "__main__":
    main()

# -
pip install imgkit


# +
import click
import pandas as pd
import imgkit

def process_data(file_path):
    cross_val_results = pd.read_csv(file_path, header=[0, 1], index_col=0)
    Fig_6 = cross_val_results.xs('mean', axis='columns', level=1)
    return Fig_6

@click.command(context_settings={'ignore_unknown_options': True})
@click.option('--file-path', default='../data/processed/cv_results.csv', help='Path to the cv_results.csv file.')
@click.option('--output-path', default='../img/Fig_6', help='Output base path for the files.')
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def main(file_path, output_path, args):
    Fig_6 = process_data(file_path)

    # Save as PNG
    png_path = f"{output_path}.png"
    styled_html = Fig_6.style.format(precision=2).background_gradient(
        cmap='coolwarm', axis=None
    ).set_caption("Figure 6 - Scores").to_html()  # Changed from .render() to .to_html()
    
    imgkit.from_string(styled_html, png_path)

if __name__ == "__main__":
    main()

# -




