import argparse
import pandas as pd

def dataframe_to_latex(df):
    # Identifiziere die besten Werte pro Spalte (außer "Synthesizer")
    highlight_columns = df.columns[1:]  # Ignoriere die erste Spalte
    best_values = {col: df[col].max() for col in highlight_columns if pd.api.types.is_numeric_dtype(df[col])}

    # Baue den LaTeX-Code
    latex_table = "\\begin{table}[ht!]\n\\centering\n\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += " & ".join(df.columns) + " \\\\\n\\midrule\n"

    for _, row in df.iterrows():
        row_cells = []
        for col in df.columns:
            value = row[col]
            if col in best_values and pd.api.types.is_numeric_dtype(df[col]) and float(value) == best_values[col]:
                value = f"{value:.2f}"
                row_cells.append(f"\\cellcolor{{gray!25}}{{{value}}}")
            elif pd.api.types.is_numeric_dtype(df[col]):
                value = f"{value:.2f}"
                row_cells.append(str(value))
            else:
                row_cells.append(str(value))
        latex_table += " & ".join(row_cells) + " \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{}\n\\end{table}"
    return latex_table

if __name__ == "__main__":

    # defines parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Select dataset csv to convert to latex table")
    args = parser.parse_args()

    # read as dataframe and convert to latex code
    df = pd.read_csv(args.data_path, index_col=1)
    df.drop(columns='F1_real', inplace=True)
    df.sort_values(by='ups', ascending=False, inplace=True)
    #df['# best'] = df['# best'].astype(str)
    print(df)
    latex_table = dataframe_to_latex(df)
    print(latex_table)