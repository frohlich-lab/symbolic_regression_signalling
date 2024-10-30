import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_log_mse_loss(formula, dataset):
    x = sp.symbols('x')
    loss = 0
    for x_val, y_val in dataset:
        predicted_value = formula.subs(x, x_val)
        # Apply log transformation to both predicted and actual values
        log_predicted_value = np.log(predicted_value) if predicted_value > 0 else 0
        log_actual_value = np.log(y_val) if y_val > 0 else 0
        loss += (log_predicted_value - log_actual_value) ** 2
    return loss / len(dataset)

def calculate_complexity(formula):
    # Complexity is defined by counting the number of operators and operands in the formula
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(formulas, dataset, output_file):
    complexities = []
    losses = []

    for formula in formulas:
        complexity = calculate_complexity(formula)
        loss = compute_log_mse_loss(formula, dataset)
        complexities.append(complexity)
        losses.append(loss)

    # Create the scatter plot
    plt.figure()
    plt.scatter(complexities, losses)
    plt.xlabel('Formula Complexity')
    plt.ylabel('Log-Space Loss (MSE)')
    plt.title('Scatter Plot of Log-Space MSE Loss vs. Formula Complexity')
    plt.grid(True)
    plt.savefig(output_file)  # Save the plot to the specified output file
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Log-Space MSE Loss against Formula Complexity')
    parser.add_argument('--formulas', nargs='+', required=True, help='List of formulas in quotes, e.g., "x**2 + 2*x + 1" "x**3 - x + 2"')
    parser.add_argument('--dataset', nargs='+', required=True, help='List of dataset points in the form x:y, e.g., 1:2 2:4 3:9')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')

    args = parser.parse_args()

    # Convert formula strings to sympy expressions
    formulas = [sp.sympify(formula) for formula in args.formulas]

    # Convert dataset points from strings to tuples of floats
    dataset = [(float(point.split(':')[0]), float(point.split(':')[1])) for point in args.dataset]

    scatter_plot_formulas(formulas, dataset, args.output)

if __name__ == '__main__':
    main()