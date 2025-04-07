import numpy as np
import ast  # For safely evaluating string representations of lists

def parse_params(file_path):
    params = []
    with open(file_path, 'r') as f:
        # Read the entire file and attempt to evaluate the list
        data = f.read()
        
        try:
            # Convert the entire string representation into a list of lists
            params = ast.literal_eval(data)
        except Exception as e:
            print(f"Error parsing the file: {e}")
    
    return np.array(params)

# Load and parse parameters
params = parse_params('emcee_control_purple_Kfixed1.dat')

# Save to CSV
def save_to_csv(params):
    np.savetxt('format_purple_Kfixed1.csv', params, delimiter=',',
               header='lamb, beta, k, delta, p, c, K', comments='')
    print("Data saved ")

save_to_csv(params)
