import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import time
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Levelling functions
def read_input_from_text_levelling(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.strip().split()
            if len(parts) == 6 and parts[0] == 'HDIF' and parts[5] == 'm':
                start_point = parts[1]
                end_point = parts[2]
                elevation = float(parts[3])
                sd = float(parts[4])
                data.append([start_point, end_point, elevation, sd])

    return data

def read_control_points_levelling(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_control_points = int(lines[0].split(':')[1].strip())
    names = lines[1].split(':')[1].strip().split(',')
    elevations = list(map(float, lines[2].split(':')[1].strip().split(',')))

    control_points = dict(zip(names, elevations))
    return control_points

def main_levelling(config_folder):
    start_time = time.time()
    timestamp = get_timestamp()

    # Reading input text files
    hpl_folder = os.path.join(config_folder, "HPL")
    hpl_input_folder = os.path.join(hpl_folder, "HPL_input_folder")
    
    if not os.path.exists(hpl_input_folder):
        print(f"Error: HPL_input_folder not found in {hpl_folder}")
        return

    # Read all .txt files in the HPL_input_folder
    data_list = []
    for filename in os.listdir(hpl_input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(hpl_input_folder, filename)
            data = read_input_from_text_levelling(file_path)
            data_list.extend(data)

    if not data_list:
        print("Error: No valid input files found in HPL_input_folder")
        return

    # Reading control points from HPL_input.txt
    control_points_file_path = os.path.join(hpl_folder, "HPL_input.txt")
    if not os.path.exists(control_points_file_path):
        print(f"Error: HPL_input.txt not found in {hpl_folder}")
        return
    
    results_folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "HPL_results")
    os.makedirs(results_folder_path, exist_ok=True)

    control_points = read_control_points_levelling(control_points_file_path)

    filtered_data_list = []
    removed_observations = 0
    for start, end, elev, sd in data_list:
        if not (start in control_points and end in control_points):
            filtered_data_list.append((start, end, elev, sd))
        else:
            removed_observations += 1

    # Prepare observations and weights
    observations = [(start, end, elev) for start, end, elev, _ in filtered_data_list]
    weights = [1 / (sd ** 2) if sd > 0 else 0 for _, _, _, sd in filtered_data_list]

    # Identify junction points
    all_points = set()
    for start, end, _ in observations:
        all_points.add(start)
        all_points.add(end)
    junction_points = list(all_points - set(control_points.keys()))

    # Calculate matrices
    A, L, B, RHS, W, std_devs, observation_numbers = calculate_matrices_levelling(junction_points, control_points, observations, weights)

    # Normal matrix
    N = A.T @ W @ A
    U = A.T @ W @ RHS
    
    # Adjusted heights
    try:
        X_hat = np.linalg.inv(N) @ U
        adjusted_elevations = X_hat
    except np.linalg.LinAlgError:
        print("Error: Unable to invert the normal matrix. The system might be singular.")
        return
    
    # Residuals
    V = A @ X_hat - RHS
    
    # VtWV
    VtWV = V.T @ W @ V
    
    # Reference standard deviation
    degrees_of_freedom = len(V) - len(X_hat) - removed_observations
    if degrees_of_freedom <= 0:
        print(f"Error: Degrees of freedom ({degrees_of_freedom}) is zero or negative. Check your input data.")
        return
    if VtWV[0, 0] < 0:
        print(f"Error: VtWV ({VtWV[0, 0]}) is negative. Check your input data and calculations.")
        return
    SnoT = np.sqrt(VtWV[0, 0] / degrees_of_freedom)
    
    # Variance-covariance matrix of adjusted elevations
    try:
        N_inv = np.linalg.inv(N)
        var_cov_matrix = SnoT ** 2 * N_inv
    except np.linalg.LinAlgError:
        print("Error: Unable to invert the normal matrix for variance-covariance calculation.")
        return
    
    # Sigma_ll matrix row-wise calculation
    A_N_inv_A_T = A @ N_inv @ A.T
    Sigma_ll = SnoT * np.sqrt(np.diag(A_N_inv_A_T))
    Sigma_ll = Sigma_ll.reshape(-1, 1)
    

    # Standard errors for observations
    Sz = SnoT * np.sqrt(np.diag(N_inv)) 
    
    # Chi-square test
    alpha = 0.05
    chi2_upper = (stats.chi2.ppf(1 - alpha / 2, degrees_of_freedom)) / (degrees_of_freedom)
    chi2_lower = (stats.chi2.ppf(alpha / 2, degrees_of_freedom)) / (degrees_of_freedom)
    chi2_test_value = 1  # The test value is 1 for the chi-square test
    chi2_result = "PASSED" if chi2_lower <= SnoT**2/chi2_test_value <= chi2_upper else "FAILED"

    ##################################
    alpha_0 = 0.01  # You may adjust this value
    tau_critical = stats.t.ppf(1 - alpha_0, degrees_of_freedom)
    tau_values = np.abs(V.flatten()) / (SnoT * np.sqrt(np.diag(A @ N_inv @ A.T)))

    outliers = []
    for i, tau in enumerate(tau_values):
        if tau > tau_critical:
            start, end, _ = observations[i]
            outliers.append((i+1, start, end, tau))

    if outliers:
        print("Potential outliers detected:")
        for obs_num, start, end, tau in outliers:
            print(f"Observation {obs_num} ({start} - {end}): tau = {tau:.4f}")
    else:
        print("No outliers detected")
    ########################################
    
    # Trace of variance-covariance matrix
    trace_value = np.trace(var_cov_matrix)
    
    # Normalized residuals
    normalized_residuals = (V.flatten())
    normalized_residuals = np.abs(normalized_residuals)
    normalized_residuals = np.sort(normalized_residuals)[::-1]  # Sort in descending order

    end_time = time.time()
    duration = end_time - start_time

    max_sigma_ll = np.max(Sigma_ll)
    max_sigma_ll_obs = np.argmax(Sigma_ll) + 1
    max_sigma_ll_std_dev = std_devs[max_sigma_ll_obs - 1]

    min_sigma_ll = np.min(Sigma_ll)
    min_sigma_ll_obs = np.argmin(Sigma_ll) + 1
    min_sigma_ll_std_dev = std_devs[min_sigma_ll_obs - 1]

    results_folder_path = os.path.join(os.path.expanduser("~"), "Desktop", "HPL_results")
    os.makedirs(results_folder_path, exist_ok=True)

    
    summary = f"""Levelling Summary:
Timestamp: {timestamp}
Time Duration: {duration:.2f} seconds
Input Folder: {hpl_input_folder}
Number of Junction Points: {len(junction_points)}
Number of Control Points: {len(control_points)}
Degrees of Freedom: {degrees_of_freedom}
Shape of A Matrix: {A.shape}
Chi-square Limits: [{chi2_lower:.4f}, {chi2_upper:.4f}]
Chi-square Test Result: {chi2_result}
Value of SnoT: {SnoT:.6f}
Highest Sigma_ll: {max_sigma_ll:.10f} (Observation {max_sigma_ll_obs}, Std Dev: {max_sigma_ll_std_dev:.10f})
Lowest Sigma_ll: {min_sigma_ll:.10f} (Observation {min_sigma_ll_obs}, Std Dev: {min_sigma_ll_std_dev:.10f})
"""
    summary += f"""
Local Test Results:
Critical value (tau): {tau_critical:.6f}
Potential outliers detected:
"""
    
    if outliers:
        for obs_num, start, end, tau in outliers:
            summary += f"Observation {obs_num} ({start} - {end}): tau = {tau:.4f}\n"
    else:
        summary += "No outliers detected\n"

    
    with open(os.path.join(results_folder_path, 'levelling_summary.txt'), 'w') as f:
        f.write(summary)

    summary += f"Number of removed observations (connecting only control points): {removed_observations}\n"

    print(f"Levelling results and summary have been saved to: {results_folder_path}")

    # Save results
    save_results_to_folder_levelling(results_folder_path, A, L, B, RHS, W, observations, junction_points, std_devs, adjusted_elevations, V, weights, control_points, VtWV, SnoT, N_inv, var_cov_matrix, Sigma_ll, Sz, (chi2_lower, chi2_upper), chi2_result, trace_value, normalized_residuals,observation_numbers, tau_values, tau_critical, outliers,removed_observations)

    print(f"Levelling results have been saved to: {results_folder_path}")

def calculate_matrices_levelling(junction_points, control_points, observations, weights):
    num_junctions = len(junction_points)
    num_observations = len(observations)
    
    A = np.zeros((num_observations, num_junctions))
    L = np.zeros((num_observations, 1))
    B = np.zeros((num_observations, 1))
    
    for i, (start, end, elev_diff) in enumerate(observations):
        if start in junction_points:
            A[i, junction_points.index(start)] = -1
        if end in junction_points:
            A[i, junction_points.index(end)] = 1
        if start in control_points:
            B[i] -= control_points[start]
        if end in control_points:
            B[i] += control_points[end]
        L[i] = elev_diff
    
    RHS = L - B
    W = np.diag(weights)
    std_devs = [1 / np.sqrt(w) if w != 0 else '-' for w in weights]
    
    observation_numbers = list(range(1, num_observations + 1))
    return A, L, B, RHS, W, std_devs, observation_numbers

def save_results_to_folder_levelling(results_folder_path, A, L, B, RHS, W, observations, junction_points, std_devs, adjusted_elevations, residuals, weights, control_points, VtWV, SnoT, N_inv, var_cov_matrix, Sigma_ll, Sz, chi2_interval, chi2_result, trace_value, normalized_residuals, observation_numbers, tau_values, tau_critical, outliers, removed_observations):
    os.makedirs(results_folder_path, exist_ok=True)
    
    # First file: Design Matrix (A), Observation Matrix (L), Benchmark Matrix (B), Right-Hand Side Matrix (RHS = L - B), Weight Matrix (W), Observation Equations
    with open(os.path.join(results_folder_path, 'results_part1.txt'), 'w') as file:
        file.write("Design Matrix (A):\n")
        np.savetxt(file, A, fmt='%.4f')
        
        file.write("\nObservation Matrix (L):\n")
        np.savetxt(file, L, fmt='%.4f')
        
        file.write("\nBenchmark Matrix (B):\n")
        np.savetxt(file, B, fmt='%.4f')
        
        file.write("\nRight-Hand Side Matrix (RHS = L - B):\n")
        np.savetxt(file, RHS, fmt='%.4f')
        
        file.write("\nWeight Matrix (W):\n")
        np.savetxt(file, W, fmt='%.4f')
        
        file.write("\nObservation Equations:\n")
        for i, (start, end, elev_diff) in enumerate(observations):
            start_str = control_points.get(start, start)
            end_str = control_points.get(end, end)
            std_dev_str = f"{std_devs[i]:.4f}" if isinstance(std_devs[i], float) else std_devs[i]
            eq = f"{end_str} - {start_str} = {elev_diff} ± {std_dev_str} + v{i+1}"
            file.write(f"{i+1}. {eq}\n")
    
    # Second file: Adjusted Coordinates of Junction Points, Residuals (A*X - RHS)
    with open(os.path.join(results_folder_path, 'results_part2.txt'), 'w') as file:
        file.write("Adjusted Elevations of Junction Points:\n")
        for i, point in enumerate(junction_points):
            file.write(f"{point} = {adjusted_elevations[i, 0]:.4f}\n")
        
        file.write("\nResiduals (A*X - RHS):\n")
        for i, res in enumerate(residuals):
            file.write(f"Observation {i+1}: {res[0]:.10f}\n")
    
    # Third file: V^T * W * V, Reference Standard Deviation (SnoT), (A^T * W * A)^-1 (N^-1), Variance-Covariance Matrix, Standard Deviations of Adjusted Coordinates, Sigma_ll Matrix (A * (SnoT)^2 * N^-1 * A^T), Chi-square test interval, result of the chi square test , Trace of variance-covariance matrix and Normalized Residuals, (in descending order)
    with open(os.path.join(results_folder_path, 'results_part3.txt'), 'w') as file:
        file.write(f"\nNumber of removed observations (connecting only control points): {removed_observations}\n")
        file.write(f"V^T * W * V: {VtWV[0, 0]}\n")
        file.write(f"\nReference Standard Deviation (SnoT): {SnoT}\n")
        
        file.write("\n(A^T * W * A)^-1 (N^-1):\n")
        np.savetxt(file, N_inv, fmt='%.4f')
        
        file.write("\nVariance-Covariance Matrix:\n")
        np.savetxt(file, var_cov_matrix, fmt='%.10f')
        
        file.write("\nStandard Errors (Sz) for Junction Points:\n")
        for i, point in enumerate(junction_points):
            file.write(f"{point}: {Sz[i]:.10f}\n")
        
        file.write("\nChi-square test interval:\n")
        file.write(f"Lower Bound: {chi2_interval[0]}\n")
        file.write(f"Upper Bound: {chi2_interval[1]}\n")
        
        file.write(f"\nResult of the Chi-square test: {chi2_result}\n")


        file.write("\nTau Values for Each Observation:\n")
        for i, tau in enumerate(tau_values):
            start, end, _ = observations[i]
            file.write(f"Observation {i+1} ({start} - {end}): tau = {tau:.6f}\n")

        file.write(f"\nCritical tau value: {tau_critical:.6f}\n")

        file.write("\nPotential Outliers:\n")
        if outliers:
            for obs_num, start, end, tau in outliers:
                file.write(f"Observation {obs_num} ({start} - {end}): tau = {tau:.4f}\n")
        else:
            file.write("No outliers detected\n")

        file.write(f"\nTrace of Variance-Covariance Matrix: {trace_value}\n")
        
        file.write("\nNormalized Residuals (in descending order):\n")
        for i, norm_res in enumerate(normalized_residuals):
            file.write(f"{i+1}: {norm_res:.10f}\n")
        
        file.write("\nSigma_ll Matrix with Observation Numbers and Standard Deviations:\n")
        sigma_ll_with_obs = list(zip(observation_numbers, Sigma_ll.flatten(), std_devs))
        sigma_ll_sorted = sorted(sigma_ll_with_obs, key=lambda x: x[1], reverse=True)
        for obs_num, sigma_ll, std_dev in sigma_ll_sorted:
            std_dev_str = f"{std_dev:.4f}" if isinstance(std_dev, float) else std_dev
            file.write(f"Observation {obs_num}: Sigma_ll = {sigma_ll:.10f}, Standard Deviation = {std_dev_str}\n")









# GPS functions
def read_input_from_text_gps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('DXYZ'):
            parts = lines[i].split()
            start_point = parts[1]
            end_point = parts[2]
            # Read the delta values, ignoring "DXYZ" at the start and "m ..." at the end
            delta = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
            
            cov_matrix = np.zeros((3, 3))
            cov_values = []
            for j in range(1, 4):
                # Remove dots and split, then convert to float
                cov_values.extend([float(val) for val in lines[i+j].replace('...', '').split()])
            
            cov_matrix[0, 0] = cov_values[0]
            cov_matrix[0, 1] = cov_matrix[1, 0] = cov_values[1]
            cov_matrix[0, 2] = cov_matrix[2, 0] = cov_values[2]
            cov_matrix[1, 1] = cov_values[3]
            cov_matrix[1, 2] = cov_matrix[2, 1] = cov_values[4]
            cov_matrix[2, 2] = cov_values[5]
            
            weight_matrix = np.linalg.inv(cov_matrix)
            
            data.append((start_point, end_point, delta, weight_matrix))
            i += 4
        else:
            i += 1

    return data

def read_control_points_from_text_gps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    control_points = {}
    i = 0
    while i < len(lines):
        if 'Enter the name of the control point' in lines[i]:
            name = lines[i].split(':')[1].strip()
            x = float(lines[i+1].split(':')[1].strip())
            y = float(lines[i+2].split(':')[1].strip())
            z = float(lines[i+3].split(':')[1].strip())
            control_points[name] = np.array([x, y, z])
            i += 4
        else:
            i += 1

    return control_points
def calculate_matrices_gps(junction_points, control_points, observations):
    num_junctions = len(junction_points) * 3
    num_observations = len(observations) * 3
    
    A = np.zeros((num_observations, num_junctions))
    L = np.zeros((num_observations, 1))
    B = np.zeros((num_observations, 1))
    W = np.zeros((num_observations, num_observations))
    
    for i, (start, end, delta, weight) in enumerate(observations):
        row = i * 3
        if start in junction_points:
            col = junction_points.index(start) * 3
            A[row:row+3, col:col+3] = -np.eye(3)
        if end in junction_points:
            col = junction_points.index(end) * 3
            A[row:row+3, col:col+3] = np.eye(3)
        if start in control_points:
            B[row:row+3, 0] -= control_points[start]
        if end in control_points:
            B[row:row+3, 0] += control_points[end]
        L[row:row+3, 0] = delta
        W[row:row+3, row:row+3] = weight
    
    RHS = L - B
    
    return A, L, B, RHS, W

def save_to_file_gps(folder, filename, content):
    with open(os.path.join(folder, filename), 'w') as file:
        file.write(content)


def iterate_gps_analysis(config_folder, initial_observations, control_points, alpha_values):
    all_results = []
    observations = initial_observations.copy()
    
    folder_name = "CORS_Results"
    os.makedirs(folder_name, exist_ok=True)
    
    matrices_file = open(os.path.join(folder_name, "matrices_and_equations.txt"), 'w')
    coordinates_file = open(os.path.join(folder_name, "adjusted_coordinates_and_residuals.txt"), 'w')
    stats_file = open(os.path.join(folder_name, "statistical_results.txt"), 'w')
    summary_file = open(os.path.join(folder_name, "summary.txt"), 'w')
    
    for iteration, alpha in enumerate(alpha_values, 1):
        print(f"\nIteration {iteration} (alpha = {alpha}):")
        matrices_file.write(f"\n\n--- Iteration {iteration} (alpha = {alpha}) ---\n\n")
        coordinates_file.write(f"\n\n--- Iteration {iteration} (alpha = {alpha}) ---\n\n")
        stats_file.write(f"\n\n--- Iteration {iteration} (alpha = {alpha}) ---\n\n")
        summary_file.write(f"\n\n--- Iteration {iteration} (alpha = {alpha}) ---\n\n")
        
        all_points = set()
        for start, end, _, _ in observations:
            all_points.add(start)
            all_points.add(end)
        
        junction_points = list(all_points - set(control_points.keys()))

        filtered_observations = []
        removed_observations = 0
        for start, end, delta, weight in observations:
            if not (start in control_points and end in control_points):
                filtered_observations.append((start, end, delta, weight))
            else:
                removed_observations += 1
        
        A, L, B, RHS, W = calculate_matrices_gps(junction_points, control_points, filtered_observations)
        
        matrices_file.write(f"Design Matrix (A):\n{A}\n")
        matrices_file.write(f"\nObservation Matrix (L):\n{L}\n")
        matrices_file.write(f"\nBenchmark Matrix (B):\n{B}\n")
        matrices_file.write(f"\nRight-Hand Side Matrix (RHS = L - B):\n{RHS}\n")
        matrices_file.write(f"\nWeight Matrix (W):\n{W}\n")
        
        matrices_file.write("\nObservation Equations:\n")
        for i, (start, end, delta, _) in enumerate(filtered_observations):
            start_str = control_points.get(start, start)
            end_str = control_points.get(end, end)
            eq = f"{end_str} - {start_str} = [{delta[0]:.8f}, {delta[1]:.8f}, {delta[2]:.8f}] + [v{3*i+1}, v{3*i+2}, v{3*i+3}]"
            matrices_file.write(f"{i+1}. {eq}\n")
        
        try:
            AtWA = A.T @ W @ A
            X = np.linalg.inv(AtWA) @ (A.T @ W @ RHS)
            Residuals = A @ X - RHS
            VtWV = Residuals.T @ W @ Residuals
            degrees_of_freedom = len(Residuals) - len(X)
            SnoT = np.sqrt(VtWV[0, 0] / degrees_of_freedom) 
            N_inv = np.linalg.inv(A.T @ W @ A)
            var_cov_matrix = SnoT**2 * N_inv

            coordinates_file.write("Adjusted Coordinates of Junction Points:\n")
            for i, point in enumerate(junction_points):
                coordinates_file.write(f"{point} = [{X[3*i, 0]:.8f}, {X[3*i+1, 0]:.8f}, {X[3*i+2, 0]:.8f}]\n")

            coordinates_file.write("\nResiduals (A*X - RHS):\n")
            for i in range(len(Residuals) // 3):
                coordinates_file.write(f"Observation {i+1}: [{Residuals[3*i, 0]:.10f}, {Residuals[3*i+1, 0]:.10f}, {Residuals[3*i+2, 0]:.10f}]\n")

            Sigma_ll = np.diagonal(A @ (SnoT**2 * N_inv) @ A.T).reshape(-1, 3)

            alpha_chi = 0.05 

            chi2_upper = (stats.chi2.ppf(1 - alpha_chi / 2, degrees_of_freedom)) / (degrees_of_freedom)
            chi2_lower = (stats.chi2.ppf(alpha_chi / 2, degrees_of_freedom)) / (degrees_of_freedom)

            tau_critical = stats.t.ppf(1 - alpha, degrees_of_freedom)
            tau_values = np.abs(Residuals.flatten()) / (SnoT * np.sqrt(np.diag(A @ N_inv @ A.T)))

            outliers = []
            tau_results = []
            for i in range(len(tau_values) // 3):
                tau_x = tau_values[3*i]
                tau_y = tau_values[3*i + 1]
                tau_z = tau_values[3*i + 2]
                
                is_outlier_x = tau_x > tau_critical
                is_outlier_y = tau_y > tau_critical
                is_outlier_z = tau_z > tau_critical
                
                start, end, _, _ = filtered_observations[i]
                tau_results.append((i+1, start, end, tau_x, is_outlier_x, tau_y, is_outlier_y, tau_z, is_outlier_z))
                
                if is_outlier_x or is_outlier_y or is_outlier_z:
                    outliers.append((i+1, start, end, max(tau_x, tau_y, tau_z)))

            stats_file.write(f"Number of observations: {len(filtered_observations)}\n")
            stats_file.write(f"Number of junction points: {len(junction_points)}\n")
            stats_file.write(f"Degrees of freedom: {degrees_of_freedom}\n")
            stats_file.write(f"V^T * W * V: {VtWV[0, 0]}\n")
            stats_file.write(f"\nReference Standard Deviation (SnoT): {SnoT}\n")
            stats_file.write(f"\n(A^T * W * A)^-1 (N^-1):\n{N_inv}\n")
            stats_file.write(f"\nVariance-Covariance Matrix:\n{var_cov_matrix}\n")

            stats_file.write("\nStandard Deviations of Adjusted Coordinates:\n")
            for i, point in enumerate(junction_points):
                Sx = np.sqrt(var_cov_matrix[3*i, 3*i])
                Sy = np.sqrt(var_cov_matrix[3*i+1, 3*i+1])
                Sz = np.sqrt(var_cov_matrix[3*i+2, 3*i+2])
                stats_file.write(f"{point}: Sx = {Sx:.10f}, Sy = {Sy:.10f}, Sz = {Sz:.10f}\n")

            stats_file.write("\nSigma_ll Matrix (A * (SnoT)^2 * N^-1 * A^T):\n")
            for i, sigma in enumerate(Sigma_ll):
                stats_file.write(f"Observation {i+1}: [{sigma[0]:.10f}, {sigma[1]:.10f}, {sigma[2]:.10f}]\n")

            stats_file.write(f"Chi-square test interval: [{chi2_lower:.4f}, {chi2_upper:.4f}]\n")
        
            test_value = 1
            if chi2_lower < SnoT**2 / test_value < chi2_upper:
                stats_file.write("Global test passed\n")
            else:
                stats_file.write("Global test failed\n")

            stats_file.write("\nLocal Test Results:\n")
            stats_file.write(f"Critical value (tau): {tau_critical:.6f}\n")
            stats_file.write("\nTau Values and Outlier Detection for Each Observation:\n")
            for obs_num, start, end, tau_x, is_outlier_x, tau_y, is_outlier_y, tau_z, is_outlier_z in tau_results:
                stats_file.write(f"Observation {obs_num} ({start} - {end}):\n")
                stats_file.write(f"  X: tau = {tau_x:.6f}, Outlier: {is_outlier_x}\n")
                stats_file.write(f"  Y: tau = {tau_y:.6f}, Outlier: {is_outlier_y}\n")
                stats_file.write(f"  Z: tau = {tau_z:.6f}, Outlier: {is_outlier_z}\n")

            stats_file.write("\nPotential Outliers:\n")
            if outliers:
                for obs_num, start, end, tau in outliers:
                    stats_file.write(f"Observation {obs_num} ({start} - {end}): max tau = {tau:.4f}\n")
            else:
                stats_file.write("No outliers detected\n")
            
            trace_value = np.trace(var_cov_matrix)
            stats_file.write(f"Trace of variance-covariance matrix: {trace_value}\n")

            normalized_residuals = [np.linalg.norm(Residuals[3*i:3*i+3]) / SnoT for i in range(len(Residuals) // 3)]
            sorted_residuals = sorted(enumerate(normalized_residuals), key=lambda x: x[1], reverse=True)
            stats_file.write("\nNormalized Residuals (in descending order):\n")
            for i, (obs_index, norm_res) in enumerate(sorted_residuals):
                stats_file.write(f"{i+1}: Observation {obs_index+1} - {norm_res:.10f}\n")

            summary_file.write(f"Number of observations: {len(filtered_observations)}\n")
            summary_file.write(f"Number of junction points: {len(junction_points)}\n")
            summary_file.write(f"Degrees of freedom: {degrees_of_freedom}\n")
            summary_file.write(f"Reference Standard Deviation (SnoT): {SnoT:.6f}\n")
            summary_file.write(f"Chi-square test interval: [{chi2_lower:.4f}, {chi2_upper:.4f}]\n")
            if chi2_lower < SnoT**2 < chi2_upper:
                summary_file.write("Global test: Passed\n")
            else:
                summary_file.write("Global test: Failed\n")
            summary_file.write(f"Number of potential outliers: {len(outliers)}\n")
            summary_file.write(f"Trace of variance-covariance matrix: {trace_value:.6f}\n\n")

            all_results.append({
                'iteration': iteration,
                'alpha': alpha,
                'outliers': len(outliers),  # Just store the count, not the details
                'SnoT': SnoT,
                'degrees_of_freedom': degrees_of_freedom,
                'chi2_lower': chi2_lower,
                'chi2_upper': chi2_upper,
                'tau_critical': tau_critical
            })
            # Modified part to check for control points before removing outliers
            observations_to_remove = []
            for obs_num, start, end, tau in outliers:
                if start not in control_points and end not in control_points:
                    observations_to_remove.append(obs_num)
                # else:
                #     print(f"Observation {obs_num} ({start} - {end}) is an outlier but contains a control point. It will not be removed.")
                #     stats_file.write(f"Observation {obs_num} ({start} - {end}) is an outlier but contains a control point. It will not be removed.\n")

            # Remove outliers for the next iteration, but keep those with control points
            observations = [obs for i, obs in enumerate(filtered_observations) if i+1 not in observations_to_remove]
            
        except Exception as e:
            print(f"An error occurred during computation in iteration {iteration}: {str(e)}")
            stats_file.write(f"An error occurred during computation: {str(e)}\n")
            summary_file.write(f"An error occurred during computation: {str(e)}\n")
            stats_file.write("Some values could not be calculated due to insufficient input or computational errors.\n")
            summary_file.write("Some values could not be calculated due to insufficient input or computational errors.\n")
    
    matrices_file.close()
    coordinates_file.close()
    stats_file.close()
    summary_file.close()
    
    return all_results 


def create_summary(config_folder, junction_points, control_points, degrees_of_freedom, A, SnoT, N_inv, var_cov_matrix, Sigma_ll, W, chi2_lower, chi2_upper, tau_critical, tau_results, outliers, removed_observations):
    timestamp = get_timestamp()
    
    max_sigma_ll = np.max(Sigma_ll)
    max_sigma_ll_obs = np.unravel_index(np.argmax(Sigma_ll), Sigma_ll.shape)[0] + 1
    max_sigma_ll_std_dev = 1 / np.sqrt(W[3*(max_sigma_ll_obs-1), 3*(max_sigma_ll_obs-1)])

    min_sigma_ll = np.min(Sigma_ll)
    min_sigma_ll_obs = np.unravel_index(np.argmin(Sigma_ll), Sigma_ll.shape)[0] + 1
    min_sigma_ll_std_dev = 1 / np.sqrt(W[3*(min_sigma_ll_obs-1), 3*(min_sigma_ll_obs-1)])
    
    chi2_result = "PASSED" if chi2_lower <= SnoT**2 <= chi2_upper else "FAILED"
    
    summary = f"""GPS Summary:
Timestamp: {timestamp}
Input Folder: {config_folder}
Number of Junction Points: {len(junction_points)}
Number of Control Points: {len(control_points)}
Degrees of Freedom: {degrees_of_freedom}
Shape of A Matrix: {A.shape}
Chi-square Limits: [{chi2_lower:.4f}, {chi2_upper:.4f}]
Chi-square Test Result: {chi2_result}
Value of SnoT: {SnoT:.6f}
Highest Sigma_ll: {max_sigma_ll:.10f} (Observation {max_sigma_ll_obs}, Std Dev: {max_sigma_ll_std_dev:.10f})
Lowest Sigma_ll: {min_sigma_ll:.10f} (Observation {min_sigma_ll_obs}, Std Dev: {min_sigma_ll_std_dev:.10f})
Local Test Results:
Critical value (tau): {tau_critical:.6f}
Tau Values and Outlier Detection:
"""
    for obs_num, start, end, tau_x, is_outlier_x, tau_y, is_outlier_y, tau_z, is_outlier_z in tau_results:
        summary += f"Observation {obs_num} ({start} - {end}):\n"
        summary += f"  X: tau = {tau_x:.6f}, Outlier: {is_outlier_x}\n"
        summary += f"  Y: tau = {tau_y:.6f}, Outlier: {is_outlier_y}\n"
        summary += f"  Z: tau = {tau_z:.6f}, Outlier: {is_outlier_z}\n"

    summary += "Potential outliers detected:\n"
    if outliers:
        for obs_num, start, end, tau in outliers:
            summary += f"Observation {obs_num} ({start} - {end}): max tau = {tau:.4f}\n"
    else:
        summary += "No outliers detected\n"
   
    summary += f"Number of removed observations (connecting only control points): {removed_observations}\n"
    
    return summary

def main_gps(config_folder):
    start_time = time.time()
    timestamp = get_timestamp()

    cors_folder = os.path.join(config_folder, "CORS")
    gps_file_path = os.path.join(cors_folder, "CORS.txt")
    control_points_file_path = os.path.join(cors_folder, "control_points.txt")

    try:
        observations = read_input_from_text_gps(gps_file_path)
        control_points = read_control_points_from_text_gps(control_points_file_path)
    except Exception as e:
        print(f"Error reading files: {str(e)}")
        return

    alpha_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    all_results = iterate_gps_analysis(config_folder, observations, control_points, alpha_values)

    # Print comparison of results
    print("\nComparison of Results:")
    print("{:<15} {:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15.6}".format(
        "Iteration", "Alpha", "Outliers", "SnoT", "DoF", "Chi2 Lower", "Chi2 Upper","Tau Critical"))
    print("-" * 100)

    for result in all_results:
        print("{:<15} {:<10} {:<15} {:<15.6f} {:<15} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            result['iteration'], result['alpha'], result['outliers'],
            result['SnoT'], result['degrees_of_freedom'],
            result['chi2_lower'], result['chi2_upper'], result['tau_critical']))
        


    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal processing time: {duration:.2f} seconds")


def main():
    print("Welcome to the Geodetic Data Processor!")
    print("This program can process both Levelling and GPS data.")
    
    while True:
        choice = input("Enter '1' for Levelling or '2' for GPS processing: ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid input. Please enter '1' or '2'.")

    config_folder = input("Enter the path to the Config folder: ").strip()
    
    if not os.path.exists(config_folder):
        print(f"Error: The folder at path '{config_folder}' does not exist.")
        return

    if choice == '1':
        print("Processing Levelling data...")
        main_levelling(config_folder)
    else:
        print("Processing GPS data...")
        main_gps(config_folder)

if __name__ == "__main__":
    main()
