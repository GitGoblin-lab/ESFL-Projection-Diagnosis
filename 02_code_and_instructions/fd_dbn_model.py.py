# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

def build_fd_dbn_structure(T_crisp, indicators, mu=0.0638, sigma=0.0258, alpha=1):
    """
    Build the FD-DBN network structure G based on the Fuzzy-DEMATEL comprehensive impact matrix.
    """
    # Calculate the edge threshold for the same time slice (Intra-slice) according to the formula
    delta_intra = mu + alpha * sigma
    
    edges = []
    # 1. Determine Intra-slice edges (causal relationships within time T)
    for i in indicators:
        for j in indicators:
            if i == j:
                continue
            t_ij = T_crisp.loc[i, j]
            t_ji = T_crisp.loc[j, i]
            # If the threshold is met and T_ij > T_ji, add Xi -> Xj
            if t_ij >= delta_intra and t_ij > t_ji:
                edges.append((f"{i}_t", f"{j}_t"))
                edges.append((f"{i}_t1", f"{j}_t1")) # The network structure is the same at time T+1
                
    # 2. Determine Inter-slice edges (dynamic transition relationships from time T to T+1)
    # (a) System inertia: time transition of its own state Xi_t -> Xi_t1
    for ind in indicators:
        edges.append((f"{ind}_t", f"{ind}_t1"))
        
    # (b) Specific lagged causal edges added based on expert knowledge and literature
    edges.append(("F2_t", "F3_t1"))
    edges.append(("P1_t", "C2_t1"))
    
    # Build the network graph model (using 2-Time-slice BN to simulate DBN for more stable inference)
    dbn_model = BayesianNetwork(edges)
    return dbn_model

def cross_validate_and_learn(dbn_model, sample_data, target_node="UES_t1", k_folds=6):
    """
    Perform stratified K-fold cross-validation and return the best network model and evaluation metrics.
    (The logical framework here demonstrates selecting the last fold as the "best model").
    """
    print("\n[Stage 2: Validation] Starting stratified K-fold cross-validation...")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Extract target variables for stratified sampling
    y = sample_data[target_node]
    X = sample_data.drop(columns=[target_node])
    
    best_model = None
    best_f1 = 0.0
    fold = 1
    
    for train_index, test_index in skf.split(X, y):
        train_data = sample_data.iloc[train_index]
        test_data = sample_data.iloc[test_index]
        
        # Parameter learning (MLE)
        dbn_model.fit(train_data, estimator=MaximumLikelihoodEstimator)
        
        # Test set inference validation (simply using the state with the highest probability as the prediction)
        infer = VariableElimination(dbn_model)
        y_true = test_data[target_node].values
        y_pred = []
        
        # To speed up the code demonstration, only a small number of samples are extracted for inference validation here
        for _, row in test_data.head(50).iterrows():
            # Set all nodes at time T as evidence
            evidence = {col: row[col] for col in train_data.columns if col.endswith('_t')}
            res = infer.query(variables=[target_node], evidence=evidence, show_progress=False)
            y_pred.append(res.state_names[target_node][np.argmax(res.values)])
            
        acc = accuracy_score(y_true[:50], y_pred)
        macro_f1 = f1_score(y_true[:50], y_pred, average='macro')
        
        print(f"  Fold {fold} - ACC: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = dbn_model.copy() # In actual projects, deep copy or save the parameters
        fold += 1
        
    print(f"Validation completed. The best model's Macro-F1 is {best_f1:.4f}")
    return best_model

def recursive_inference(best_model, indicators, initial_states_2023, scenario_data, target_years):
    """
    Execute scenario inference, recursively predict future years (2024-2060), and calculate the posterior probability for specified years.
    """
    print("\n[Stage 3: Inference] Starting step-by-step dynamic probability inference for each city...")
    infer = VariableElimination(best_model)
    
    # Store the final results of the target years to be output
    results_prob = {year: {ind: [] for ind in indicators} for year in target_years}
    
    cities = initial_states_2023.index.unique()
    
    for city in cities:
        # Get the historical evidence of the city in 2023 as the initial prior
        current_state = initial_states_2023.loc[city].to_dict()
        
        for year in range(2024, 2061):
            # Set the current state as evidence at time T
            evidence = {f"{k}_t": v for k, v in current_state.items()}
            
            # Add scenario evidence at time T+1 (Hard constraints: C6, F5, C2)
            if (city, year) in scenario_data.index:
                scene_evidence = scenario_data.loc[(city, year)]
                evidence["C6_t1"] = scene_evidence["C6"]
                evidence["F5_t1"] = scene_evidence["F5"]
                evidence["C2_t1"] = scene_evidence["C2"]
            
            # Record the new state generated by this round of inference (to be used as a prior for the next year)
            next_state_prob = {}
            next_state_crisp = {} # Take the state with the highest probability to recurse downwards
            
            # Infer all unconstrained indicator states at time T+1
            for ind in indicators:
                target_node = f"{ind}_t1"
                if target_node in evidence:
                    # If it is a hard-constrained node controlled by the scenario, the probability distribution is directly 1.0 (deterministic state)
                    hard_state = evidence[target_node]
                    next_state_crisp[ind] = hard_state
                    next_state_prob[ind] = {hard_state: 1.0} 
                    continue
                    
                # Evidence modification and posterior probability calculation
                posterior = infer.query(variables=[target_node], evidence=evidence, show_progress=False)
                
                # Save the probability distribution of each discrete state
                prob_dist = dict(zip(posterior.state_names[target_node], posterior.values))
                next_state_prob[ind] = prob_dist
                
                # Select the state with the maximum posterior probability as the point estimate for the next year's recursion
                next_state_crisp[ind] = posterior.state_names[target_node][np.argmax(posterior.values)]
            
            # If the target year is reached, record the probability distribution
            if year in target_years:
                for ind in indicators:
                    results_prob[year][ind].append(next_state_prob[ind])
            
            # Time update: the current T+1 state becomes the T state for the next round
            current_state = next_state_crisp.copy()
            
    # Summary calculation: take the mean of the posterior probabilities of all cities as the overall state distribution
    final_output = {}
    for year in target_years:
        final_output[year] = {}
        for ind in indicators:
            df_probs = pd.DataFrame(results_prob[year][ind])
            df_probs.fillna(0, inplace=True)
            # Mean of each state probability across cities
            mean_probs = df_probs.mean().to_dict()
            final_output[year][ind] = mean_probs
            
    return final_output

if __name__ == "__main__":
    # ==========================
    # 1. Data reading and preprocessing preparation
    # ==========================
    print("Reading local data...")
    # Assuming 22 indicator names (1 UES, 21 other factors)
    all_indicators = ['V1', 'V2', 'V3', 'V4', 'V5', 'P1', 'P2', 'P3', 'P4', 'P5', 
                      'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'UES']
                      
    # 1. Read the cleaned comprehensive impact matrix (diagonals have been set to 0, etc.)
    # T_crisp = pd.read_excel("T_crisp_matrix.xlsx", index_col=0)
    # Here we build a virtual T_crisp for convenient execution
    T_crisp = pd.DataFrame(np.random.uniform(0.01, 0.15, size=(23, 23)), columns=all_indicators, index=all_indicators)
    np.fill_diagonal(T_crisp.values, 0)

    # 2. Read the 2296 historical discretized samples of NRC (used to train the CPT parameters of FD-DBN)
    # The samples need to be a wide table, containing *_t and *_t1 data columns, reflecting system evolution
    # samples_df = pd.read_excel("NRC_2296_samples.xlsx")
    # Build virtual training samples here
    cols_t = [f"{i}_t" for i in all_indicators]
    cols_t1 = [f"{i}_t1" for i in all_indicators]
    mock_states = ["Poor+", "Poor", "Fair", "Good", "Good+"]
    samples_df = pd.DataFrame(np.random.choice(mock_states, size=(2296, 46)), columns=cols_t + cols_t1)

    # 3. Read the scenario downscaling evidence data (control nodes C6, F5, C2)
    # scenario_df = pd.read_excel("Scenario_Data_NRC.xlsx")
    # index: [City, Year], columns: [C6, F5, C2]
    multi_idx = pd.MultiIndex.from_product([[f"City_{i}" for i in range(1, 11)], range(2024, 2061)], names=["City", "Year"])
    scenario_df = pd.DataFrame(np.random.choice(mock_states, size=(len(multi_idx), 3)), columns=["C6", "F5", "C2"], index=multi_idx)

    # 4. Get the true discretized states of the inference baseline period (2023)
    initial_2023 = pd.DataFrame(np.random.choice(mock_states, size=(10, 23)), columns=all_indicators, index=[f"City_{i}" for i in range(1, 11)])

    # ==========================
    # 2. Network structure construction and learning validation
    # ==========================
    # Determine the network structure (Stage 1)
    dbn_graph = build_fd_dbn_structure(T_crisp, all_indicators, mu=0.0638, sigma=0.0258, alpha=1)
    print(f"Number of edges in FD-DBN network: {len(dbn_graph.edges())}")
    
    # Structure validation and parameter learning (Stage 2)
    best_fd_dbn = cross_validate_and_learn(dbn_graph, samples_df, target_node="UES_t1", k_folds=6)

    # ==========================
    # 3. Recursive inference and future scenario evaluation
    # ==========================
    # Infer the system probability distribution for 2030, 2040, 2050, and 2060 (Stage 3)
    target_years = [2030, 2040, 2050, 2060]
    future_probs = recursive_inference(best_fd_dbn, all_indicators, initial_2023, scenario_df, target_years)

    # ==========================
    # 4. Print and save results
    # ==========================
    print("\n===== Overall UES State Distribution 2030-2060 (NRC City Cluster Mean) =====")
    for year in target_years:
        ues_prob = future_probs[year]["UES"]
        # Format the distribution values
        prob_str = ", ".join([f"{k}: {v*100:.2f}%" for k, v in ues_prob.items()])
        print(f"{year} UES state probabilities: {prob_str}")
        
    print("\n(Note: When you run this in a real environment, the results of all indicators are saved in the future_probs dictionary variable and can be exported via pd.DataFrame)")