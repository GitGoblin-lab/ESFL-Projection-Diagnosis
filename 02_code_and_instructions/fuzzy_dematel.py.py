# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np

def get_tfn_mapping():
    """
    
    """
    return {
        0: (0.00, 0.00, 0.25),
        1: (0.00, 0.25, 0.50),
        2: (0.25, 0.50, 0.75),
        3: (0.50, 0.75, 1.00),
        4: (0.75, 1.00, 1.00)
    }

def run_fuzzy_dematel(file_path):
    # ==========================================
    # Step 1
    # ==========================================
    xls_data = pd.read_excel(file_path, sheet_name=None, header=None)
    sheet_names = list(xls_data.keys())
    
    num_experts = len(sheet_names)
    num_factors = 22 
    tfn_mapping = get_tfn_mapping()
    

    Z_l = np.zeros((num_factors, num_factors))
    Z_m = np.zeros((num_factors, num_factors))
    Z_u = np.zeros((num_factors, num_factors))
    

    for sheet in sheet_names:
        df = xls_data[sheet]
        
       
        matrix = df.values
        
        for i in range(num_factors):
            for j in range(num_factors):
                score = matrix[i, j]
                
                tfn = tfn_mapping.get(score, (0.0, 0.0, 0.25))
                Z_l[i, j] += tfn[0]
                Z_m[i, j] += tfn[1]
                Z_u[i, j] += tfn[2]
                
    
    Z_l /= num_experts
    Z_m /= num_experts
    Z_u /= num_experts
    
    # ==========================================
    # Step 2
    # ==========================================
    r = np.max(np.sum(Z_u, axis=1))
    
    X_l = Z_l / r
    X_m = Z_m / r
    X_u = Z_u / r
    
    # ==========================================
    # Step 3
    # ==========================================
    # T = X * (I - X)^-1 （公式 5 - 7）
    I = np.eye(num_factors)
    
    T_l = np.dot(X_l, np.linalg.inv(I - X_l))
    T_m = np.dot(X_m, np.linalg.inv(I - X_m))
    T_u = np.dot(X_u, np.linalg.inv(I - X_u))
    
    # ==========================================
    # Step 4
    # ==========================================
    min_l = np.min(T_l)
    max_u = np.max(T_u)
    delta = max_u - min_l
    

    xl = (T_l - min_l) / delta
    xm = (T_m - min_l) / delta
    xu = (T_u - min_l) / delta
    

    x_ls = xm / (1 + xm - xl)
    x_us = xu / (1 + xu - xm)
    

    y = (x_ls * (1 - x_ls) + x_us * x_us) / (1 - x_ls + x_us)
    
    T_crisp = min_l + y * delta
    
    # ==========================================
    # Step 5
    # ==========================================

    Inf = np.sum(T_crisp, axis=1)
    Aff = np.sum(T_crisp, axis=0)
    Cen = Inf + Aff
    Cau = Inf - Aff
    causal_params_df = pd.DataFrame({
        'Indicators': [f'X{i+1}' for i in range(num_factors)],
        'influence degree (Inf)': Inf,
        'affectedness degree (Aff)': Aff,
        'centrality degree (Cen)': Cen,
        'causality degree (Cau)': Cau
    })
    

    T_crisp_df = pd.DataFrame(
        T_crisp, 
        columns=[f'X{i+1}' for i in range(num_factors)], 
        index=[f'X{i+1}' for i in range(num_factors)]
    )
    
    return T_crisp_df, causal_params_df

if __name__ == "__main__":
    file_name = "Expert Scoring Matrix.xlsx"
    
    try:
        T_crisp_df, causal_params = run_fuzzy_dematel(file_name)
        
        print("===== Clarifying the Comprehensive Impact Matrix (T_crisp) 提取前5x5展示 =====")
        print(T_crisp_df.iloc[:5, :5])
        print("\n===== Calculation Results for 4 Causal Parameters =====")
        print(causal_params)
        
        with pd.ExcelWriter("Fuzzy_DEMATEL_Results.xlsx") as writer:
            T_crisp_df.to_excel(writer, sheet_name="Comprehensive Impact Matrix")
            causal_params.to_excel(writer, sheet_name="Causal Parameters", index=False)
            
        print("\n Operation successful")
        
    except Exception as e:
        print(f"An error occurred during the calculation. Please check the Excel structure. Error message: {e}")