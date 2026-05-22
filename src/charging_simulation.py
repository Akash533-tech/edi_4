import numpy as np
import pandas as pd
import os

def run_charging_simulation():
    n_cycles = 250
    t_amb = 25.0
    ri_0 = 0.08
    theta = 20.0  # Thermal coefficient (C / (A^2 * Ohm))
    alpha = 0.12  # Baseline degradation rate per cycle
    gamma = 0.08  # Temp acceleration coefficient (per C above ambient)
    delta = 0.015 # Ri increase per % SOH lost
    
    # Trajectories for Scenario A (Standard)
    soh_a = np.zeros(n_cycles)
    ri_a = np.zeros(n_cycles)
    temp_a = np.zeros(n_cycles)
    curr_a = np.zeros(n_cycles)
    
    # Trajectories for Scenario B (Smart)
    soh_b = np.zeros(n_cycles)
    ri_b = np.zeros(n_cycles)
    temp_b = np.zeros(n_cycles)
    curr_b = np.zeros(n_cycles)
    
    # Initial conditions
    soh_a[0] = 100.0
    ri_a[0] = ri_0
    
    soh_b[0] = 100.0
    ri_b[0] = ri_0
    
    # Simulation Loop
    for k in range(n_cycles):
        # --- SCENARIO A (Standard 1.0C charging always) ---
        curr_a[k] = 2.0  # 1.0C
        # Temperature rise from internal heating (I^2 * R * theta)
        dT_a = theta * (curr_a[k] ** 2) * ri_a[k]
        temp_a[k] = t_amb + dT_a
        
        # SOH decay
        if k < n_cycles - 1:
            degr_rate_a = alpha * (1.0 + gamma * dT_a)
            soh_a[k+1] = max(0.0, soh_a[k] - degr_rate_a)
            # Ri growth
            ri_a[k+1] = ri_0 * (1.0 + delta * (100.0 - soh_a[k+1]))
            
        # --- SCENARIO B (Smart Charging, cap to 0.5C if Ri grows > 15%) ---
        threshold_ri = 1.15 * ri_0
        if ri_b[k] >= threshold_ri:
            curr_b[k] = 1.0  # Cap at 0.5C
        else:
            curr_b[k] = 2.0  # Standard 1.0C
            
        dT_b = theta * (curr_b[k] ** 2) * ri_b[k]
        temp_b[k] = t_amb + dT_b
        
        if k < n_cycles - 1:
            degr_rate_b = alpha * (1.0 + gamma * dT_b)
            soh_b[k+1] = max(0.0, soh_b[k] - degr_rate_b)
            # Ri growth
            ri_b[k+1] = ri_0 * (1.0 + delta * (100.0 - soh_b[k+1]))
            
    # Compile into DataFrame
    simulation_results = []
    for k in range(n_cycles):
        simulation_results.append({
            'cycle': k + 1,
            # Scenario A
            'soh_A': soh_a[k],
            'ri_A': ri_a[k],
            'temp_A': temp_a[k],
            'curr_A': curr_a[k],
            # Scenario B
            'soh_B': soh_b[k],
            'ri_B': ri_b[k],
            'temp_B': temp_b[k],
            'curr_B': curr_b[k],
        })
        
    df_sim = pd.DataFrame(simulation_results)
    
    # Save simulation results
    output_dir = "/Users/akashsunilsomsetwar/Desktop/edi_4/data_processed"
    df_sim.to_csv(os.path.join(output_dir, "charging_simulation.csv"), index=False)
    print(f"Charging simulation completed. Saved results to charging_simulation.csv")
    
    # Find and print EOL (80% SOH) for both
    eol_cycle_a = next((i+1 for i, s in enumerate(soh_a) if s < 80.0), None)
    eol_cycle_b = next((i+1 for i, s in enumerate(soh_b) if s < 80.0), None)
    print(f"Scenario A EOL (80% SOH): Cycle {eol_cycle_a if eol_cycle_a else 'Never'}")
    print(f"Scenario B EOL (80% SOH): Cycle {eol_cycle_b if eol_cycle_b else 'Never'}")
    if eol_cycle_a and eol_cycle_b:
        extension = ((eol_cycle_b - eol_cycle_a) / eol_cycle_a) * 100.0
        print(f"Lifetime extension: {extension:.1f}%")

if __name__ == "__main__":
    run_charging_simulation()
