import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np
import time
import os
import glob

from hvac_simulation.boptest.boptest_suite import BOPTESTClient as bt
from hvac_simulation.tess_control.kpi import HVAC_KPI
from hvac_simulation.tess_control.tess_control import TESSControl
from hvac_simulation.utils.logger import setup_logger

def print_elapsed_time(start_time, description, logger):
    elapsed = time.time() - start_time
    logger.info(f"  └─ {description}: {elapsed:.2f} seconds")

def get_next_run_number():
    """Get the next run number by finding the latest RUN_* folder and incrementing its number."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    run_folders = glob.glob(os.path.join(data_dir, "RUN_*"))
    if not run_folders:
        return 1
    
    # Extract numbers from folder names and find the maximum
    numbers = [int(folder.split('_')[1]) for folder in run_folders]
    return max(numbers) + 1

# Create run folder
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

current_date = datetime.now().strftime("%Y%m%d")
run_number = get_next_run_number()
run_folder = os.path.join(data_dir, f"RUN_{run_number}_{current_date}")
os.makedirs(run_folder, exist_ok=True)

# Setup logger
logger = setup_logger(run_folder)
logger.info(f"\nCreated run folder: {run_folder}")

BOPTEST_URL = "http://127.0.0.1"

test_case_parameters = {
    "test_case": "bestest_hydronic_heat_pump",
    "start_date": "2023-01-07",
    "number_of_days": 5,
    "temperature_unit": "F",
    "control_step": 900,
    "warmup_days": 1,
}

test_case = test_case_parameters["test_case"]
control_step = test_case_parameters["control_step"]
temperature_unit = test_case_parameters["temperature_unit"]
simul_start_date = datetime.strptime(test_case_parameters["start_date"], "%Y-%m-%d")
simul_days = test_case_parameters["number_of_days"]
simul_hours = simul_days * 24
warmup_days = test_case_parameters["warmup_days"]

k_value_range = np.arange(0.1, 15, 0.5)
kpi_values_list = []

logger.info("\n" + "=" * 50)
logger.info("EXPERIMENT CONFIGURATION")
logger.info("=" * 50)
logger.info(f"Total iterations: {len(k_value_range)}")
logger.info(f"Test case: {test_case}")
logger.info(f"Simulation period: {simul_days} days")
logger.info(f"K value range: {k_value_range[0]} to {k_value_range[-1]}")
logger.info(f"Control step: {control_step} seconds")
logger.info(f"Temperature unit: {temperature_unit}")
logger.info("=" * 50 + "\n")

total_start_time = time.time()

for idx, k_value in enumerate(k_value_range, 1):
    iteration_start_time = time.time()
    logger.info(
        f"\n{'='*20} Iteration {idx}/{len(k_value_range)} - K value: {k_value:.2f} {'='*20}"
    )

    # Initialize BOPTEST
    step_start = time.time()
    logger.info("\n1. Initializing BOPTEST instance...")
    bt_instance = bt(
        base_url=BOPTEST_URL,
        testcase=test_case_parameters["test_case"],
        control_step=test_case_parameters["control_step"],
        temp_unit=test_case_parameters["temperature_unit"],
        timeout=300,
    )
    print_elapsed_time(step_start, "BOPTEST initialization completed", logger)

    # Setup parameters
    step_start = time.time()
    logger.info("\n2. Setting up simulation parameters...")
    customer_parameters = {"K_hvac": k_value, "desired_temp": 70}
    market_parameters = {
        "min_price": 0.001 * 1000,
        "max_price": 2 * 1000,
        "interval": 300,
    }
    boptest_param = {
        "base_url": BOPTEST_URL,
        "testcase": test_case,
        "control_step": control_step,
        "temp_unit": temperature_unit,
        "timeout": 300,
    }
    print_elapsed_time(step_start, "Parameter setup completed", logger)

    # Setup TESS simulation
    step_start = time.time()
    logger.info("\n3. Setting up TESS simulation...")
    tess_simul = TESSControl(
        bt_instance=bt_instance,
        start_time=simul_start_date,
        duration_hours=simul_hours,
        warmup_period=warmup_days * 24,
        market_interval=control_step,
        avaialble_control_inputs=[
            "oveHeaPumY_activate",
            "oveHeaPumY_u",
            "oveTSet_u",
            "oveTSet_activate",
        ],
        control_default_values=[0, None, 294.261, 1],
        control_dr_values=[1, 0, 0, None],
    )
    print_elapsed_time(step_start, "TESS simulation setup completed", logger)

    # Setup parameters
    step_start = time.time()
    logger.info("\n4. Setting HVAC parameters...")
    tess_simul.set_hvac_customer_parameters(customer_parameters)
    tess_simul.set_hvac_market_parameters(market_parameters)
    print_elapsed_time(step_start, "Parameter setting completed", logger)

    # Calculate market prices
    step_start = time.time()
    logger.info("\n5. Calculating market prices...")
    timesteps = int((simul_hours * 3600) / control_step)
    one_array = np.ones(timesteps)
    outdoor_temp = tess_simul.forecasted_data["TDryBul"]
    temp_deviations = np.mean(outdoor_temp) - outdoor_temp

    market_cleared_price = one_array * 50 + temp_deviations
    market_expected_mean_price = market_cleared_price + np.random.normal(0, 2, timesteps)
    market_expected_std_price = temp_deviations / np.linalg.norm(temp_deviations) * 1
    market_expected_std_price = market_expected_mean_price * 0.01
    print_elapsed_time(step_start, "Market price calculation completed", logger)

    # Run simulation
    step_start = time.time()
    logger.info("\n6. Running TESS simulation...")
    simul_result = tess_simul.run_tess_simulation(
        market_expected_mean_price, market_expected_std_price, market_cleared_price
    )
    print_elapsed_time(step_start, "TESS simulation completed", logger)

    # Calculate KPIs
    step_start = time.time()
    logger.info("\n7. Calculating KPIs...")
    kpi = HVAC_KPI(
        tess_simul.simulation_results,
        control_step=control_step,
        heating_power_variable="reaPHeaPum_y",
        temperature_variable="reaTZon_y",
        setpoint_temperature=(customer_parameters["desired_temp"] - 32) * 5 / 9
        + 273.15,
        temperature_unit="K",
        logger=logger
    )

    kpi_values = {
        "K_hvac": k_value,
        "Energy Consumption (kWh)": kpi.calculate_energy_consumption() / 1000,
        "Peak Power (kW)": kpi.calculate_peak_power() / 1000,
        "Temperature Discomfort": kpi.calculate_temperature_discomfort(),
        "Average Heating Cycles per hour": kpi.calculate_cycles(
            kpi.heating_power_variable
        )["average_cycles_per_hour"],
    }
    boptest_kpi = bt_instance.get_kpis().to_dict()["Value"]  # dict of kpis from boptest
    # merge boptest kpis with tess kpis
    kpi_values = {**kpi_values, **boptest_kpi}
    kpi_values_list.append(kpi_values)
    logger.info("  └─ KPI Results:")
    for key, value in kpi_values.items():
        if key and value:
            logger.info(f"     - {key}: {value:.3f}")
    print_elapsed_time(step_start, "KPI calculation completed", logger)

    # Get BOPTEST KPIs
    step_start = time.time()
    logger.info("\n8. Getting BOPTEST KPIs...")
    boptest_kpi = bt_instance.get_kpis()
    print_elapsed_time(step_start, "BOPTEST KPI retrieval completed", logger)

    # Save results
    step_start = time.time()
    logger.info("\n9. Saving simulation results...")
    pd.DataFrame(tess_simul.simulation_results).to_csv(f"{run_folder}/tess_simulation_results_{k_value}.csv", index=False)
    kpi_values_list.append(kpi_values)
    print_elapsed_time(step_start, "Results saving completed", logger)

    # Stop BOPTEST
    step_start = time.time()
    logger.info("\n10. Stopping BOPTEST instance...")
    bt_instance.stop()
    print_elapsed_time(step_start, "BOPTEST shutdown completed", logger)

    # Print iteration summary
    iteration_end_time = time.time()
    iteration_duration = iteration_end_time - iteration_start_time
    logger.info("\n" + "-" * 50)
    logger.info("ITERATION SUMMARY")
    logger.info("-" * 50)
    logger.info(f"Total iteration time: {iteration_duration:.2f} seconds")

    # Progress and estimates
    avg_time_per_iteration = (iteration_end_time - total_start_time) / idx
    remaining_iterations = len(k_value_range) - idx
    estimated_remaining_time = avg_time_per_iteration * remaining_iterations
    logger.info(f"Average time per iteration: {avg_time_per_iteration:.2f} seconds")
    logger.info(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
    logger.info(
        f"Estimated completion time: {datetime.now() + timedelta(seconds=estimated_remaining_time)}"
    )
    logger.info("-" * 50)

# Final Summary
total_end_time = time.time()
total_duration = total_end_time - total_start_time

logger.info("\n" + "="*50)
logger.info("FINAL EXPERIMENT SUMMARY")
logger.info("="*50)
logger.info(f"Start time: {datetime.fromtimestamp(total_start_time)}")
logger.info(f"End time: {datetime.fromtimestamp(total_end_time)}")
logger.info(f"Total time elapsed: {total_duration/60:.1f} minutes")
logger.info(f"Average time per iteration: {total_duration/len(k_value_range):.2f} seconds")
logger.info(f"Total iterations completed: {len(k_value_range)}")
logger.info("="*50)

logger.info("\nSaving final KPI values...")
kpi_df = pd.DataFrame(kpi_values_list)
kpi_df.to_csv(f"{run_folder}/kpi_values.csv", index=False)

# Save experiment configuration
config_data = {
    "run_folder": run_folder,
    "test_case_parameters": test_case_parameters,
    "k_value_range": k_value_range.tolist(),
    "start_time": datetime.fromtimestamp(total_start_time).isoformat(),
    "end_time": datetime.fromtimestamp(total_end_time).isoformat(),
    "total_duration_minutes": total_duration/60
}
with open(f"{run_folder}/experiment_config.json", 'w') as f:
    json.dump(config_data, f, indent=4)

logger.info(f"All data saved to folder: {run_folder}")
logger.info("Experiment completed successfully!") 
