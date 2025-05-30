===============
hvac-simulation
===============


.. image:: https://img.shields.io/pypi/v/hvac_simulation.svg
        :target: https://pypi.python.org/pypi/hvac_simulation

.. image:: https://img.shields.io/travis/ThyanRevolter/hvac_simulation.svg
        :target: https://travis-ci.com/ThyanRevolter/hvac_simulation

.. image:: https://readthedocs.org/projects/hvac-simulation/badge/?version=latest
        :target: https://hvac-simulation.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




This project simulates the behavior of HVAC (Heating, Ventilation, and Air Conditioning) systems to optimize energy consumption and maintain desired indoor climate conditions.


* Free software: MIT
* Documentation: https://hvac-simulation.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `briggySmalls/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage

=====================================
HVAC Simulation with TESS Control
=====================================

How to Run the Simulation
==========================

Prerequisites
-------------

Before running the simulation, ensure you have the following installed:

1. **Python 3.8+**
2. **Docker**
   
   - Install Docker Desktop
   - Clone the BOPTEST repository:
   
   .. code-block:: bash
   
      git clone https://github.com/ibpsa/project1-boptest.git

Installation Steps
------------------

1. **Clone the HVAC Simulation Repository**

   .. code-block:: bash
   
      git clone https://github.com/ThyanRevolter/hvac-simulation.git
      cd hvac-simulation

2. **Install the Package**

   .. code-block:: bash
   
      pip install -e .

Configuration
-------------

The simulation uses several key parameters that can be configured:

**Test Case Parameters** (in ``tess_experiment.py``):

- BOPTEST Parameters:
.. code-block:: python

   test_case_parameters = {
       "test_case": "bestest_hydronic_heat_pump",
       "start_date": "2023-01-07",
       "number_of_days": 5,
       "temperature_unit": "F",
       "control_step": 900,  # 15 minutes in seconds
       "warmup_days": 1,
   }

**Customer Parameters**:

- ``K_hvac``: Customer flexibility parameter (0.1 to 15.0)
- ``desired_temp``: Desired temperature in Fahrenheit (e.g., 70°F)

**Market Parameters**:

- ``min_price``: Minimum electricity price ($/kWh)
- ``max_price``: Maximum electricity price ($/kWh)
- ``interval``: Market clearing interval (seconds)

Running the Simulation
----------------------

1. **Start BOPTEST Framework**

   Navigate to your BOPTEST directory and start the services:

   .. code-block:: bash
   
      cd project1-boptest
      docker compose up web worker provision

2. **Run the Main Experiment**

   In a separate terminal, navigate to your HVAC simulation directory:

   .. code-block:: bash
   
      cd hvac-simulation
      python tess_experiment.py

3. **Monitor Progress**

   The simulation will:
   
   - Create a run folder: ``data/RUN_X_YYYYMMDD/``
   - Log progress to console and ``simulation.log``
   - Show detailed temperature and pricing information
   - Display estimated completion time

4. **Simulation Workflow**

   For each K_hvac value (0.1 to 15.0 in 0.5 increments):
   
   - Initialize BOPTEST instance
   - Set up simulation parameters
   - Configure TESS control system
   - Calculate market prices based on outdoor temperature
   - Run HVAC simulation with demand response
   - Calculate KPIs (energy consumption, peak power, comfort, cycles)
   - Save results to CSV files

5. **Shutdown BOPTEST**

   When finished, properly shutdown BOPTEST:

   .. code-block:: bash
   
      cd project1-boptest
      docker compose down

Understanding the Output
------------------------

**Generated Files**:

- ``data/RUN_X_YYYYMMDD/simulation.log``: Detailed execution log
- ``data/RUN_X_YYYYMMDD/kpi_values.csv``: KPI results for all K values
- ``data/RUN_X_YYYYMMDD/tess_simulation_results_X.X.csv``: Detailed simulation data per K value
- ``data/RUN_X_YYYYMMDD/experiment_config.json``: Experiment configuration and timing

**Key KPIs**:

- **Energy Consumption (kWh)**: Total HVAC energy usage
- **Peak Power (kW)**: Maximum instantaneous power draw
- **Temperature Discomfort**: Average deviation from setpoint
- **Average Heating Cycles per Hour**: System on/off cycling frequency
- **BOPTEST KPIs**: Additional metrics from the building simulation

**Console Output Example**:

.. code-block:: text

   ==================== Iteration 1/29 - K value: 0.10 ====================
   
   1. Initializing BOPTEST instance...
     └─ BOPTEST initialization completed: 2.45 seconds
   
   Temperature Parameters:
     - Current Temperature: 68.50°F
     - Desired Temperature: 70.00°F
     - Last Temperature Change: 0.00°F
     - Mode: heating
     
   Heating Price Calculation:
     - Current Temp: 68.50°F
     - Min Temp: 41.00°F
     - Desired Temp: 70.00°F
     - Temperature Ratio: 0.0182
     - Final Price: $52.36

Troubleshooting
---------------

**Common Issues**:

1. **BOPTEST Connection Error**:
   
   .. code-block:: text
   
      ConnectionError: Unable to connect to BOPTEST
   
   **Solution**: Ensure BOPTEST framework is running:
   
   .. code-block:: bash
   
      cd project1-boptest
      docker compose up web worker provision
      
   **Available Test Cases**: The simulation uses ``bestest_hydronic_heat_pump``. 
   You can verify available test cases by visiting ``http://127.0.0.1:80/testcases``

2. **Import Errors**:
   
   .. code-block:: text
   
      ModuleNotFoundError: No module named 'hvac_simulation'
   
   **Solution**: Install the package in development mode:
   
   .. code-block:: bash
   
      pip install -e .

3. **Unicode Encoding Error**:
   
   .. code-block:: text
   
      UnicodeEncodeError: 'charmap' codec can't encode characters
   
   **Solution**: The logging system handles UTF-8 encoding automatically.

4. **Memory Issues with Long Simulations**:
   
   **Solution**: Reduce the K value range or simulation duration:
   
   .. code-block:: python
   
      k_value_range = np.arange(0.1, 5.0, 1.0)  # Smaller range
      test_case_parameters["number_of_days"] = 1  # Shorter simulation

**Performance Tips**:

- **Typical Runtime**: ~2-3 minutes per K value iteration
- **Total Time**: ~60-90 minutes for full parameter sweep (29 iterations)
- **Memory Usage**: ~500MB-1GB depending on simulation length
- **Disk Space**: ~50-100MB per run folder

Advanced Configuration
----------------------

**Custom K Value Range**:

.. code-block:: python

   k_value_range = np.array([0.5, 1.0, 2.0, 5.0])  # Custom values
   # OR
   k_value_range = np.arange(0.1, 10.0, 0.2)  # Finer resolution

**Different Test Cases**:

.. code-block:: python

   test_case_parameters["test_case"] = "bestest_air"  # Different building type

**Market Price Customization**:

Modify the market price calculation in ``tess_experiment.py``:

.. code-block:: python

   # Custom price volatility
   market_expected_std_price = temp_deviations / np.linalg.norm(temp_deviations) * 5  # Higher volatility

**Logging Levels**:

Adjust logging detail in ``hvac_simulation/utils/logger.py``:

.. code-block:: python

   console_handler.setLevel(logging.WARNING)  # Less console output
   file_handler.setLevel(logging.DEBUG)       # Full file logging

Support
-------

For issues and questions:

1. **HVAC Simulation Issues**:
   - Check the simulation log file: ``data/RUN_X_YYYYMMDD/simulation.log``
   - Review the experiment configuration: ``data/RUN_X_YYYYMMDD/experiment_config.json``
   - Verify package installation: ``pip show hvac-simulation``

2. **BOPTEST Framework Issues**:
   - Verify BOPTEST status: ``docker ps`` (should show web and worker containers)
   - Check BOPTEST logs: ``docker compose logs`` (from project1-boptest directory)
   - Test API connectivity: ``curl http://127.0.0.1:80/testcases``
   - List available test cases: ``curl http://127.0.0.1:80/testcases``

3. **General Support**:
   - HVAC Simulation Repository: Open an issue on GitHub with log files attached
   - BOPTEST Framework: Visit `BOPTEST documentation <https://ibpsa.github.io/project1-boptest/>`_
   - Docker Issues: Check Docker Desktop status and restart if necessary
