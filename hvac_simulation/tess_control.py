"""This module implements TESS control for the hvac simulation in boptest"""

from typing import Dict, Optional, List, Union
import pandas as pd
from datetime import datetime, timedelta

from hvac_simulation.heuristic_order import HVACOrder
from hvac_simulation.boptest_suite import BOPTESTClient as bp
from hvac_simulation.kpi import HVAC_KPI

class TESSControl:
    """
    This class implements the TESS control for the HVAC simulation in BOPTest.
    It manages the interaction between market parameters and HVAC control settings.

    Parameters
    ----------
    boptest_parms : dict
        Parameters needed to instantiate the boptest class:
        - base_url: str
        - testcase: str
        - timeout: int (optional)
        - control_step: float (optional)
        - temp_unit: str (optional)
        - start_date: datetime (optional)
    """

    def __init__(self, boptest_parms: dict):
        """Initialize TESS Control with BOPTEST parameters."""
        self.bp_instance = bp(**boptest_parms)
        self.hvac_order = HVACOrder()
        self.current_price = None
        self.current_power = None
        self.kpi_calculator = None
        self.simulation_results = None

    def set_hvac_market_parameters(self, market_parms: dict) -> None:
        """
        Set the market parameters for the HVAC simulation.

        Parameters
        ----------
        market_parms : dict
            Market parameters including:
            - expected_stdev: float
            - expected_price: float
            - min_price: float
            - max_price: float
            - interval: float (market interval in seconds)
        """
        self.hvac_order.set_market_parameters(market_parms)

    def set_hvac_device_parameters(self, device_parms: dict) -> None:
        """
        Set the device parameters for the HVAC simulation.

        Parameters
        ----------
        device_parms : dict
            Device parameters including:
            - current_temp: float
            - last_change_temp: float
            - max_temp: float
            - min_temp: float
            - mode: str
            - state: str
            - fan_mode: str
            - fan_state: str
            - power_rating: dict
            - seconds_since_on_change: float
        """
        self.hvac_order.set_device_parameters(device_parms)

    def set_hvac_customer_parameters(self, customer_parms: dict) -> None:
        """
        Set the customer parameters for the HVAC simulation.

        Parameters
        ----------
        customer_parms : dict
            Customer parameters including:
            - K_hvac: float (customer sensitivity to price)
            - desired_temp: float
        """
        self.hvac_order.set_customer_parameters(customer_parms)

    def get_hvac_order(self) -> tuple[float, float]:
        """
        Calculate the HVAC order based on current parameters.

        Returns
        -------
        tuple[float, float]
            A tuple containing (price, quantity) for the HVAC order
        """
        price = self.hvac_order.get_power_price()
        quantity = self.hvac_order.get_power_quantity()
        return price, quantity

    def get_market_cleared_price(self, orders: List[Dict[str, float]]) -> float:
        """
        Calculate the market clearing price based on submitted orders.

        Parameters
        ----------
        orders : List[Dict[str, float]]
            List of orders with 'price' and 'quantity' keys

        Returns
        -------
        float
            Market clearing price
        """
        if not orders:
            return self.hvac_order.market_parameters['expected_price']

        # Sort orders by price
        sorted_orders = sorted(orders, key=lambda x: x['price'])
        
        # Find intersection of supply and demand
        cumulative_quantity = 0
        for order in sorted_orders:
            cumulative_quantity += order['quantity']
            if cumulative_quantity >= 0:  # Market clears when cumulative quantity becomes positive
                return order['price']

        # If no clearing price found, return the last price
        return sorted_orders[-1]['price']

    def set_hvac_setpoint(self, setpoint: float) -> None:
        """
        Set the HVAC temperature setpoint in the simulation.

        Parameters
        ----------
        setpoint : float
            Temperature setpoint in the unit specified in BOPTEST client
        """
        control_input = {
            'oveTSetCoo_u': setpoint + 273.15 if self.bp_instance.temp_unit == 'C' else (setpoint - 32) * 5/9 + 273.15,
            'oveTSetCoo_activate': 1
        }
        self.bp_instance.advance(control_input)

    def run_tess_simulation(
            self, 
            duration_hours: float = 24,
            market_interval: float = 900,  # 15 minutes in seconds
            control_inputs: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
        """
        Run the TESS-controlled HVAC simulation.

        Parameters
        ----------
        duration_hours : float
            Duration of simulation in hours
        market_interval : float
            Market clearing interval in seconds
        control_inputs : Optional[pd.DataFrame]
            Pre-defined control inputs for the simulation

        Returns
        -------
        pd.DataFrame
            Simulation results including temperatures, power consumption, and market prices
        """
        # Initialize simulation
        start_time = datetime.now()
        self.bp_instance.initialize(start_time)

        # Calculate number of market periods
        num_market_periods = int((duration_hours * 3600) / market_interval)
        
        # Storage for results
        results = []
        
        for period in range(num_market_periods):
            # Get current system state
            current_state = self.bp_instance.advance(None)
            
            # Update device parameters with current state
            device_params = {
                'current_temp': current_state[self.bp_instance.measurements['Zone Temperature']['Variable Name']],
                'last_change_temp': current_state.get('last_temp_change', 0),
                'mode': 'auto',
                'state': 'on' if current_state.get('power', 0) > 0 else 'off',
                'fan_mode': 'auto',
                'fan_state': 'on' if current_state.get('fan_power', 0) > 0 else 'off',
                'power_rating': {'cool': 1000, 'heat': 1000, 'fan': 100},  # Example values
                'seconds_since_on_change': market_interval
            }
            self.set_hvac_device_parameters(device_params)
            
            # Get HVAC order
            price, quantity = self.get_hvac_order()
            
            # Store results
            result = {
                'timestamp': start_time + timedelta(seconds=period * market_interval),
                'temperature': device_params['current_temp'],
                'power': quantity if quantity > 0 else 0,
                'price': price,
                'setpoint': self.hvac_order.customer_parameters.get('desired_temp', None)
            }
            results.append(result)
            
            # Apply control based on market clearing
            if price <= self.get_market_cleared_price([{'price': price, 'quantity': quantity}]):
                self.set_hvac_setpoint(self.hvac_order.customer_parameters['desired_temp'])
        
        # Convert results to DataFrame
        self.simulation_results = pd.DataFrame(results)
        
        # Initialize KPI calculator
        self.kpi_calculator = HVAC_KPI(
            self.simulation_results,
            temperature_variable='temperature',
            setpoint_temperature=self.hvac_order.customer_parameters.get('desired_temp', 22.0)
        )
        
        return self.simulation_results

    def get_simulation_kpis(self) -> Dict[str, float]:
        """
        Calculate and return KPIs for the completed simulation.

        Returns
        -------
        Dict[str, float]
            Dictionary containing KPI values:
            - energy_consumption: Total energy consumption in kWh
            - peak_power: Peak power consumption in kW
            - temperature_discomfort: Average temperature deviation
            - total_cost: Total energy cost
        """
        if self.simulation_results is None or self.kpi_calculator is None:
            raise ValueError("Simulation must be run before calculating KPIs")

        kpis = {
            'energy_consumption': self.kpi_calculator.calculate_energy_consumption(),
            'peak_power': self.kpi_calculator.calculate_peak_power(),
            'temperature_discomfort': self.kpi_calculator.calculate_temperature_discomfort(),
            'total_cost': (self.simulation_results['power'] * self.simulation_results['price']).sum()
        }
        
        return kpis