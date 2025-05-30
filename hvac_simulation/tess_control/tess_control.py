"""This module implements TESS control for the hvac simulation in boptest"""

from typing import Dict, Optional, List, Union
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from hvac_simulation.bidding_strategy.heuristic_order import HVACOrder
from hvac_simulation.boptest.boptest_suite import BOPTESTClient as bp
from hvac_simulation.boptest.boptest_suite import seconds_to_datetime
from hvac_simulation.tess_control.kpi import HVAC_KPI


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

    def __init__(
            self,
            start_time: datetime,
            duration_hours: float = 24,
            warmup_period: float = 24,
            market_interval: float = 900,  # 15 minutes in seconds
            boptest_parms: dict=None,
            bt_instance=None,
            avaialble_control_inputs: list = ["con_oveTSetCoo_u", "con_oveTSetCoo_activate", "con_oveTSetHea_u", "con_oveTSetHea_activate"],
            control_default_values: list = [None, 0, None, 0],
            control_dr_values: list = [308.15, 1, 278.15, 1],
        ) -> None:
        """Initialize TESS Control with BOPTEST parameters."""
        self.bt_instance = bp(**boptest_parms) if boptest_parms else bt_instance
        self.hvac_order = HVACOrder(auction_id="linear_hvac_auction")
        self.current_price = None
        self.current_power = None
        self.kpi_calculator = None
        self.simulation_results = None
        self.start_time = start_time
        self.duration_hours = duration_hours
        self.warmup_period = warmup_period
        self.market_interval = market_interval
        self.initial_state = self.bt_instance.initialize(start_time, warmup_period=warmup_period*3600)  # Warmup for 1 day
        self.forecasted_data = self.bt_instance.get_forecast(horizon_hours=duration_hours)
        self.available_control_inputs = avaialble_control_inputs
        self.control_default_values = control_default_values
        self.control_dr_values = control_dr_values

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

    def get_hvac_setpoint_control_input(self, mode="cooling") -> None:
        """
        Set the HVAC temperature setpoint in the simulation.

        Parameters
        ----------
        setpoint : float
            Temperature setpoint in the unit specified in BOPTEST client
        """
        if mode == "cooling":
            return {
                'con_oveTSetCoo_u': (self.hvac_order.customer_parameters.get('desired_temp', None) - 32)*(5/9) + 273.15, # F to K
                'con_oveTSetCoo_activate': 1,
                'con_oveTSetHea_u': 278.15,
                'con_oveTSetHea_activate': 1
            }
        if mode == "heating":
            return {
                'con_oveTSetHea_u': (self.hvac_order.customer_parameters.get('desired_temp', None) - 32)*(5/9) + 273.15,
                'con_oveTSetHea_activate': 1,
                'con_oveTSetCoo_u': 308.15,
                'con_oveTSetCoo_activate': 1
            }
        if mode == "off":
            return {
                'con_oveTSetHea_u': 278.15,
                'con_oveTSetHea_activate': 1,
                'con_oveTSetCoo_u': 308.15,
                'con_oveTSetCoo_activate': 1
            }
        if mode == "default":
            return None
        raise ValueError(f"Invalid mode: {mode}")


    def get_hvac_setpoint_control_input_dr(self, mode="off"):
        control_input = {}
        if mode == "off":
            for i, c_input in enumerate(self.available_control_inputs):
                control_input[c_input] = self.control_dr_values[i]
        else:
            for i, c_input in enumerate(self.available_control_inputs):
                control_input[c_input] = self.control_default_values[i]
        return control_input
    
    def seconds_since_last_on_change(self, control_list: list) -> float:
        """
        Calculate the number of seconds since the last 'on' (heating or cooling) mode change.
        Returns 30 * 60 if the last recorded mode is 'off'.
        """
        seconds_since_on_change = 0
        
        # Iterate through control_list in reverse order
        for control in reversed(control_list):
            if control["mode"] in ["heating", "cooling"]:
                seconds_since_on_change += self.market_interval
            else:
                break  # Stop counting when an 'off' state is encountered

        # If the last state was 'off', return 30 minutes (1800 seconds)
        if seconds_since_on_change == 0:
            return 30 * 60
        return seconds_since_on_change
    
    def run_tess_simulation(
            self,
            market_expected_mean_price: np.array,
            market_expected_std_price: np.array,
            market_cleared_price: np.array,
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
        
        states_list = [self.initial_state]
        controls_list = [
            {
                c_input: self.initial_state[c_input] for i, c_input in enumerate(self.available_control_inputs)   
            }
        ]
        controls_list[0]["mode"] = "off"

        # Calculate number of market periods
        num_market_periods = int((self.duration_hours * 3600) / self.market_interval) - 1

        if market_expected_mean_price.shape[0] != num_market_periods + 1:
            raise ValueError("Market price data must match the number of market periods")
        if market_expected_std_price.shape[0] != num_market_periods + 1:
            raise ValueError("Market price data must match the number of market periods")
        if market_cleared_price.shape[0] != num_market_periods + 1:
            raise ValueError("Market price data must match the number of market periods")
        
        # Storage for results
        results = [
            {
                "timestamp": self.start_time,
                "temperature": self.initial_state["zon_reaTRooAir_y"],
                "power": 0,
                "price": market_cleared_price[0],
                "setpoint": self.hvac_order.customer_parameters.get('desired_temp', None)
            }
        ]
        
        for period in range(num_market_periods):
            current_state = states_list[-1]
            current_temp = current_state["zon_reaTRooAir_y"]
            # convert K to unit in BOPTEST
            current_temp = self.bt_instance.convert_temperature_value(current_temp)

            fan_power = current_state["fcu_reaPFan_y"]
            if current_temp < self.hvac_order.customer_parameters.get('desired_temp', None) - 1:
                mode_to_be = "heating"
            elif current_temp > self.hvac_order.customer_parameters.get('desired_temp', None) + 1:
                mode_to_be = "cooling"
            else:
                mode_to_be = "off"  # Temperature within deadband
            fan_mode = "auto"
            fan_state = "on" if fan_power > 0 else "off"
            power_rating = {'cooling': 1000, 'heating': 1000, 'fan': 100}  # Example values
            seconds_since_on_change = self.seconds_since_last_on_change(controls_list)
            # Update device parameters with current state
            device_params = {
                'current_temp': current_temp,
                'last_change_temp': current_temp,
                'max_temp': self.bt_instance.convert_temperature_value(308.15),
                'min_temp': self.bt_instance.convert_temperature_value(278.15),
                'mode': mode_to_be,
                'state': 'on' if mode_to_be != 'off' else 'off',
                'fan_mode': fan_mode,
                'fan_state': fan_state,
                'power_rating': power_rating,
                'seconds_since_on_change': seconds_since_on_change
            }
            self.set_hvac_device_parameters(device_params)
            self.hvac_order.market_parameters["expected_price"] = market_expected_mean_price[period+1]
            self.hvac_order.market_parameters["expected_stdev"] = market_expected_std_price[period+1]
            
            # Get HVAC order
            price, quantity = self.get_hvac_order()
            
            # when K value and std deviation is zero then there is no point of flexibly running the hvac system and set the temp to the desired temp
            
            # Store results
            result = {
                'timestamp': self.start_time + timedelta(seconds=(period+1) * self.market_interval),
                'temperature': device_params['current_temp'],
                'power': quantity if quantity > 0 else 0,
                'price': price,
                'setpoint': self.hvac_order.customer_parameters.get('desired_temp', None)
            }
            results.append(result)

            run=False
            
            # Apply control based on market clearing
            if price >= market_cleared_price[period+1]:
                run=True
                mode = "cooling" if mode_to_be == "cooling" else "heating"
            else:
                mode = "off"
                device_params['mode'] = "off"

            
            control_input = self.get_hvac_setpoint_control_input_dr(mode=mode)
            print(f"current_temp: {current_temp} and time: {self.start_time + timedelta(seconds=(period+1) * self.market_interval)}")
            print(f"mode: {mode}")
            print(f"control_input: {control_input}")
            new_state = self.bt_instance.advance(control_input)
            control_input["mode"] = mode
            states_list.append(new_state)
            controls_list.append(control_input)
        
        # Convert results to DataFrame
        self.simulation_results = pd.DataFrame(states_list)
        self.simulation_results["datetime"] = self.simulation_results['time'].apply(seconds_to_datetime)
        self.control_list = pd.DataFrame(controls_list)
        self.control_list["datetime"] = self.simulation_results["datetime"]
        self.tess_results = pd.DataFrame(results)
        self.tess_results["datetime"] = self.tess_results["timestamp"]
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