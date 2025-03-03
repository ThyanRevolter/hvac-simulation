import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import requests
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta, date
    import numpy as np 
    import random
    return (
        date,
        datetime,
        json,
        mo,
        np,
        pd,
        plt,
        random,
        requests,
        timedelta,
    )


@app.cell
def _(pd):
    ami_data = pd.read_csv("ami_data_973_customers.csv")
    number_of_customers = len(ami_data["customer_id"].unique())
    current_year = pd.Timestamp.now().year
    ami_data["datetime"] = pd.to_datetime(ami_data["record_date"] + f"/{current_year}", format="%m/%d/%Y") + pd.to_timedelta(ami_data["hour_id"], unit='h')
    ami_data["datetime_utc"] = ami_data["datetime"] - pd.to_timedelta(ami_data["utc_offset"], unit="h")
    return ami_data, current_year, number_of_customers


@app.cell
def _(ami_data, date, mo):
    subset_customers = ami_data["customer_id"].unique()[0:100]
    select_day = mo.ui.date(
        start=date(2025, 1, 1),
        stop=date(2025, 1, 31),
        value=date(2025, 1, 1),
        label="Select Day"
    )
    select_day
    return select_day, subset_customers


@app.cell
def _(ami_data, select_day, subset_customers):
    # filter the data for the selected day and the customer subset
    subset_data = ami_data[ami_data["customer_id"].isin(subset_customers) & (ami_data["datetime"].dt.date == select_day.value)]
    return (subset_data,)


@app.cell
def _(
    batt_total_load,
    hour_agg,
    hvac_total_load,
    hvac_total_load_og,
    np,
    plt,
):
    plt.figure(figsize=(10, 6), dpi=500)
    plt.plot(hour_agg.index, hour_agg['energy_value'], color="black", linewidth=0.2)
    plt.plot(hour_agg.index, hvac_total_load, color="black", linewidth=0.2)
    plt.plot(hour_agg.index, batt_total_load, color="black", linewidth=0.2)
    plt.plot(hour_agg.index, hvac_total_load_og, color="black", linewidth=0.2, linestyle='--')

    plt.fill_between(hour_agg.index, hour_agg['energy_value'], np.zeros_like(hour_agg['energy_value']), color="red", alpha=0.2)
    plt.fill_between(hour_agg.index, hour_agg['energy_value'], hvac_total_load, color="blue", alpha=0.2)
    plt.fill_between(hour_agg.index, hvac_total_load, batt_total_load, color="green", alpha=0.2)
    plt.fill_between(hour_agg.index, hvac_total_load, hvac_total_load_og , hatch='///', facecolor="white", edgecolors="blue", hatch_linewidth=0.5, alpha=0.2)

    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Energy (kWh)', fontsize=12)
    plt.xticks(np.arange(0, 25, 1), fontsize=12)
    plt.xlim(0, 24) 
    plt.title('Energy Load')
    return


@app.cell
def _(np, subset_data):
    hour_agg = subset_data.groupby('hour_id').agg({'energy_value': 'sum'})
    hvac_addition = np.random.normal(500, 100, size=hour_agg.shape[0])
    # smooth the hvac addition data using convolution
    hvac_addition = np.convolve(hvac_addition, np.ones(5)/5, mode='same')
    hvac_addition_og = np.copy(hvac_addition)
    hvac_total_load_og = hour_agg['energy_value'] + hvac_addition_og

    hvac_addition[18:21] = hvac_addition[18:21]/5
    hvac_addition[hvac_addition<0] = 0

    hvac_total_load = hour_agg['energy_value'] + hvac_addition
    hvac_total_load[hvac_total_load >= 5000] = 5000
    hvac_total_load[18:21] = 5000

    # battery addition only on during from 9am to 3pm

    batt_addition = np.zeros_like(hour_agg['energy_value'])
    batt_addition[9:15] = np.random.normal(200, 50, 6)
    batt_addition[batt_addition<0] = 0

    batt_total_load = hvac_total_load + batt_addition
    return (
        batt_addition,
        batt_total_load,
        hour_agg,
        hvac_addition,
        hvac_addition_og,
        hvac_total_load,
        hvac_total_load_og,
    )


@app.cell
def _(json, requests):
    auction_url = "https://i74x56i74zlaxwfbb5keaxt4ne0joclv.lambda-url.us-east-1.on.aws"

    auction_payload = json.dumps({
      "orders": [
        {
          "price": 0.42,
          "quantity": 25.2,
          "order_id": 12321232,
          "flexible": 0
        },
        {
          "price": 0.5,
          "quantity": 25.2,
          "order_id": 12321232,
          "flexible": 0
        },
        {
          "price": 0.38,
          "quantity": 8.2,
          "order_id": 12321233,
          "flexible": 0
        },
        {
          "price": 0.36,
          "quantity": -10.1,
          "order_id": 12321234,
          "flexible": 0
        },
        {
          "price": 0.3,
          "quantity": -10.1,
          "order_id": 12321234,
          "flexible": 0
        },
        {
          "price": 0.25,
          "quantity": -10.1,
          "order_id": 12321234,
          "flexible": 0
        }
      ],
      "market": {
        "interval": 300,
        "units": "kW",
        "price_ceiling": 1000,
        "price_floor": 0
      }
    })

    headers = {
      'Content-Type': 'application/json'
    }

    auction_response = requests.request("POST", auction_url, headers=headers, data=auction_payload)

    print(auction_response.text)
    return auction_payload, auction_response, auction_url, headers


@app.cell
def _(headers, json, requests):
    bidding_url = "https://xmjr5rlkezflcg2f2drq2zu36q0uisvg.lambda-url.us-east-1.on.aws/multibid"

    bidding_payload = json.dumps({
      "battery_1": {
        "device_type": "battery",
        "parameters": {
          "device": {
            "buffer": 0,
            "charging": True,
            "discharging": False,
            "on_grid": True,
            "state_of_charge": 0.5,
            "max_state_of_charge": 1,
            "min_state_of_charge": 0,
            "battery_rating": 7.2,
            "battery_capacity": 10.3,
            "mode": "idle"
          },
          "customer": {
            "current_energy_requirement": 0,
            "K_battery": 1,
            "desired_state_of_charge": 0.4
          },
          "market": {
            "expected_stdev": 0,
            "expected_price": 0.3,
            "min_price": 0,
            "max_price": 2,
            "interval": 60
          }
        }
      },
      "hvac_1": {
        "device_type": "hvac",
        "parameters": {
          "device": {
            "current_temp": 70,
            "last_change_temp": -1,
            "max_temp": 84,
            "min_temp": 65,
            "mode": "auto",
            "state": "on",
            "fan_mode": "auto",
            "fan_state": "on",
            "power_rating": {
              "fan": 0.3,
              "heat": 3.5,
              "cool": 0
            }
          },
          "customer": {
            "K_hvac": 10,
            "desired_temp": 72
          },
          "market": {
            "expected_stdev": 0,
            "expected_price": 0.3,
            "min_price": 0,
            "max_price": 2
          }
        }
      }
    })

    bidding_response = requests.request("POST", bidding_url, headers=headers, data=bidding_payload)

    print(bidding_response.text)
    return bidding_payload, bidding_response, bidding_url


@app.cell
def _(auction_url, bidding_url, headers, json, random, requests):
    # Number of payloads to generate
    num_payloads = 5

    # Function to generate a unique payload
    def generate_battery_payload():
        return {
                "device_type": "battery",
                "parameters": {
                    "device": {
                        "buffer": 0,
                        "charging": True,
                        "discharging": False,
                        "on_grid": True,
                        "state_of_charge": random.uniform(0.2, 0.8),
                        "max_state_of_charge": 1,
                        "min_state_of_charge": 0,
                        "battery_rating": random.uniform(5, 10),
                        "battery_capacity": random.uniform(8, 15),
                        "mode": "idle"
                    },
                    "customer": {
                        "current_energy_requirement": 0,
                        "K_battery": random.uniform(0.5, 2.0),
                        "desired_state_of_charge": random.uniform(0.3, 0.6)
                    },
                    "market": {
                        "expected_stdev": 0,
                        "expected_price": random.uniform(0.2, 0.5),
                        "min_price": 0,
                        "max_price": 2,
                        "interval": 60
                    }
                }
            }
    def generate_hvac_payload():
         return {
                "device_type": "hvac",
                "parameters": {
                    "device": {
                        "current_temp": random.randint(65, 80),
                        "last_change_temp": -1,
                        "max_temp": 84,
                        "min_temp": 65,
                        "mode": "auto",
                        "state": "on",
                        "fan_mode": "auto",
                        "fan_state": "on",
                        "power_rating": {
                            "fan": 0.3,
                            "heat": 3.5,
                            "cool": 0
                        }
                    },
                    "customer": {
                        "K_hvac": random.randint(5, 20),
                        "desired_temp": random.randint(68, 78)
                    },
                    "market": {
                        "expected_stdev": 0,
                        "expected_price": random.uniform(0.2, 0.5),
                        "min_price": 0,
                        "max_price": 2
                    }
                }
            }
    def generate_payloads(num_hvac_payload, num_batt_payload):
        payloads = {}
        for i in range(num_hvac_payload):
            payloads[f"hvac_{i}"] = generate_hvac_payload()
        for i in range(num_batt_payload):
            payloads[f"battery_{i}"] = generate_battery_payload()
        return json.dumps(payloads)
    synth_orders = generate_payloads(10,10)
    synth_bidding_response = requests.request("POST", bidding_url, headers=headers, data=synth_orders)
    synth_bidding_response  = synth_bidding_response.text
    # convert to dictionary
    synth_bidding_response = json.loads(synth_bidding_response)
    order_list = []
    for dev_key, order in synth_bidding_response.items():
        order["order_id"] = dev_key
        order["flexible"] = 0
        order_list.append(order)
    order_list = json.dumps({"orders": order_list})
    synth_auction_response = requests.request("POST", auction_url, headers=headers, data=order_list)

    print(synth_auction_response.text)
    return (
        dev_key,
        generate_battery_payload,
        generate_hvac_payload,
        generate_payloads,
        num_payloads,
        order,
        order_list,
        synth_auction_response,
        synth_bidding_response,
        synth_orders,
    )


if __name__ == "__main__":
    app.run()
