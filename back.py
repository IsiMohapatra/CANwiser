import mdfreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from asammdf import MDF
import re
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import itertools
import ipywidgets as widgets
from IPython.display import display, clear_output
from bs4 import BeautifulSoup
from scipy.interpolate import CubicSpline
import customtkinter as ctk
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from matplotlib.figure import Figure
import base64
from typing import List, Dict, Optional, Any
import io



def load_signals_from_excel(CAN_Matrix_path):
    """Loads Excel data once to avoid repeated file access."""
    try:
        tx_df = pd.read_excel(CAN_Matrix_path, sheet_name='VCU_TX_SIGNALS')
        rx_df = pd.read_excel(CAN_Matrix_path, sheet_name='VCU_RX_SIGNALS')
    except Exception as e:
        print(f"Failed to load sheets: {str(e)}")
        return None, None
    return tx_df, rx_df

def get_signal_periodicity(CAN_Matrix, signal_name):
    """ Extract periodicity for a given signal from the CAN Matrix Excel sheet. """
    if CAN_Matrix is not None and 'Signal Name' in CAN_Matrix.columns and 'Periodicity' in CAN_Matrix.columns:
        signal_row = CAN_Matrix[CAN_Matrix['Signal Name'] == signal_name]
        if not signal_row.empty:
            periodicity_ms = pd.to_numeric(CAN_Matrix['Periodicity'].iloc[0], errors='coerce') if 'Periodicity' in CAN_Matrix.columns else None
            return periodicity_ms / 1000 if pd.notna(periodicity_ms) else None
    return None

def shift_missing_data_dict(missing_data_dict, offset):
    if missing_data_dict is None:
        return None

    shifted_dict = {}

    for signal, gaps in missing_data_dict.items():
        if gaps is None:
            return None  # Return None immediately if any signal has None as gaps

        if not gaps:  # Skip empty lists
            continue

        shifted_gaps = []
        for entry in gaps:
            shifted_entry = {
                "Start": entry["Start"] - offset,
                "End": entry["End"] - offset,
                "Duration": entry["Duration"],  # unchanged
                "Missing Timestamps": [ts + offset for ts in entry["Missing Timestamps"]]
            }
            shifted_gaps.append(shifted_entry)

        shifted_dict[signal] = shifted_gaps

    return shifted_dict if shifted_dict else None

def raw_all_signals_mf4(mf4_file_path, signal_name, raster_precision=3, round_raster=True):
    # Load the MF4 file
    mdf_file = mdfreader.Mdf(mf4_file_path)
    
    # Check if signal exists
    if signal_name not in mdf_file:
        print(f"Signal '{signal_name}' not found.")
        return None

    # Get raw timestamps using master channel
    try:
        master_channel = mdf_file.get_channel_master(signal_name)
        raw_timestamps = np.array(mdf_file.get_channel_data(master_channel))
    except Exception as e:
        print(f"Error retrieving timestamps for '{signal_name}': {e}")
        return None

    if len(raw_timestamps) < 2:
        return None  # Not enough data points

    # Compute raster value (mean diff)
    raster = raw_timestamps[1] - raw_timestamps[0]
    if round_raster:
        raster = round(raster, raster_precision)
    
   
    # Compute time differences
    time_diffs = np.diff(raw_timestamps)
    time_diffs = np.round(time_diffs, 3)

    # Find gaps larger than the expected raster
    gap_indices = np.where(time_diffs > raster)[0]
    if gap_indices.size == 0:
        return None  # No missing data

    # Prepare missing ranges
    missing_ranges = np.column_stack((
        raw_timestamps[gap_indices],
        raw_timestamps[gap_indices + 1],
        time_diffs[gap_indices]
    ))

    # Prepare output dictionary
    missing_data_dict = {
        signal_name: [
            {
                "Start": float(start),
                "End": float(end),
                "Duration": float(duration),
                "Missing Timestamps": list(np.round(np.arange(start + raster, end, raster), raster_precision))
            }
            for start, end, duration in missing_ranges
        ]
    }

    return missing_data_dict

def extract_date_and_time_of_mf4(file_path):
    data = mdfreader.Mdf(file_path)

    if hasattr(data, 'fileMetadata'):
        metadata = str(data.fileMetadata)

        # Match date and time using regex (handles both "Time" and "Temps")
        date_match = re.search(r"Date:\s*([\d/]+)", metadata)
        time_match = re.search(r"(?:Time|Temps)[.:]*\s*([\d:]+\s*[APMapm]*)", metadata)

        if date_match and time_match:
            datetime_str = f"{date_match.group(1)} {time_match.group(1)}"
            print(f"mf4_date_time: {datetime_str}")

            possible_formats = [
                "%m/%d/%Y %I:%M:%S %p",  # Format: 04/17/2025 03:23:41 PM
                "%d/%m/%Y %H:%M:%S",     # Format: 17/04/2025 15:23:41
            ]

            for fmt in possible_formats:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    return dt.strftime("%m/%d/%Y"), dt.strftime("%I:%M:%S.%f %p")
                except ValueError:
                    continue  # try the next format

            print(f"Error parsing date and time: {datetime_str}")
            return None, None
    return None, None

def extract_date_time_of_asc(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()

            if "date" in first_line:
                date_time_str = first_line.split("date ")[1].strip()
                print(f"Extracted datetime string: {date_time_str}")

                # Clean milliseconds part manually (remove junk after dot)
                match = re.match(r'(.*:\d{2})\.(\d{1,6})\s+(am|pm|AM|PM)\s+(\d{4})', date_time_str)
                if match:
                    time_prefix = match.group(1)
                    milliseconds = match.group(2).ljust(6, '0')  # pad to 6 digits
                    meridian = match.group(3)
                    year = match.group(4)
                    cleaned_str = f"{time_prefix}.{milliseconds} {meridian} {year}"

                    dt = datetime.strptime(cleaned_str, '%a %b %d %I:%M:%S.%f %p %Y')
                    return dt.strftime("%m/%d/%Y"), dt.strftime("%I:%M:%S.%f %p")

                # Try fallback format (e.g., 04/17/2025 03:23:27 PM)
                try:
                    dt = datetime.strptime(date_time_str, '%m/%d/%Y %I:%M:%S %p')
                    return dt.strftime("%m/%d/%Y"), dt.strftime("%I:%M:%S.%f %p")
                except ValueError:
                    print(f"Unrecognized datetime format after cleaning: {date_time_str}")

    except Exception as e:
        print(f"Error reading the file: {e}")

    return None, None


def calculate_time_difference_between_mf4_and_asc(mf4_file, asc_file):

    # Extract dates and times from both files
    mf4_date, mf4_time = extract_date_and_time_of_mf4(mf4_file)
    asc_date, asc_time = extract_date_time_of_asc(asc_file)
    # If ASC file date/time extraction failed, return None
    if asc_date is None or asc_time is None:
        print("Failed to extract date/time from ASC file. Skipping processing.")
        return None

    print(mf4_date, asc_date)
    if mf4_date != asc_date:
        print("Dates do not match. Synchronization aborted.")
        return None
    
    # Convert time strings to datetime
    mf4_datetime = datetime.strptime(f"{mf4_date} {mf4_time}", "%m/%d/%Y %I:%M:%S.%f %p")
    asc_datetime = datetime.strptime(f"{asc_date} {asc_time}", "%m/%d/%Y %I:%M:%S.%f %p")
    
    # Compute time difference in seconds
    time_shift = int((mf4_datetime - asc_datetime).total_seconds())

    return time_shift

def get_signals_periodicity(CAN_Matrix, signal_name):
    """Extract periodicity for a given signal from the CAN Matrix Excel sheet."""
    if CAN_Matrix is not None and 'Signal Name' in CAN_Matrix.columns and 'Periodicity' in CAN_Matrix.columns:
        signal_row = CAN_Matrix.loc[CAN_Matrix['Signal Name'] == signal_name, 'Periodicity']
        if not signal_row.empty:
            periodicity_ms = pd.to_numeric(signal_row.iloc[0], errors='coerce')
            return periodicity_ms / 1000 if pd.notna(periodicity_ms) else None  # Convert to seconds
    return None

def round_func(arr):
    return np.round(arr, 1)

def raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, signal_name):
    mdf_file = mdfreader.Mdf(mdf_file_path)
    missing_data_dict={}
    # Load CAN Matrix efficiently
    tx_df, rx_df = load_signals_from_excel(CAN_Matrix_path)
    CAN_Matrix = pd.concat([tx_df, rx_df], ignore_index=True) if tx_df is not None and rx_df is not None else None
    if CAN_Matrix is None:
        return None

    # Process only the requested signal
    if signal_name not in mdf_file:
        print(f"Signal {signal_name} not found in MDF file.")
        return None

    raw_timestamps = np.array(mdf_file.get_channel_data(mdf_file.get_channel_master(signal_name)))

    if len(raw_timestamps) < 2:
        return None  # Not enough data points

    # Try to get periodicity from CAN Matrix
    periodicity = get_signals_periodicity(CAN_Matrix, signal_name)
    
    # Fallback: calculate periodicity from first two valid timestamps
    if periodicity is None:
        periodicity = 2 * round_func(raw_timestamps[1] - raw_timestamps[0])
        print(f"Info: Periodicity for {signal_name} was not found in CAN Matrix. Using fallback: {periodicity} sec")
    else:
        periodicity = 2 * periodicity
    # Compute time differences using NumPy
    time_diffs = round_func(np.diff(raw_timestamps))

    # Find missing timestamps where gaps exceed periodicity
    gap_indices = np.where(time_diffs > periodicity)[0]

    # If no missing timestamps, return None
    if gap_indices.size == 0:
        return None  

    # Store missing ranges per signal
    missing_ranges = np.column_stack((raw_timestamps[gap_indices], raw_timestamps[gap_indices + 1], time_diffs[gap_indices]))
    missing_data_dict = {signal_name: [{"Start": start, "End": end, "Duration": duration, "Missing Timestamps": list(np.arange(start, end, 0.5))}
            for start, end, duration in missing_ranges]}

    return missing_data_dict


def process_all_signals_mdf(mdf_file_path,time_shift):

    # Load the MDF or MF4 file
    mdf_file = mdfreader.Mdf(mdf_file_path)
    all_signals = list(mdf_file.keys())
    synchronized_signals = []
    periodicity = 0.1  # Sampling interval

    for sig in all_signals:
        raw_signal_data = mdf_file.get_channel_data(sig)
        master_channel = mdf_file.get_channel_master(sig)
        raw_timestamps = mdf_file.get_channel_data(master_channel)

        # Convert data to float, ignoring invalid values
        try:
            signal_data = np.array(raw_signal_data, dtype=np.float64)
            timestamps = np.array(raw_timestamps, dtype=np.float64)
        except ValueError:
            print(f"Skipping signal {sig} due to non-numeric values.")
            continue  # Skip this signal

        if len(signal_data) == 0 or len(timestamps) == 0:
            continue  # Skip if no data is found

        # Apply time shift only if provided
        if time_shift is not None:
            valid_indices = np.where(timestamps >= time_shift)[0]
            if len(valid_indices) == 0:
                continue  # Skip storing this signal if no valid timestamps
            timestamps = timestamps[valid_indices]
            signal_data = signal_data[valid_indices]

        # Generate uniform timestamps based on periodicity
        start_time = timestamps[0]
        end_time = timestamps[-1]
        uniform_timestamps = np.arange(start_time, end_time, periodicity)
        idx = np.searchsorted(timestamps, uniform_timestamps, side='right') - 1
        idx = np.clip(idx, 0, len(signal_data) - 1)  # Ensure valid index range

        resampled_signal_data = signal_data[idx] 
        resampled_signal_data = np.round(resampled_signal_data + 0.00001).astype(int)

        synchronized_signals.append({
            'dbc_signal': sig,
            'asw_signal': None,  
            'dbc_data': resampled_signal_data,
            'asw_data': None,
            'dbc_timestamps': uniform_timestamps,
            'asw_timestamps': None
        })

    return synchronized_signals  # Return the updated list

def plot_ini_calculation(ini_region_dict, signals, signal_data_list):
    start_dbc_missing = ini_region_dict.get('start_dbc_missing')
    start_asw_missing = ini_region_dict.get('start_asw_missing')
    end_dbc_missing = ini_region_dict.get('end_dbc_missing')
    end_asw_missing = ini_region_dict.get('end_asw_missing')
    ini_signal_found = ini_region_dict.get('ini_signal_found')

    if not signals:
        print("Signal name list is empty.")
        return

    signal_name = signals[0]
    signal_info = next((item for item in signal_data_list if item['dbc_signal'] == signal_name), None)

    if not signal_info:
        print(f"No data found for DBC Signal: {signal_name}")
        return
    
    dbc_signal = signal_info['dbc_signal']
    asw_signal = signal_info['asw_signal']
    dbc_timestamps = np.array(signal_info['dbc_timestamps'])
    dbc_data = np.array(signal_info['dbc_data'])
    asw_timestamps = np.array(signal_info['asw_timestamps'])
    asw_data = np.array(signal_info['asw_data'])
    
    print("INI Region Dictionary:")
    for key, value in ini_region_dict.items():
        print(f"{key}: {value}")

    # Plotting
    figure = plt.figure(figsize=(10,5))

    # ---- DBC Signal ----
    dbc_before = dbc_timestamps < start_dbc_missing
    dbc_during = (dbc_timestamps >= start_dbc_missing) & (dbc_timestamps <= end_dbc_missing)
    dbc_after = dbc_timestamps > end_dbc_missing

    plt.plot(dbc_timestamps[dbc_before], dbc_data[dbc_before], color='blue', label=f"{dbc_signal}")
    plt.plot(dbc_timestamps[dbc_during], dbc_data[dbc_during], color='blue', linestyle='--')
    plt.plot(dbc_timestamps[dbc_after], dbc_data[dbc_after], color='blue')

    # ---- ASW Signal ----
    asw_before = asw_timestamps < start_asw_missing
    asw_during = (asw_timestamps >= start_asw_missing) & (asw_timestamps <= end_asw_missing)
    asw_after = asw_timestamps > end_asw_missing

    plt.plot(asw_timestamps[asw_before], asw_data[asw_before], color='orange', label=f"{asw_signal}")
    plt.plot(asw_timestamps[asw_during], asw_data[asw_during], color='orange', linestyle='--')
    plt.plot(asw_timestamps[asw_after], asw_data[asw_after], color='orange')

    # Highlighting VCU Sleep Region
    if end_dbc_missing < end_asw_missing:
        vcu_sleep_start = end_dbc_missing
        vcu_sleep_end = end_asw_missing
        plt.axvspan(vcu_sleep_start, vcu_sleep_end, color='skyblue', alpha=0.5, label="VCU Sleep;No valid ASW to comapre")
    else:
        vcu_sleep_start = end_asw_missing
        vcu_sleep_end = end_dbc_missing
        plt.axvspan(vcu_sleep_start, vcu_sleep_end, color='skyblue', alpha=0.5, label="VCU Sleep")
    

    plt.title("DBC and ASW Signal Plot with VCU Sleep Region")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return figure

def process_and_resample_signal(file_path, signal_name, CAN_Matrix_path, synchronized_signals_mdf, round_raster=True, raster_precision=3):
    # Load the CAN Matrix
    tx_df, rx_df = load_signals_from_excel(CAN_Matrix_path)
    CAN_Matrix = pd.concat([tx_df, rx_df], ignore_index=True)
    if CAN_Matrix is None:
        return None, None, None

    # Extract periodicity from the CAN Matrix
    periodicity = 0.01
    try:
        # Load the MDF or MF4 file and retrieve signal data
        mdf_obj = mdfreader.Mdf(file_path)
        signal_data = mdf_obj.get_channel_data(signal_name)

        if signal_data is None or len(signal_data) == 0:
            print(f"Error: No data found for signal '{signal_name}' in {file_path}.")
            return None, None, None

        # Retrieve timestamps for the signal
        master_channel = mdf_obj.get_channel_master(signal_name)
        timestamps = mdf_obj.get_channel_data(master_channel)

        if timestamps is None or len(timestamps) == 0:
            print(f"Error: No timestamp data found for signal '{signal_name}' in {file_path}.")
            return None, None, None

        # If processing MDF signals
        if str(file_path).endswith('.mdf'):
            for entry in synchronized_signals_mdf:
                if entry.get("dbc_signal") == signal_name:
                    signal_data = entry.get("dbc_data")
                    timestamps = entry.get("dbc_timestamps")

                    if periodicity is not None:
                        resample_timestamps = np.arange(timestamps[0], timestamps[-1], periodicity)
                        resampled_signal_data = np.interp(resample_timestamps, timestamps, signal_data)
                        is_interpolated = np.zeros_like(resample_timestamps, dtype=bool)
                    else:
                        resample_timestamps = timestamps
                        resampled_signal_data = signal_data
                        is_interpolated = np.array([])
        # If processing MF4 signals
        else:
            asw_raster_ms = None
            try:
                asw_raster_ms = mdf_obj.get_channel_master(signal_name).Raster
            except AttributeError:
                if len(timestamps) > 1:
                    asw_raster_ms = np.mean(np.diff(timestamps))  # Approximate raster

            if asw_raster_ms is not None:
                asw_raster_ms = round(asw_raster_ms, raster_precision) if round_raster else asw_raster_ms

            if asw_raster_ms is not None:
                resample_timestamps = np.arange(timestamps[0], timestamps[-1], asw_raster_ms)
                resampled_signal_data = np.interp(resample_timestamps, timestamps, signal_data)

                idx = np.searchsorted(timestamps, resample_timestamps)
                is_interpolated = np.ones_like(resample_timestamps, dtype=bool)
                is_interpolated[idx < len(timestamps)] = timestamps[idx[idx < len(timestamps)]] != resample_timestamps[idx < len(timestamps)]
            else:
                resample_timestamps = timestamps
                resampled_signal_data = signal_data
                is_interpolated = np.zeros_like(resample_timestamps, dtype=bool)

        # Round the resampled signal data to the nearest integer if needed
        if not np.issubdtype(resampled_signal_data.dtype, np.integer):
            resampled_signal_data = np.round(resampled_signal_data + 0.00001).astype(int)

        return resample_timestamps, resampled_signal_data, is_interpolated

    except Exception as e:
        print(f"Failed to process signal '{signal_name}': {str(e)}")
        return None, None, None

def find_corresponding_signal(signal_name, CAN_Matrix):
    
    # Check if the input signal matches a DBC signal (in "Signal Name" column)
    dbc_match = CAN_Matrix[CAN_Matrix['Signal Name'] == signal_name]
    if not dbc_match.empty:
        corresponding_signal = dbc_match['ASW interface'].values[0]
        return corresponding_signal, 'DBC'
    
    # Check if the input signal matches an ASW signal (in "ASW Interface" column)
    asw_match = CAN_Matrix[CAN_Matrix['ASW interface'] == signal_name]
    if not asw_match.empty:
        corresponding_signal = asw_match['Signal Name'].values[0]
        return corresponding_signal, 'ASW'
    
    # If no match is found
    print("Error: Signal not found in the CAN matrix.")
    return None, None

def find_state_changes(signal_data, timestamps):
    changes = []
    
    # Iterate through the signal data to detect state changes
    for i in range(1, len(signal_data)):
        if signal_data[i] != signal_data[i - 1]:
            # Capture the state change type and the timestamp
            change_type = 'rising' if signal_data[i] > signal_data[i - 1] else 'falling'
            changes.append((change_type, timestamps[i]))
    
    return changes

def calculate_offset_from_state_changes(dbc_timestamps, dbc_data, asw_timestamps, asw_data):
    tolerance = 0.1  # Tolerance value of 0.1 seconds

    # Find state changes for both DBC and ASW signals
    dbc_changes = find_state_changes(dbc_data, dbc_timestamps)
    asw_changes = find_state_changes(asw_data, asw_timestamps)

    print("\nDBC State Changes:")
    for i in range(len(dbc_changes) - 1):
        change_type, timestamp = dbc_changes[i]
        next_timestamp = dbc_changes[i + 1][1]
        time_diff = next_timestamp - timestamp
        print(f"State Change: {change_type}, Timestamp: {timestamp:.3f}, Time Difference to Next: {time_diff:.3f} seconds")

    print("\nASW State Changes:")
    for i in range(len(asw_changes) - 1):
        change_type, timestamp = asw_changes[i]
        next_timestamp = asw_changes[i + 1][1]
        time_diff = next_timestamp - timestamp
        print(f"State Change: {change_type}, Timestamp: {timestamp:.3f}, Time Difference to Next: {time_diff:.3f} seconds")

    # Case: Only one state change in both signals
    if len(dbc_changes) == 1 and len(asw_changes) == 1:
        dbc_type, dbc_time = dbc_changes[0]
        asw_type, asw_time = asw_changes[0]

        if dbc_type == asw_type:
            offset = asw_time - dbc_time
            print(f"\nSingle state change match found! Offset: {offset:.3f} seconds")
            return offset

    # Case: Two state changes in both signals
    if len(dbc_changes) == 2 and len(asw_changes) == 2:
        dbc_types = [change[0] for change in dbc_changes]
        asw_types = [change[0] for change in asw_changes]
        dbc_time_diffs = dbc_changes[1][1] - dbc_changes[0][1]
        asw_time_diffs = asw_changes[1][1] - asw_changes[0][1]

        if dbc_types == asw_types and abs(dbc_time_diffs - asw_time_diffs) <= tolerance:
            offset = asw_changes[0][1] - dbc_changes[0][1]
            print(f"\nTwo state changes matched! Offset: {offset:.3f} seconds")
            return offset

    # Case: General triplet matching (three or more state changes)
    if len(dbc_changes) >= 3 and len(asw_changes) >= 3:
        for i in range(len(dbc_changes) - 2):
            dbc_triplet = dbc_changes[i:i+3]
            dbc_types = [change[0] for change in dbc_triplet]
            dbc_time_diffs = np.diff([change[1] for change in dbc_triplet])

            print(f"\nConsidering DBC Triplet at index {i}: {dbc_types}")

            for j in range(len(asw_changes) - 2):
                asw_triplet = asw_changes[j:j+3]
                asw_types = [change[0] for change in asw_triplet]
                asw_time_diffs = np.diff([change[1] for change in asw_triplet])

                if dbc_types == asw_types and np.allclose(dbc_time_diffs, asw_time_diffs, atol=tolerance):
                    offset = asw_triplet[0][1] - dbc_triplet[0][1]
                    print(f"\nTriplet match found! Offset: {offset:.3f} seconds")
                    return offset

    print("\nNo matching state change pattern found between DBC and ASW signals.")
    return None


def adjust_signals_with_offset(dbc_data, asw_data, dbc_timestamps, asw_timestamps, offset):

    
    # Shift only the ASW signal
    asw_timestamps_shifted = asw_timestamps - offset  # Shift ASW timestamps backward or forward based on offset
    
    # Determine the common overlapping time range
    start_time = max(dbc_timestamps[0], asw_timestamps_shifted[0])
    end_time = min(dbc_timestamps[-1], asw_timestamps_shifted[-1])
    
    # Get indices for the overlapping range
    dbc_indices = np.where((dbc_timestamps >= start_time) & (dbc_timestamps <= end_time))[0]
    asw_indices = np.where((asw_timestamps_shifted >= start_time) & (asw_timestamps_shifted <= end_time))[0]
    
    # Trim signals to the overlapping range
    dbc_data = dbc_data[dbc_indices]
    dbc_timestamps = dbc_timestamps[dbc_indices]  # DBC timestamps remain unchanged
    asw_data = asw_data[asw_indices]
    asw_timestamps = asw_timestamps_shifted[asw_indices]
    
    return dbc_data, asw_data, dbc_timestamps, asw_timestamps

# Plot Individual Signals
def plot_individual_signals(dbc_data, asw_data, dbc_timestamps, asw_timestamps,
                            dbc_name, asw_name,
                            missing_timestamps_dbc, missing_timestamps_asw):
    fig = plt.figure(figsize=(8, 4))

    # --- DBC Plot ---
    plt.subplot(1, 2, 1)

    # Only try to get missing entries if the dict is not None
    missing_entries_dbc = missing_timestamps_dbc.get(dbc_name, []) if missing_timestamps_dbc else []
    last_idx = 0

    if not missing_entries_dbc:
        plt.plot(dbc_timestamps, dbc_data, color='blue')
    else:
        for entry in missing_entries_dbc:
            start, end = entry["Start"], entry["End"]
            missing_mask = (dbc_timestamps >= start) & (dbc_timestamps <= end)
            missing_indices = np.where(missing_mask)[0]
            if missing_indices.size == 0:
                continue
            first_missing_idx = missing_indices[0]

            if last_idx < first_missing_idx:
                plt.plot(dbc_timestamps[last_idx:first_missing_idx],
                         dbc_data[last_idx:first_missing_idx], color='blue')

            plt.plot(dbc_timestamps[first_missing_idx:missing_indices[-1] + 1],
                     dbc_data[first_missing_idx:missing_indices[-1] + 1],
                     color='blue', linestyle='dashed', linewidth=1)

            last_idx = missing_indices[-1] + 1

        plt.plot(dbc_timestamps[last_idx:], dbc_data[last_idx:], color='blue')

    plt.title(f"DBC: {dbc_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Value")

    # --- ASW Plot ---
    plt.subplot(1, 2, 2)

    missing_entries_asw = missing_timestamps_asw.get(asw_name, []) if missing_timestamps_asw else []
    last_idx = 0

    if not missing_entries_asw:
        plt.plot(asw_timestamps, asw_data, color='orange')
    else:
        for entry in missing_entries_asw:
            start, end = entry["Start"], entry["End"]
            missing_mask = (asw_timestamps >= start) & (asw_timestamps <= end)
            missing_indices = np.where(missing_mask)[0]
            if missing_indices.size == 0:
                continue
            first_missing_idx = missing_indices[0]

            if last_idx < first_missing_idx:
                plt.plot(asw_timestamps[last_idx:first_missing_idx],
                         asw_data[last_idx:first_missing_idx], color='orange')

            plt.plot(asw_timestamps[first_missing_idx:missing_indices[-1] + 1],
                     asw_data[first_missing_idx:missing_indices[-1] + 1],
                     color='orange', linestyle='dashed', linewidth=1)

            last_idx = missing_indices[-1] + 1

        plt.plot(asw_timestamps[last_idx:], asw_data[last_idx:], color='orange')

    plt.title(f"ASW: {asw_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Value")

    plt.tight_layout()
    return fig
# Plot Synchronized Signals
def plot_synchronized_signals(dbc_data, asw_data, dbc_timestamps, asw_timestamps, 
                              dbc_name, asw_name, missing_timestamps_dbc, missing_timestamps_asw):
    # Handle None missing timestamp dictionaries
    if missing_timestamps_dbc is None:
        missing_timestamps_dbc = {}
    if missing_timestamps_asw is None:
        missing_timestamps_asw = {}

    # Determine the start and end times for both signals
    start_time = max(dbc_timestamps[0], asw_timestamps[0])
    end_time = min(dbc_timestamps[-1], asw_timestamps[-1])
    
    # Create masks to select only the overlapping part
    dbc_mask = (dbc_timestamps >= start_time) & (dbc_timestamps <= end_time)
    asw_mask = (asw_timestamps >= start_time) & (asw_timestamps <= end_time)

    # Filter data for overlapping timestamps
    dbc_data_trimmed = dbc_data[dbc_mask]
    asw_data_trimmed = asw_data[asw_mask]
    dbc_timestamps_trimmed = dbc_timestamps[dbc_mask]
    asw_timestamps_trimmed = asw_timestamps[asw_mask]

    fig = plt.figure(figsize=(8,4))

    # --- Plot DBC signal ---
    missing_entries_dbc = missing_timestamps_dbc.get(dbc_name, [])
    if not missing_entries_dbc:  # Plot normally if no missing entries
        plt.plot(dbc_timestamps_trimmed, dbc_data_trimmed, color='blue', alpha=0.7, label=f"DBC: {dbc_name}")
    else:
        last_idx = 0
        for entry in missing_entries_dbc:
            start, end = entry["Start"], entry["End"]
            missing_mask = (dbc_timestamps_trimmed >= start) & (dbc_timestamps_trimmed <= end)
            missing_indices = np.where(missing_mask)[0]
            if missing_indices.size == 0:
                continue
            first_missing_idx = missing_indices[0]

            if last_idx < first_missing_idx:
                plt.plot(dbc_timestamps_trimmed[last_idx:first_missing_idx], 
                         dbc_data_trimmed[last_idx:first_missing_idx], color='blue', alpha=0.7, label=f"DBC: {dbc_name}" if last_idx == 0 else None)

            line, = plt.plot(dbc_timestamps_trimmed[first_missing_idx:missing_indices[-1] + 1], 
                             dbc_data_trimmed[first_missing_idx:missing_indices[-1] + 1], 
                             color='blue',label=f"DBC Missing Data", linewidth=1)
            line.set_dashes([3, 2])
            last_idx = missing_indices[-1] + 1

        if last_idx < len(dbc_data_trimmed):
            plt.plot(dbc_timestamps_trimmed[last_idx:], 
                     dbc_data_trimmed[last_idx:], color='blue',alpha=0.7)

    # --- Plot ASW signal ---
    missing_entries_asw = missing_timestamps_asw.get(asw_name, [])
    if not missing_entries_asw:  # Plot normally if no missing entries
        plt.plot(asw_timestamps_trimmed, asw_data_trimmed, color='orange', alpha=0.7, label=f"ASW: {asw_name}")
    else:
        last_idx = 0
        for entry in missing_entries_asw:
            start, end = entry["Start"], entry["End"]
            missing_mask = (asw_timestamps_trimmed >= start) & (asw_timestamps_trimmed <= end)
            missing_indices = np.where(missing_mask)[0]
            if missing_indices.size == 0:
                continue
            first_missing_idx = missing_indices[0]

            if last_idx < first_missing_idx:
                plt.plot(asw_timestamps_trimmed[last_idx:first_missing_idx], 
                         asw_data_trimmed[last_idx:first_missing_idx], color='orange', alpha=0.7, label=f"ASW: {asw_name}" if last_idx == 0 else None)

            line, = plt.plot(asw_timestamps_trimmed[first_missing_idx:missing_indices[-1] + 1], 
                             asw_data_trimmed[first_missing_idx:missing_indices[-1] + 1], 
                             color='orange', linewidth=1)
            line.set_dashes([3, 2])
            last_idx = missing_indices[-1] + 1

        if last_idx < len(asw_data_trimmed):
            plt.plot(asw_timestamps_trimmed[last_idx:], 
                     asw_data_trimmed[last_idx:], color='orange', alpha=0.7)

    # Plot formatting
    plt.title(f"Synchronized DBC vs. ASW Signals: {dbc_name} vs. {asw_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig

def find_first_nonzero_timestamp(timestamps, data):
    for i, val in enumerate(data):
        if val != 0:
            return timestamps[i]
    return None
 
# Adjust Signals by Aligning Non-Zero Timestamps
def adjust_signals(dbc_data, asw_data, dbc_timestamps, asw_timestamps, dbc_length, asw_length):
    if dbc_data is not None and asw_data is not None:
        # Determine which signal is longer
        if dbc_length > asw_length:
            longest_signal, shortest_signal = dbc_data, asw_data
            longest_timestamps, shortest_timestamps = dbc_timestamps, asw_timestamps
            longest_length = dbc_length
            shortest_length = asw_length
        else:
            longest_signal, shortest_signal = asw_data, dbc_data
            longest_timestamps, shortest_timestamps = asw_timestamps, dbc_timestamps
            longest_length = asw_length
            shortest_length = dbc_length
 
        # Find the first non-zero values in both signals
        delta1 = np.where(longest_timestamps == find_first_nonzero_timestamp(longest_timestamps, longest_signal))[0][0]
        delta2 = np.where(shortest_timestamps == find_first_nonzero_timestamp(shortest_timestamps, shortest_signal))[0][0]

        
        offset = longest_timestamps[delta1] - shortest_timestamps[delta2]
        print(f"Calculated Offset from Timestamp Alignment: {offset} seconds")
        # Shift the longer signal's timestamps by the offset
        shifted_longest_timestamps = longest_timestamps - offset
 
        # Determine the start and end time for both signals after the shift
        start_time = max(shifted_longest_timestamps[0], shortest_timestamps[0])
        end_time = min(shifted_longest_timestamps[-1], shortest_timestamps[-1])
 
        # Trim both signals to the common time range
        mask_longest = (shifted_longest_timestamps >= start_time) & (shifted_longest_timestamps <= end_time)
        mask_shortest = (shortest_timestamps >= start_time) & (shortest_timestamps <= end_time)
 
        # Apply the masks to trim the signals
        longest_signal_trimmed = longest_signal[mask_longest]
        shortest_signal_trimmed = shortest_signal[mask_shortest]
        shifted_longest_timestamps_trimmed = shifted_longest_timestamps[mask_longest]
        shortest_timestamps_trimmed = shortest_timestamps[mask_shortest]
 
        # Ensure both signals are trimmed to the same length
        min_len = min(len(longest_signal_trimmed), len(shortest_signal_trimmed))
        longest_signal_trimmed = longest_signal_trimmed[:min_len]
        shortest_signal_trimmed = shortest_signal_trimmed[:min_len]
        shifted_longest_timestamps_trimmed = shifted_longest_timestamps_trimmed[:min_len]
        shortest_timestamps_trimmed = shortest_timestamps_trimmed[:min_len]
 
        return longest_signal_trimmed, shortest_signal_trimmed, shifted_longest_timestamps_trimmed, shortest_timestamps_trimmed, offset
 
    else:
        print("Error: One or both signals are None.")
        return None, None, None, None,None
 
 
# Main Comparison and Alignment Logic
def calculate_offset_for_synchronization(mdf_file_path, mf4_file_path, CAN_Matrix_path, synchronized_signals_mdf, input_signal):
    # If MDF file is not provided, exit early
    if not mdf_file_path:
        print("Skipping offset calculation: MDF file not found.")
        return None
    
    tx_df, rx_df = load_signals_from_excel(CAN_Matrix_path)
    CAN_Matrix = pd.concat([tx_df, rx_df], ignore_index=True)

    # Find the corresponding signal using the CAN matrix
    corresponding_signal, signal_type = find_corresponding_signal(input_signal, CAN_Matrix)
    if not corresponding_signal:
        print("Error: Corresponding signal not found in the CAN matrix.")
        return None
    
    dbc_signal = input_signal if signal_type == 'DBC' else corresponding_signal
    asw_signal = corresponding_signal if signal_type == 'DBC' else input_signal
    missing_timestamps_dbc=raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, dbc_signal)
    missing_timestamps_asw = raw_all_signals_mf4(mf4_file_path,asw_signal)

    dbc_timestamps, dbc_data, dbc_tracking_flag = process_and_resample_signal(mdf_file_path, dbc_signal, CAN_Matrix_path, synchronized_signals_mdf)
    asw_timestamps, asw_data, asw_tracking_flag = process_and_resample_signal(mf4_file_path, asw_signal, CAN_Matrix_path, synchronized_signals_mdf)
    
    # Check for errors in signal processing
    if dbc_data is None or asw_data is None:
        print("Error in processing the signals.")
        return None
    
    # Plot the individual signals
    fig1= plot_individual_signals(dbc_data, asw_data, dbc_timestamps, asw_timestamps,dbc_signal,asw_signal,missing_timestamps_dbc,missing_timestamps_asw)

    # Synchronization
    offset = None
    
    offset = calculate_offset_from_state_changes(dbc_timestamps, dbc_data, asw_timestamps, asw_data)
    if offset is not None:
            dbc_data, asw_data, dbc_timestamps, asw_timestamps = adjust_signals_with_offset(
                dbc_data, asw_data, dbc_timestamps, asw_timestamps, offset)
    missing_timestamps_asw = shift_missing_data_dict(missing_timestamps_asw, offset)
    # Plot synchronized signals
    fig2 = plot_synchronized_signals(dbc_data, asw_data, dbc_timestamps, asw_timestamps,dbc_signal, asw_signal,
                              missing_timestamps_dbc,missing_timestamps_asw)

    return offset,fig1,fig2

def process_all_signals_mf4(mf4_file_path, offset):
    synchronized_signals_mf4 = []
    mdf_file = mdfreader.Mdf(mf4_file_path)
    
    # Find all available signals
    all_signals = list(mdf_file.keys())
    
    for sig in all_signals:
        signal_data = np.array(mdf_file.get_channel_data(sig))
        master_channel = mdf_file.get_channel_master(sig)
        timestamps = np.array(mdf_file.get_channel_data(master_channel))
        
        # Check if signal_data is numeric
        if not np.issubdtype(signal_data.dtype, np.number):
            print(f"Skipping signal '{sig}' as it contains non-numeric data.")
            continue
        
        # Convert signal_data to integers if necessary
        if not np.issubdtype(signal_data.dtype, np.integer):
            signal_data = np.round(signal_data + 0.00001).astype(int)
        
        if len(signal_data) == 0:
            print(f"No data found for {sig}")
            continue  # Skip if no data is found

        # Shift timestamps only if offset is not None
        if offset is not None:
            timestamps = timestamps - offset
        
        # Append the processed signal data to the list
        synchronized_signals_mf4.append({
            'dbc_signal': None,
            'asw_signal': sig,  
            'dbc_data': None,
            'asw_data': signal_data,
            'dbc_timestamps': None,
            'asw_timestamps': timestamps,
            'dbc_tracking_signal': None,
            'asw_tracking_signal': None
        })
    
    if not synchronized_signals_mf4:
        print("No valid signals found to process.")
    
    return synchronized_signals_mf4  # Return the list instead of a dictionary

def fetch_signals_by_frame_name(CAN_Matrix_path, frame_name):
    df = CAN_Matrix_path

    # Filter based on frame name
    frame_data = df[df['Frame Name'] == frame_name]

    if frame_data.empty:
        print(f"No data found for frame name: {frame_name}")
        return [], [], None, None, [], None, None, None

    dbc_signals = list(frame_data['Signal Name'])
    asw_signals = frame_data['ASW interface'].dropna().tolist() if 'ASW interface' in frame_data.columns else []

    frame_id = frame_data.get('Frame ID', pd.Series([None])).iloc[0]
    periodicity_ms = pd.to_numeric(frame_data.get('Periodicity', pd.Series([None])).iloc[0], errors='coerce')
    periodicity = periodicity_ms / 1000 if pd.notna(periodicity_ms) else None
    transmitter = frame_data.get('Transmitter', pd.Series([None])).iloc[0]
    receiver = frame_data.get('Receiver', pd.Series([None])).iloc[0]

    # Extract gateway signals
    if 'Gateway' in frame_data.columns:
        gateway_signals = list(frame_data.loc[
            (frame_data['Gateway'].notna()) & (frame_data['Gateway'] != "Non applicable"), 'Signal Name'
        ])
    else:
        gateway_signals = []

    normalized_columns = {col.strip().casefold(): col for col in frame_data.columns}
    node = None

    node_columns_map = {
        'hsn1(vehicle can)': 'HSN1',
        'hsn5(chademo can)': 'HSN5',
        'hsn4': 'HSN4',
        'hsn3(epwt)': 'HSN3'
    }

    for normalized_col, short_name in node_columns_map.items():
        if normalized_col in normalized_columns:
            original_col = normalized_columns[normalized_col]
            col_series = frame_data[original_col].astype(str).str.strip().str.casefold()
            if col_series.eq('yes').any():
                node = short_name
                break

    print(f"Signals for Frame '{frame_name}':")
    for dbc_signal, asw_signal in zip(dbc_signals, asw_signals):
        print(f"DBC Signal: {dbc_signal.ljust(40)} ASW Interface: {asw_signal}")

    print(f"Node found for Frame ID {frame_id}: {node}")

    return dbc_signals, asw_signals, frame_id, periodicity, gateway_signals, node, transmitter, receiver



def process_all_signals_of_frame(dbc_signals, asw_signals, offset,mdf_file_path,mf4_file_path,CAN_Matrix_path,synchronized_signals_mdf,synchronized_signals_mf4):
    for dbc_signal, asw_signal in zip(dbc_signals, asw_signals):
        if mdf_file_path is None:
            dbc_timestamps, dbc_data, dbc_tracking_flag = process_and_resample_signal(
                mf4_file_path, dbc_signal, CAN_Matrix_path, synchronized_signals_mdf
            )
            asw_timestamps, asw_data, asw_tracking_flag = process_and_resample_signal(
                mf4_file_path, asw_signal, CAN_Matrix_path, synchronized_signals_mdf
            )
        else:
            dbc_timestamps, dbc_data, dbc_tracking_flag = process_and_resample_signal(
                mdf_file_path, dbc_signal, CAN_Matrix_path, synchronized_signals_mdf
            )
            asw_timestamps, asw_data, asw_tracking_flag = process_and_resample_signal(
                mf4_file_path, asw_signal, CAN_Matrix_path, synchronized_signals_mdf
            )

        # ✅ Check for None before using len()
        if dbc_timestamps is None or dbc_data is None or asw_timestamps is None or asw_data is None:
            print(f"Skipping {dbc_signal} or {asw_signal} due to None values.")
            continue

        # ✅ Fix: Use .size == 0 instead of `not`
        if dbc_timestamps.size == 0 or dbc_data.size == 0 or asw_timestamps.size == 0 or asw_data.size == 0:
            print(f"Skipping {dbc_signal} or {asw_signal} due to empty data.")
            continue

        # ✅ Apply offset only if not None
        if offset is not None:
            asw_timestamps = asw_timestamps - offset

        found = False
        for entry in synchronized_signals_mf4:
            if entry['asw_signal'] == asw_signal:
                entry['dbc_signal'] = dbc_signal
                entry['dbc_data'] = dbc_data
                entry['dbc_timestamps'] = dbc_timestamps
                entry['asw_timestamps'] = asw_timestamps
                entry['asw_data'] = asw_data
                entry['dbc_tracking_signal'] = dbc_tracking_flag
                entry['asw_tracking_signal'] = asw_tracking_flag
                found = True
                break

        if not found:
            print(f"Warning: ASW signal {asw_signal} was not found in synchronized_signals!")

    return synchronized_signals_mf4

def determine_signal_type(CAN_Matrix_path, frame_name):
    tx_signals, rx_signals = load_signals_from_excel(CAN_Matrix_path)
    if tx_signals is None or rx_signals is None:
        print("Error loading signal data.")
        return [], [],[]

    required_cols = {'Transmitter', 'Frame Name', 'Signal Name'}
    if not required_cols.issubset(tx_signals.columns) or not required_cols.issubset(rx_signals.columns):
        print("One or more required columns are missing in the signal data.")
        return [], [],[]

    # Normalize the 'Transmitter' column
    tx_signals['Transmitter'] = tx_signals['Transmitter'].astype(str).str.strip().str.upper()
    rx_signals['Transmitter'] = rx_signals['Transmitter'].astype(str).str.strip().str.upper()

    keywords = {'VCU', 'VCUGW', 'ECM', 'CCM'}

    # Filter TX and RX signals by frame name
    tx_frame_signals = tx_signals[tx_signals['Frame Name'] == frame_name]
    rx_frame_signals = rx_signals[rx_signals['Frame Name'] == frame_name]

    # Combine filtered signals
    frame_signals = pd.concat([tx_frame_signals, rx_frame_signals])

    if frame_signals.empty:
        print(f"No signals found for frame name: {frame_name}")
        return [], [],[]

    transmitter_signals = []
    receiver_signals = []

    for _, row in frame_signals.iterrows():
        signal = row['Signal Name']
        transmitters = row['Transmitter'].split(',')
        # Check if any keyword is substring in any transmitter
        if any(any(keyword in transmitter for transmitter in transmitters) for keyword in keywords):
            transmitter_signals.append(signal)
        else:
            receiver_signals.append(signal)
    
    if len(transmitter_signals) > 0 and len(receiver_signals) == 0:
        signal_type_flag = "Tx"
    else:
        signal_type_flag = "Rx"

    return transmitter_signals, receiver_signals, signal_type_flag

# Function to search for signals in the MF4 file that contain the base search string
def search_signals_in_mf4(mf4_file_path, search_string):
    try:
        mdf = mdfreader.Mdf(mf4_file_path)
        signal_names = mdf.keys()  # Get the list of all signal names in the MF4 file
        matching_signals = [signal for signal in signal_names if search_string in signal]
        return matching_signals, mdf
    except Exception as e:
        print(f"Error reading MF4 file: {str(e)}")
        return [], None
def search_signals_in_mf4(mf4_file_path, search_string):
    try:
        mdf = mdfreader.Mdf(mf4_file_path)
        signal_names = mdf.keys()  # Get the list of all signal names in the MF4 file
        matching_signals = [signal for signal in signal_names if search_string in signal]
        return matching_signals, mdf
    except Exception as e:
        print(f"Error reading MF4 file: {str(e)}")
        return [], None

# Function to find error signals for a given frame name
def find_error_signals_for_frame(frame_name_input, mf4_file_path, frame_id):
    if frame_id is None:
        print(f"No Frame ID found for frame '{frame_name_input}'.")
        return [], None
    frame_id_str = str(frame_id).strip() 
    search_string = f"DFC_st.DFC_Com{frame_id.replace('0x', '') if frame_id.startswith('0x') else frame_id}"
    matching_signals, mdf = search_signals_in_mf4(mf4_file_path, search_string)
    return matching_signals, mdf


# Function to search for matching signals in the DSM sheet and collect activated signals
def search_signals_in_dsm(DTC_Matrix_path, matching_signals):
    activated_signals = []
    try:
        dsm_df = pd.read_excel(DTC_Matrix_path, sheet_name='Matrix', header=5)
        dsm_df['Supplier label'] = dsm_df['Supplier label'].astype(str).str.strip()

        for signal in matching_signals:
            signal = str(signal).strip()  # Convert to string and strip whitespace
            signal_full = signal
            signal_clean = signal[7:] if signal.startswith('DFC_st.') else signal

            signal_info = dsm_df[dsm_df['Supplier label'] == signal_clean]
            if not signal_info.empty:
                activation_status = signal_info['Activation target'].values[0]
                if activation_status == "Activated":
                    activated_signals.append(signal_full)
    except Exception as e:
        print(f"Error reading DSM sheet: {str(e)}")
    return activated_signals


# Function to search the enabled signals in the disable mask list Excel sheet
def search_enabled_signals_in_excel(DTC_Matrix_path, activated_signals, label_column='Label name', cal_column='CAL'):
    enabled_signals = []
    try:
        # Load the Excel sheet
        excel_df = pd.read_excel(DTC_Matrix_path, sheet_name='Disable_Mask')
        excel_df[label_column] = excel_df[label_column].astype(str).str.strip()

        for signal_full in activated_signals:
            signal_full = str(signal_full).strip()  # Ensure it's a string
            signal_clean = signal_full[7:] if signal_full.startswith('DFC_st.') else signal_full
            modified_signal_name = f"DFC_DisblMsk2.{signal_clean}_C"

            signal_info = excel_df[excel_df[label_column] == modified_signal_name]
            if not signal_info.empty:
                cal_value = signal_info[cal_column].values[0]
                if cal_value != 65535:
                    enabled_signals.append(signal_full)
            else:
                print(f"{modified_signal_name} not found in Excel sheet.")
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
    print(enabled_signals)
    return enabled_signals

def normalize_gateway(gateway_value):
    """Normalize gateway string: lowercase, remove spaces and plus signs."""
    if not isinstance(gateway_value, str):
        return ""
    return gateway_value.lower().strip().replace(" ", "").replace("+", "")

def type_of_gateway(receiver_signals, CAN_Matrix_path):
    """Classify each signal into gateway types based on CAN Matrix."""
    tx_signals, rx_signals = load_signals_from_excel(CAN_Matrix_path)
    if tx_signals is None or rx_signals is None:
        print("Error loading CAN Matrix. Exiting analysis.")
        return

    # Combine and deduplicate
    can_matrix = pd.concat([tx_signals, rx_signals], ignore_index=True)

    if 'Gateway' not in can_matrix.columns or 'ASW interface' not in can_matrix.columns:
        print("Required columns are missing.")
        return

    can_matrix['Signal Name'] = can_matrix['Signal Name'].str.strip()
    receiver_signals = [signal.strip() for signal in receiver_signals]
    can_matrix['Gateway'] = can_matrix['Gateway'].fillna(method='ffill')
    filtered_signals = can_matrix[can_matrix['Signal Name'].isin(receiver_signals)]

    if filtered_signals.empty:
        print("No matching signals found.")
        return

    signal_gateway_types = {}

    for _, row in filtered_signals.iterrows():
        signal_name = row['Signal Name']
        gateway = row['Gateway']
        asw_interface = str(row['ASW interface']).lower()
        # Skip if gateway is blank or marked as non-applicable
        if not gateway or gateway.strip().lower() in ['nonapplicable', 'non-applicable', 'n/a']:
            continue
        
        norm_gateway = normalize_gateway(gateway)
       
        # Handle combined used+frame+signal gateway as a special type
        if all(term in norm_gateway for term in ['used', 'frame', 'signal']):
            
            signal_gateway_types[signal_name] = 'used+frame+signal gateway'
            continue

        # Check for frame-related gateway
        elif 'frame' in norm_gateway:
            
            if 'gw signal' in asw_interface:
                signal_gateway_types[signal_name] = 'frame gateway'
            else:
                signal_gateway_types[signal_name] = 'used+frame gateway'

        # Check for signal-related gateway
        elif 'signal' in norm_gateway:
            if 'gw signal' in asw_interface:
                signal_gateway_types[signal_name] = 'signal gateway'
            else:
                signal_gateway_types[signal_name] = 'used+signal gateway'

        # Else: no recognized gateway type
        else:
            continue
    return signal_gateway_types


def identify_multiplexor_signals(CAN_Matrix_path, receiver_signals):
    # Load data
    tx_signals, rx_signals = load_signals_from_excel(CAN_Matrix_path)
    if tx_signals is None or rx_signals is None:
        print("Error loading signal data.")
        return [], [], {}, []

    # Normalize the 'Signal Name' column
    rx_signals['Signal Name'] = rx_signals['Signal Name'].astype(str).str.strip()

    # Filter RX signals to only include those in the receiver_signals list
    filtered_signals = rx_signals[rx_signals['Signal Name'].isin(receiver_signals)]

    if filtered_signals.empty:
        print("No relevant receiving signals found.")
        return [], []

    multiplexor_signals = []
    selector_signals = {}
    
    # Iterate through filtered signals and classify them
    for _, row in filtered_signals.iterrows():
        signal = row['Signal Name']
        multiplexing_group = row.get('Multiplexing/Group', '')

        # Check if the 'Multiplexing/Group' column has a value
        if pd.notna(multiplexing_group) and str(multiplexing_group).strip():
            # Check if this signal is a multiplexor
            if 'Multiplexor' in str(multiplexing_group):
                multiplexor_signals.append(signal)
            else:
                try:
                    # Convert the value to int from hexadecimal and store in the dictionary
                    selector_signals[signal] = int(multiplexing_group.strip(), 16)
                except ValueError:
                    print(f"Warning: Could not convert '{multiplexing_group}' for signal '{signal}' to int.")

    # Get all signals with some value in 'Multiplexing/Group'
    #signals_with_value = multiplexor_signals + list(selector_signals.keys())

    return  multiplexor_signals, selector_signals
def normalize_frame(frame):
    frame = frame.lower().replace('0x', '').replace('h', '')
    return frame 

def extract_basic_receiving_signal(receiver_signals, multiplexor_signals, selector_signals):
    if isinstance(multiplexor_signals, dict):
        multiplexor_signals = list(multiplexor_signals.keys())
    if isinstance(selector_signals, dict):
        selector_signals = list(selector_signals.keys())

    signals_to_remove = set(multiplexor_signals + selector_signals)
    basic_receiving_signals = [signal for signal in receiver_signals if signal not in signals_to_remove]
    
    return basic_receiving_signals

def find_frame_gateway_info(signal_name, frame_id, gateway_sheet, mdf_file_path):
    try:
        # Load frame gateway sheet
        frame_gateway_df = pd.read_excel(gateway_sheet, sheet_name='Frame Gateway', header=1)

        # Validate required columns
        required_columns = ['Frame upstream network', 'Frame downstream network', 'From Network', 'To Network']
        if not all(col in frame_gateway_df.columns for col in required_columns):
            print("Required columns missing in 'Frame Gateway' sheet.")
            return None, None, None

        # Convert frame_id to string
        frame_id_str = str(frame_id).strip()

        # Match frame_id in both upstream and downstream columns
        matched_rows = frame_gateway_df[
            (frame_gateway_df['Frame upstream network'].astype(str).str.strip() == frame_id_str) &
            (frame_gateway_df['Frame downstream network'].astype(str).str.strip() == frame_id_str)
        ]

        if matched_rows.empty:
            print(f"⚠️ Frame ID '{frame_id_str}' not found in both upstream and downstream network columns.")
            return None, None, None

        # Extract network info
        from_network = matched_rows['From Network'].values[0].strip()
        to_network = matched_rows['To Network'].values[0].strip()
        frame_downstream_id = matched_rows['Frame downstream network'].values[0].strip()
        print(f"🔁 Frame ID {frame_id_str} | From: {from_network}, To: {to_network}")

        # Load MDF
        mdf_obj = mdfreader.Mdf(mdf_file_path)

        print(f"\n🔍 Searching for signal: {signal_name}")
        all_occurrences = [key for key in mdf_obj.keys() if signal_name in key]
        if not all_occurrences:
            print(f"  ⚠️ No occurrence found for signal '{signal_name}'.")
            return from_network, to_network, None

        # Map each occurrence to its network
        signal_occurrences = {}
        for key in all_occurrences:
            metadata = mdf_obj.get(key, {})
            network_id = metadata.get("id")
            network_name = None
            if network_id and len(network_id) > 2:
                raw_info = network_id[2][1]
                network_name = raw_info.split()[0]
                
            signal_occurrences[key] = {"network": network_name}

        # Identify downstream signal
        downstream_candidates = [k for k, v in signal_occurrences.items() if v["network"] == to_network]
        if len(downstream_candidates) > 1:
           for key in downstream_candidates:
                metadata_key = mdf_obj.get(key, {})
                network_id_key = metadata_key.get("id")
                network_frame_id = network_id_key[2][0].split()[0] if network_id_key and len(network_id_key) > 2 else None
                if '_' in network_frame_id:
                    network_frame_id =  network_frame_id.split('_')[-1]
               
                norm1 = normalize_frame(frame_downstream_id)
                norm2 = normalize_frame(network_frame_id)
                
                if norm1 == norm2:
                    downstream_key = key
                    break
        else:
            downstream_key = downstream_candidates[0]
        if not downstream_key:
            print("⚠️ Downstream signal not found.")
        else:
            print(f"  🔽 Downstream Signal: {downstream_key}")

        return from_network, to_network, downstream_key

    except Exception as e:
        print(f"❌ Error in find_frame_gateway_info: {e}")
        return None, None, None


  

def find_signal_gateway_info(dbc_signal, frame_id, gateway_sheet, mdf_file_path):
    
    # Read the signal gateway sheet
    signal_gateway_df = pd.read_excel(gateway_sheet, sheet_name='Signal Gateway', header=1)
    
    # Required columns
    required_columns = [
        "From Frame ID", "From Network", "Signal upstream network",
        "To Frame ID", "To Network", "Signal downstream network",
        "Gateway FIDs for fault detection", "Gateway Substitution Calibration"
    ]
    
    # Check for missing columns
    if not all(col in signal_gateway_df.columns for col in required_columns):
        print(f"Missing one or more required columns: {required_columns}")
        return None
    
    # Filter rows based on the given frame_id
    matching_rows = signal_gateway_df[signal_gateway_df["From Frame ID"] == frame_id]
    if matching_rows.empty:
        print(f"No matching frame_id '{frame_id}' found in 'From Frame ID' column.")
        return None

    # Search for the dbc_signal in the filtered rows under 'Signal upstream network'
    signal_match = matching_rows[matching_rows["Signal upstream network"] == dbc_signal]
    
    if signal_match.empty:
        print(f"Signal '{dbc_signal}' not found in 'Signal upstream network' under Frame ID '{frame_id}'.")
        return None
    
    # Extract the value from 'Signal downstream network'
    from_network = signal_match["From Network"].values[0]
    to_network = signal_match["To Network"].values[0]
    to_frame_id = signal_match["To Frame ID"].values[0]
    upstream_signal =   signal_match["Signal upstream network"].values[0]
    downstream_signal = signal_match["Signal downstream network"].values[0]

    # Optional: load MDA data if needed
    mda_data = mdfreader.Mdf(mdf_file_path)
    if dbc_signal == upstream_signal:
        all_occurrences = [key for key in mda_data.keys() if upstream_signal in key]
        upstream_occurence = upstream_signal
        downstream_occurrence = downstream_signal
        if len(all_occurrences) == 1:
            key = all_occurrences[0]
            metadata = mda_data[key]
            network_id = metadata.get("id")
            network_info = network_id[2][1].split()[0] if network_id and len(network_id) > 2 else None  
            if network_info == from_network:
                upstream_occurrence = key
            elif network_info == to_network:
                downstream_occurrence = key
        elif len(all_occurrences) >= 2:
            to_network_occurrences = []
            for key in all_occurrences:
                metadata = mda_data[key]
                network_id = metadata.get("id")
                network_info = network_id[2][1].split()[0] if network_id and len(network_id) > 2 else None
                
                if network_info == from_network:
                    upstream_occurrence = key
                elif network_info == to_network:
                    to_network_occurrences.append(key)
                  
            if len(to_network_occurrences) > 1:
                for keys in to_network_occurrences:
                    metadata_key = mda_data[keys]
                    network_id_key = metadata_key.get("id")
                    network_id_info = network_id_key[2][0].split()[0] if network_id_key and len(network_id_key) > 2 else None
                    if '_' in network_id_info:
                        network_id_info =  network_id_info.split('_')[-1]
                   
                    norm1 = normalize_frame(network_id_info)
                    norm2 = normalize_frame(to_frame_id) 
                    
                    if norm1 == norm2:
                        downstream_occurrence = keys
                        break
            elif len(to_network_occurrences) == 1:
                downstream_occurrence = to_network_occurrences[0]
            else:
                downstream_occurrence = downstream_signal
    return from_network,to_network,upstream_occurrence,downstream_occurrence

def basic_receiving_info(dbc_signal, can_matrix_path):
    tx_df, rx_df = load_signals_from_excel(can_matrix_path)
    CAN_Matrix = pd.concat([tx_df, rx_df], ignore_index=True)

    # Mapping of full column names to simplified names
    node_columns_map = {
        "HSN3(ePWT)": "HSN3",
        "HSN1(Vehicle CAN)": "HSN1",
        "HSN4": "HSN4",
        "HSN5 (CHADEMO CAN)": "HSN5"
    }

    result = CAN_Matrix[CAN_Matrix["Signal Name"] == dbc_signal]

    if not result.empty:
        for full_col, short_col in node_columns_map.items():
            if full_col in result.columns:
                value = str(result[full_col].values[0]).strip().lower()
                if value == "yes":
                    return short_col  # Return simplified column name like "hsn1", "hsn2"
        return None
    else:
        print(f"Signal '{dbc_signal}' not found in CAN Matrix.")
        return None

def status_table(frame_id, dbc_signals, asw_signals, synchronized_signals_mdf, can_matrix_path,
                 gateway_signals, multiplexor_signals, selector_signals,
                 basic_receiving_signals, transmitter_signals, gateway_types,gateway_sheet,mdf_file_path,synchronized_signals_mf4):
    
    unique_dbc_signals = list(set(dbc_signals))
    dbc_signal_set_mdf = {signal_data.get('dbc_signal') for signal_data in synchronized_signals_mdf}
    dbc_signal_set_mf4 = {signal_data.get('dbc_signal') for signal_data in synchronized_signals_mf4}
    asw_signal_set_mf4 = {signal_data.get('asw_signal') for signal_data in synchronized_signals_mf4}
    headers = ["Node","Network Signal", "ASW Signal", "Status", "Frame Id", "Gateway", "Multiplexing", "Type"]
    rows=[]
    signal_info_dict = {}
    
    for dbc_signal in unique_dbc_signals:
        print(dbc_signal)
        if dbc_signal in gateway_signals:
            gateway_type = gateway_types.get(dbc_signal, "").lower()
            
            if gateway_type in ["frame gateway","signal gateway"]:
                asw_signal = "---"
                frame_id = frame_id
                from_net, to_net, downstream_signal = find_frame_gateway_info (dbc_signal,frame_id,gateway_sheet,mdf_file_path)
                node = f"Upstream Node: {from_net}\nDownstream Node: {to_net}"
                multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
                type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
                if gateway_type:
                    gateway_status = gateway_type
                else:
                    gateway_status = "✘"

                if dbc_signal in dbc_signal_set_mdf and downstream_signal in dbc_signal_set_mdf:
                    present = "✅"
                else:
                    present = "❌"
                rows.append([node,dbc_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
                signal_info_dict[dbc_signal] = {
                    "from_node": from_net,
                    "to_node": to_net,
                    "upstream_signal": dbc_signal,
                    "downstream_signal": downstream_signal,
                    "asw_signal" : asw_signal}
            
            elif gateway_type in ["used+frame gateway"]:
                missing_signals =[]
                asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
                if len(asw_signal) > 1:
                    asw_signal = asw_signal[1]
                elif len(asw_signal) == 1:
                    asw_signal = asw_signal[0]
                frame_id = frame_id
                from_net, to_net, downstream_signal = find_frame_gateway_info(dbc_signal,frame_id,gateway_sheet,mdf_file_path)
                node = f"Upstream Node: {from_net}\nDownstream Node: {to_net}"
                network_signal = f"Upstream Signal: {dbc_signal}\nDownstream Signal: {downstream_signal}"
                multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
                type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
                if gateway_type:
                    gateway_status = gateway_type
                else:
                    gateway_status = "✘"

                if dbc_signal not in dbc_signal_set_mf4:
                    missing_signals.append(f"DBC signal '{dbc_signal}'")
                if asw_signal not in asw_signal_set_mf4:
                    missing_signals.append(f"ASW signal '{asw_signal}'")
                if downstream_signal not in dbc_signal_set_mdf:
                    missing_signals.append(f"Downstream signal '{downstream_signal}'")

                if not missing_signals:
                    present = "✅"
                elif len(missing_signals) == 1 and f"Downstream signal '{downstream_signal}'" in missing_signals:
                    present = "✅ \n Gateway Operation cannot be checked" 
                else:
                    present = "❌"
                rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
                signal_info_dict[dbc_signal] = {
                    "from_node": from_net,
                    "to_node": to_net,
                    "upstream_signal": dbc_signal,
                    "downstream_signal": downstream_signal,
                    "asw_signal" : asw_signal}
            
            elif gateway_type in ["used+signal gateway"]:
                missing_signals =[]
                asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
                if len(asw_signal) > 1:
                    asw_signal = asw_signal[1]
                elif len(asw_signal) == 1:
                    asw_signal = asw_signal[0]
                frame_id = frame_id
                from_net,to_net,upstream_signal,down_signal = find_signal_gateway_info(dbc_signal,frame_id,gateway_sheet,mdf_file_path)
                node = f"Upstream Node: {from_net}\nDownstream Node: {to_net}"
                multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
                type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")

                network_signal = f"Upstream Signal: {dbc_signal}\nDownstream Signal: {down_signal}" 
                if gateway_type:
                    gateway_status = gateway_type
                else:
                    gateway_status = "✘"

                if upstream_signal not in dbc_signal_set_mf4:
                    missing_signals.append(f"DBC signal '{dbc_signal}'")
                if asw_signal not in asw_signal_set_mf4:
                    missing_signals.append(f"ASW signal '{asw_signal}'")
                if down_signal not in dbc_signal_set_mdf:
                    missing_signals.append(f"Downstream signal '{down_signal}'")
                if not missing_signals:
                    present = "✅"
                elif len(missing_signals) == 1 and f"Downstream signal '{down_signal}'" in missing_signals:
                    present = "✅ \n Gateway Operation cannot be checked" 
                else:
                    present = "❌"
                rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
                signal_info_dict[dbc_signal] = {
                    "from_node": from_net,
                    "to_node": to_net,
                    "upstream_signal": dbc_signal,
                    "downstream_signal": down_signal,
                    "asw_signal" : asw_signal
                }

            elif gateway_type in ["used+frame+signal gateway"]:
                missing_signals =[]
                asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
                if len(asw_signal) > 1:
                    asw_signal = asw_signal[1]
                elif len(asw_signal) == 1:
                    asw_signal = asw_signal[0]
                frame_id = frame_id 
                from_net,to_net,upstream_signal_signal_gateway,down_signal_signal_gateway = find_signal_gateway_info(dbc_signal,frame_id,gateway_sheet,mdf_file_path)
                node = f"Upstream Node: {from_net}\nDownstream Node: {to_net}"
                from_net_frame, to_net_frame, downstream_signal_frame = find_frame_gateway_info (dbc_signal,frame_id,gateway_sheet,mdf_file_path)
                network_signal = f"Upstream Signal: {dbc_signal}\nDownstream Signal: {down_signal_signal_gateway}"
                multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
                type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
                if gateway_type:
                    gateway_status = gateway_type
                else:
                    gateway_status = "✘"
                if dbc_signal not in dbc_signal_set_mdf:
                    missing_signals.append(f"DBC signal '{dbc_signal}'")
                if asw_signal not in asw_signal_set_mf4:
                    missing_signals.append(f"ASW signal '{asw_signal}'")
                if downstream_signal_frame not in dbc_signal_set_mdf:
                    missing_signals.append(f"Downstream signal '{downstream_signal_frame}'")
                if down_signal_signal_gateway not in dbc_signal_set_mdf:
                    missing_signals.append(f"Downstream signal '{down_signal_signal_gateway}'")
                if not missing_signals:
                    present = "✅"
                elif len(missing_signals) == 1 and f"Downstream signal '{down_signal_signal_gateway}'" in missing_signals:
                    present = "✅ \n Gateway Operation cannot be checked" 
                else:
                    present = "❌"
                rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
                signal_info_dict[dbc_signal] = {
                    "from_node": from_net,
                    "to_node": to_net,
                    "upstream_signal": upstream_signal_signal_gateway,
                    "downstream_signal_signal_gateway": down_signal_signal_gateway,
                    "downstream_signal_frame" : downstream_signal_frame,
                    "asw_signal" : asw_signal
                }
        
        elif dbc_signal in basic_receiving_signals:
            print("basic_reveiving_signal")
            missing_signals =[]
            network_signal = dbc_signal
            frame_id = frame_id
            asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
            if len(asw_signal) > 1:
                asw_signal = asw_signal[1]
            elif len(asw_signal) == 1:
                asw_signal = asw_signal[0]
            if dbc_signal not in dbc_signal_set_mf4:
                missing_signals.append(f"DBC signal '{dbc_signal}'")
            if asw_signal not in asw_signal_set_mf4:
                missing_signals.append(f"ASW signal '{asw_signal}'")
            if not missing_signals:
                present = "✅"
            else:
                present = "❌"
            gateway_status = "✘"
            multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
            type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
            node = basic_receiving_info(dbc_signal,can_matrix_path)
            node = node
            rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
            signal_info_dict[dbc_signal] = {
                    "node": node,
                    "network_signal": dbc_signal,
                    "asw_signal" : asw_signal
                }
        elif dbc_signal in multiplexor_signals or dbc_signal in selector_signals:
            missing_signals = []
            network_signal = dbc_signal
            frame_id = frame_id 
            asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
            if len(asw_signal) > 1:
                asw_signal = asw_signal[1]
            elif len(asw_signal) == 1:
                asw_signal = asw_signal[0]
            gateway_status = "✘"
            multiplexor_signal = multiplexor_signals[0]
            if dbc_signal in multiplexor_signals:
                multiplexing_status = "Multiplexor"
            elif dbc_signal in selector_signals:
                selector_value = selector_signals.get(dbc_signal)
                multiplexing_status = f"{multiplexor_signal} = 0x{selector_value:02X}"
            else:
                multiplexing_status = "✘"
            type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
            if dbc_signal not in dbc_signal_set_mf4:
                missing_signals.append(f"DBC signal '{dbc_signal}'")
            if asw_signal not in asw_signal_set_mf4:
                missing_signals.append(f"ASW signal '{asw_signal}'")
            if not missing_signals:
                present = "✅"
            else:
                present = "❌"
            node = basic_receiving_info(dbc_signal,can_matrix_path)
            node = node
            rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
            signal_info_dict[dbc_signal] = {
                    "node": node,
                    "network_signal": dbc_signal,
                    "asw_signal" : asw_signal
                }
        
        elif dbc_signal in transmitter_signals:
            print("transmitter signal")
            missing_signals = []
            network_signal = dbc_signal
            frame_id = frame_id 
            asw_signal =  [asw_signals[i] for i in range(len(dbc_signals)) if dbc_signals[i] == dbc_signal]
            if len(asw_signal) > 1:
                asw_signal = asw_signal[1]
            elif len(asw_signal) == 1:
                asw_signal = asw_signal[0]
            gateway_status = "✘"
            multiplexing_status = "Multiplexor" if dbc_signal in multiplexor_signals else ("Multiplexed" if dbc_signal in selector_signals else "✘")
            type_status = "Rx" if dbc_signal in basic_receiving_signals else ("Tx" if dbc_signal in transmitter_signals else "✘")
            node = basic_receiving_info(dbc_signal,can_matrix_path) 
            if dbc_signal not in dbc_signal_set_mf4:
                missing_signals.append(f"DBC signal '{dbc_signal}'")
            if asw_signal not in asw_signal_set_mf4:
                missing_signals.append(f"ASW signal '{asw_signal}'")
            if not missing_signals:
                present = "✅"
            else:
                present = "❌"
            rows.append([node,network_signal, asw_signal, present,frame_id, gateway_status, multiplexing_status, type_status])
            signal_info_dict[dbc_signal] = {
                    "node": node,
                    "network_signal": dbc_signal,
                    "asw_signal" : asw_signal
                }

    table_df = pd.DataFrame(rows, columns=headers)
    return table_df,signal_info_dict
    


def search_strings_in_synchronized_signals(synchronized_signals, frame_id, dbc_signal_name, timestamp):
    # Remove '0x' prefix from frame_id if present
    frame_id = frame_id[2:] if frame_id.startswith("0x") else frame_id

    # Initialize the strings to search for
    dfc_string_1 = f"DFC_st.DFC_ComFrbd_{frame_id}h_{dbc_signal_name}"
    dfc_string_2 = f"DFC_st.DFC_ComInvld_{frame_id}h_{dbc_signal_name}"
    found_signals = {}

    for signal_entry in synchronized_signals:
        signal_name = signal_entry.get('asw_signal', '')
        if any(signal_name in dfc_string for dfc_string in [dfc_string_1, dfc_string_2]):
            # Get timestamps and data of the matching signal
            shifted_timestamps = signal_entry.get('asw_timestamps', [])
            shifted_data = signal_entry.get('asw_data', [])

            # Find the closest timestamp index
            closest_idx = np.argmin(np.abs(np.array(shifted_timestamps) - timestamp))
            closest_timestamp = shifted_timestamps[closest_idx]
            signal_value = shifted_data[closest_idx] if closest_idx < len(shifted_data) else None

            # Store only if the value is not 40
            if signal_value is not None and signal_value not in {40, 0, 32}:
                found_signals[signal_name] = (closest_timestamp, signal_value)

    # If any errors were found, return in the required format
    return found_signals if found_signals else None

def check_error_signals_at_timestamp(timestamp, synchronized_signals, enabled_error_signals, frame_id, signal_name_check):
    error_status = []  # List to store error signal details

    # Check for enabled error signals only if they are provided
    if enabled_error_signals:
        for error_signal in enabled_error_signals:
            sync_error_signal = next((sync for sync in synchronized_signals if sync['asw_signal'] == error_signal), None)
            
            if sync_error_signal is None:
                print(f"Error Signal '{error_signal}' is not present in synchronized signals. Skipping.")
                continue

            signal_values = np.array(sync_error_signal['asw_data'])
            signal_timestamps = np.array(sync_error_signal['asw_timestamps'])
            # Find the closest timestamp using numpy
            closest_idx = np.argmin(np.abs(signal_timestamps - timestamp))
            error_value = signal_values[closest_idx]
            
            # Add to error_status if value is not 40, 0, or 32
            if error_value not in {40, 0, 32}:
                error_status.append({
                    'error_signal': error_signal,
                    'error_value': error_value
                })

    # **Always** check for additional error signals, regardless of enabled error signals
    additional_error_values = search_strings_in_synchronized_signals(
        synchronized_signals, frame_id, signal_name_check, timestamp
    )
    if additional_error_values:
        # Append the additional error signals and their values to error_status
        for key, value in additional_error_values.items():
            error_status.append({
                'error_signal': key,
                'error_value': value[1],  # value[1] is the error value at the closest timestamp
            })

    return error_status

def compare_signal1_with_signal2(signal_list, signal_list_structure, enabled_error_signals, frame_id, synchronized_signals_mf4,
                                 mdf_file_path, CAN_Matrix_path, calculated_offset, mf4_file_path, mode):
    
    print("entered")

    
    
    mismatch_info_dict = {}
    ini_region_info_dict = {}
    current_group = None
    grouped_ranges = []
    all_data = []
    mismatch_rows = []
    signal_type = 'dbc_signal'
   
    for signal_name in signal_list:
       
        signal_struc = next((sync for sync in signal_list_structure if sync.get(signal_type) and sync[signal_type].strip() == signal_name.strip()), None)
       
        if signal_struc is None:
            continue

        dbc_signal = signal_struc['dbc_signal']
        asw_signal = signal_struc['asw_signal']
        asw_values = np.array(signal_struc['asw_data'])
        dbc_values = np.array(signal_struc['dbc_data'])
        asw_timestamps = np.array(signal_struc['asw_timestamps'])
        dbc_timestamps = np.array(signal_struc['dbc_timestamps'])
        print(f"dbc signal : {dbc_signal} and asw signal is : {asw_signal}")
        if mode == "Basic":
            missing_timestamps_dbc = raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, dbc_signal)
            missing_timestamps_asw = raw_all_signals_mf4(mf4_file_path, asw_signal)
            missing_timestamps_asw = shift_missing_data_dict(missing_timestamps_asw, calculated_offset)
        else:
            print("else_part")
            missing_timestamps_dbc = raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, dbc_signal)
            missing_timestamps_asw = raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, asw_signal)

        dbc_ranges = []
        asw_ranges = []

        if missing_timestamps_dbc and isinstance(missing_timestamps_dbc, dict):
            dbc_ranges = missing_timestamps_dbc.get(dbc_signal, [])

        if missing_timestamps_asw and isinstance(missing_timestamps_asw, dict):
            asw_ranges = missing_timestamps_asw.get(asw_signal, [])

        # === INI Region check only if both ranges exist ===
        if dbc_ranges and asw_ranges:
            first_entry = dbc_ranges[0]
            start_dbc_missing, end_dbc_missing = first_entry["Start"], first_entry["End"]
            
            for entry in asw_ranges:
                start_asw_missing, end_asw_missing = entry["Start"], entry["End"]
                
                ini_region = None
                overlap_start = max(start_dbc_missing, start_asw_missing)
                overlap_end = min(end_dbc_missing, end_asw_missing)

                if overlap_start < overlap_end:
                    mask = (dbc_timestamps < overlap_start) | (dbc_timestamps > overlap_end)
                    dbc_timestamps = dbc_timestamps[mask]
                    dbc_values = dbc_values[mask]

                    ini_region = end_dbc_missing - end_asw_missing

                    if ini_region < 0:
                        ini_region_info_dict = {
                            'dbc_signal': dbc_signal,
                            'asw_signal': asw_signal,
                            'start_dbc_missing': start_dbc_missing,
                            'start_asw_missing': start_asw_missing,
                            'end_dbc_missing': end_dbc_missing,
                            'end_asw_missing': end_asw_missing
                        }
                    else:
                        frame_id_clean = frame_id[2:]
                        signal = f"{asw_signal}_{frame_id_clean}hIni_C"

                        ini_c_struc = next((sync for sync in synchronized_signals_mf4 if signal in sync.get('asw_signal', '')), None)
                        if ini_c_struc is None:
                            ini_region_info_dict = {
                                'dbc_signal': dbc_signal,
                                'asw_signal': asw_signal,
                                'start_dbc_missing': start_dbc_missing,
                                'start_asw_missing': start_asw_missing,
                                'end_dbc_missing': end_dbc_missing,
                                'end_asw_missing': end_asw_missing,
                                'ini_signal_found': False
                            }
                        else:
                            ini_c_values = np.array(ini_c_struc['asw_data'])
                            ini_c_timestamps = np.array(ini_c_struc['asw_timestamps'])

                            ini_start = end_dbc_missing
                            ini_end = end_asw_missing
                            asw_mask = (asw_timestamps >= ini_start) & (asw_timestamps <= ini_end)
                            asw_region_timestamps = asw_timestamps[asw_mask]

                            ini_region_info_dict = {
                                'dbc_signal': dbc_signal,
                                'asw_signal': asw_signal,
                                'start_dbc_missing': start_dbc_missing,
                                'start_asw_missing': start_asw_missing,
                                'end_dbc_missing': end_dbc_missing,
                                'end_asw_missing': end_asw_missing,
                                'ini_values': []
                            }

                            for i, asw_timestamp in enumerate(asw_region_timestamps):
                                closest_idx = np.argmin(np.abs(ini_c_timestamps - asw_timestamp))
                                ini_c_value = int(ini_c_values[closest_idx])
                                ini_c_time = ini_c_timestamps[closest_idx]

                                ini_region_info_dict['ini_values'].append({'timestamp': float(ini_c_time), 'value': ini_c_value})

        # === Continue with mismatch checking regardless ===
        for i, dbc_timestamp in enumerate(dbc_timestamps):
            dbc_value = int(dbc_values[i])
            closest_idx = np.argmin(np.abs(asw_timestamps - dbc_timestamp))
            asw_timestamp = asw_timestamps[closest_idx]
            asw_value = asw_values[closest_idx]

            all_data.append({
                'dbc_signal': dbc_signal,
                'dbc_timestamp': dbc_timestamp,
                'dbc_value': dbc_value,
                'asw_signal': asw_signal,
                'asw_timestamp': asw_timestamp,
                'asw_value': asw_value,
            })

            if dbc_value != asw_value:
                if mode == "Gateway":
                    error_status = check_error_signals_at_timestamp(dbc_timestamp, synchronized_signals_mf4, enabled_error_signals, frame_id, asw_signal)
                else:  # Basic mode
                    error_status = check_error_signals_at_timestamp(dbc_timestamp, synchronized_signals_mf4, enabled_error_signals, frame_id, dbc_signal)

                if not error_status:
                    continue

                mismatch_info = {
                    'start_timestamp': dbc_timestamp,
                    'end_timestamp': dbc_timestamp,
                    'dbc_signal': dbc_signal,
                    'asw_signal': asw_signal,
                    'dbc_value': dbc_value,
                    'asw_timestamp': asw_timestamp,
                    'asw_value': asw_value,
                    'error_status': error_status
                }

                if current_group is None:
                    current_group = mismatch_info
                else:
                    if error_status == current_group['error_status']:
                        current_group['end_timestamp'] = dbc_timestamp
                    else:
                        if current_group['start_timestamp'] != current_group['end_timestamp']:
                            grouped_ranges.append(current_group)
                        current_group = mismatch_info

                mismatch_rows.append(len(all_data) - 1)

        if current_group and current_group['start_timestamp'] != current_group['end_timestamp']:
            grouped_ranges.append(current_group)

        for idx, group in enumerate(grouped_ranges):
            start_ts = group['start_timestamp']
            end_ts = group['end_timestamp']
            range_str = f"{start_ts} to {end_ts}"
            unique_key = f"{group['dbc_signal']}_{idx}"

            mismatch_info_dict[unique_key] = {
                'start_ts': group['start_timestamp'],
                'end_ts': group['end_timestamp'],
                'range': range_str,
                'dbc_signal': group['dbc_signal'],
                'asw_signal': group['asw_signal'],
                'dbc_value': group['dbc_value'],
                'asw_value': group['asw_value'],
                'activated_error_signals': group['error_status']
            }

    return mismatch_info_dict, ini_region_info_dict


def normalize_value(value):
    """Normalize values by checking if they are off by a factor of 100."""
    if value >= 10000 and value % 100 == 35:  # Likely missing decimal
        return value / 100  # Convert 65535 → 655.35
    return int(value)  

def extract_rcf_cal_values(known_template,mf4_file_path, frame_id,CAN_Matrix_path, asw_signals=None, calibration_signal=None):
    rcf_cal_values = {}

    try:
        # Load the MF4 file
        mdf_obj = mdfreader.Mdf(mf4_file_path)
    except Exception as e:
        print(f"Error loading MF4 file: {str(e)}")
        return rcf_cal_values  # Return empty dict in case of error

    # **Gateway Analysis (Single Calibration Signal)**
    if calibration_signal:
        try:
            if calibration_signal in mdf_obj.keys():
                cal_values = mdf_obj.get_channel_data(calibration_signal)
                if cal_values is not None and len(cal_values) > 0:
                    rcf_cal_values[calibration_signal] = int(cal_values[-1])  # Take the last value
                else:
                    print(f"No data found for Calibration Signal '{calibration_signal}'.")
            else:
                print(f"Calibration Signal '{calibration_signal}' not found in the MF4 file.")
                tx_signals, rx_signals = load_signals_from_excel(CAN_Matrix_path)
                if tx_signals is None or rx_signals is None:
                    print("Error loading signal data.")
                    return [], [], {}, []
                df = pd.concat([tx_signals, rx_signals], ignore_index=True)
                combined_parts = []
                parts = calibration_signal.split("_")
                for part in parts[1:]:
                    if 'GW' in part:
                       combined_parts.append(part.replace('GW', ''))
                       break
                    combined_parts.append(part)
                asw_signal = "_".join(combined_parts)
                matching_row = df[df["Signal Name"] == asw_signal]
                if not matching_row.empty:
                    signal_invalid_value = matching_row["Signal Invalid value"].values[0]
                    signal_offset = matching_row["Signal Offset"].values[0]
                    signal_factor = matching_row["Signal Factor"].values[0]
                    signal_invalid_value = int(signal_invalid_value,16) 
                    signal_invalid_value = (signal_invalid_value * signal_factor) + signal_offset
                    signal_invalid_value = np.round(signal_invalid_value + 0.00001).astype(int)
                    rcf_cal_values[asw_signal] = signal_invalid_value
                else:
                    print(f"ASW Signal '{asw_signal}' not found in the CAN matrix.")
        except Exception as e:
            print(f"Error processing calibration signal '{calibration_signal}': {str(e)}")
    
    onfail_template = known_template.get("onfail", "")
    # **Normal Analysis (Multiple ASW Signals)**
    if asw_signals:
        if isinstance(asw_signals, str):  # Convert single string to a list
            asw_signals = [asw_signals]
        
        for asw_signal in asw_signals:
            try:
                # Construct the RCF signal name
                frame_id = frame_id[2:]
                if onfail_template:
                    rcf_signal_name = onfail_template

                    # Only replace if placeholders exist in the template
                    if "{asw_signal}" in rcf_signal_name:
                        rcf_signal_name = rcf_signal_name.replace("{asw_signal}", asw_signal)
                    if "{frame_id}" in rcf_signal_name:
                        rcf_signal_name = rcf_signal_name.replace("{frame_id}",frame_id)

                    print(f"RCF signal name: {rcf_signal_name}")
                    # Proceed with rcf_signal_name usage...

                else:
                    print(f"No onfail template selected, skipping signal '{asw_signal}'.")
                
                # Check if the constructed RCF signal exists
                if rcf_signal_name in mdf_obj.keys():
                    rcf_values = mdf_obj.get_channel_data(rcf_signal_name)

                    if rcf_values is not None and len(rcf_values) > 0:
                        rcf_cal_value = rcf_values[-1]  # Taking the last value
                        if not np.issubdtype(rcf_values.dtype, np.integer):
                            rcf_cal_value = np.round(rcf_cal_value + 0.00001).astype(int)
                        rcf_cal_values[asw_signal] = rcf_cal_value
                    else:
                        print(f"No data found for Rcf Signal '{rcf_signal_name}'.")
                else:
                    #print(f"Rcf Signal '{rcf_signal_name}' not found in the MF4 file.")
                    tx_signals, rx_signals = load_signals_from_excel(CAN_Matrix_path)
                    if tx_signals is None or rx_signals is None:
                        print("Error loading signal data.")
                        return [], [], {}, []
                    df = pd.concat([tx_signals, rx_signals], ignore_index=True)
                    
                    matching_row = df[df["ASW interface"] == asw_signal]
                    if not matching_row.empty:
                        signal_invalid_value = matching_row["Signal Invalid Value"].values[0]
                        signal_invalid_value = int(signal_invalid_value, 16)
                        print(f"ASW Signal '{asw_signal}' found in CAN matrix. Signal Invalid Value: {signal_invalid_value}")
                        rcf_cal_values[asw_signal] = signal_invalid_value
                    else:
                        print(f"ASW Signal '{asw_signal}' not found in the CAN matrix.")
            except Exception as e:
                print(f"Error processing ASW signal '{asw_signal}': {str(e)}")

    return rcf_cal_values

def process_temporary_error(start_ts, end_ts, asw_signal, list_of_signals):
    # Step 1: Search for ASW signal in the list of signals
    
    matched_signal = next((sig for sig in list_of_signals if sig['asw_signal'] == asw_signal), None)
    if not matched_signal:
        print(f"ASW Signal '{asw_signal}' not found in the signal list.")
        return None, None  # Return None values for consistency
    timestamps = np.array(matched_signal['asw_timestamps'])
    values = np.array(matched_signal['asw_data'])

    # Step 2: Find the last valid value before the error range
    valid_mask = timestamps < start_ts
    last_valid_value = None  # Default to None in case no valid value is found

    if np.any(valid_mask):
        last_valid_idx = np.where(valid_mask)[0][-1]  # Last valid index before error
        last_valid_value = values[last_valid_idx]
        print(f"Last valid ASW value before error: {last_valid_value} at timestamp {timestamps[last_valid_idx]:.3f}")
    else:
        print(f"No valid ASW values found before {start_ts:.3f}")
        return None, None  # Return None if no valid values are found

    # Step 3: Extract ASW values within the error range
    error_mask = (timestamps >= start_ts) & (timestamps <= end_ts)
    error_values = values[error_mask]
    error_timestamps = timestamps[error_mask]

    if error_values.size == 0:
        print(f"No ASW data found during the error range {start_ts:.3f} to {end_ts:.3f}")
        return {asw_signal: last_valid_value}, True  # Consider frozen if no data is found

    # Step 4: Check for mismatches
    mismatch_mask = error_values != last_valid_value
    if np.any(mismatch_mask):
        mismatch_values = error_values[mismatch_mask]
        mismatch_timestamps = error_timestamps[mismatch_mask]

        mismatch_duration = mismatch_timestamps[-1] - mismatch_timestamps[0]  # Duration of the mismatch

        if mismatch_duration < 0.005:
            print(f"For the error range, {asw_signal} is frozen to its last valid value (mismatch duration < 5 ms).")
            return {asw_signal: last_valid_value}, True
        else:
            print(f"Mismatches detected for ASW Signal '{asw_signal}' during the error range:")
            for ts, val in zip(mismatch_timestamps, mismatch_values):
                print(f"  Mismatch at {ts:.3f} s → {asw_signal} :: ASW Value: {val}, Expected last_valid_value : {last_valid_value}")
            return {asw_signal: last_valid_value}, False  # Indicate mismatches found
    else:
        print(f"For the error range, {asw_signal} is frozen to its last valid value.")
        return {asw_signal: last_valid_value}, True


def compare_asw_to_rcf(asw_signal, rcf_value, start_ts, end_ts, synchronized_signals, mode):
    # Select the correct signal based on mode
    if mode == "gateway":
        matched_signal = next((sig for sig in synchronized_signals if sig['asw_signal'] == asw_signal), None)
    else:  # Normal analysis mode
        matched_signal = next((sig for sig in synchronized_signals if sig['asw_signal'] == asw_signal), None)

    # Check if signal was found
    if not matched_signal:
        print(f"ASW Signal '{asw_signal}' not found in the signal list.")
        return {}, False  # Return an empty dictionary and False

    # Extract timestamps and values
    timestamps = np.array(matched_signal.get('asw_timestamps' if mode == "gateway" else 'asw_timestamps', []))
    values = np.array(matched_signal.get('asw_data' if mode == "gateway" else 'asw_data', []))
    # Ensure timestamps exist
    if timestamps.size == 0:
        print(f"No timestamps found for {asw_signal}.")
        return {}, False  # Return an empty dictionary and False
    
    # Apply time range mask
    error_mask = (timestamps >= start_ts) & (timestamps <= end_ts)
    error_values = values[error_mask]
    error_timestamps = timestamps[error_mask]
    # Extract and normalize RCF value
    original_rcf_value = list(rcf_value.values())[0]
    
    if np.all(error_values == original_rcf_value):
        print(f"All values for {asw_signal} during the error range match the RCF value.")
        return {asw_signal: original_rcf_value}, True

    else:
        print(f"Not all values are equal to the RCF value during the error range.")
        mismatch_indices = np.where(error_values != int(original_rcf_value))[0]  # Extract valid indices
    
    if len(mismatch_indices) == 1:
        print(f"Only one mismatch for {asw_signal} during the error range, considering it as a match.")
        return {asw_signal: original_rcf_value}, True
    # If no mismatches or exactly one mismatch, consider the array as matched
    elif len(mismatch_indices) == 0:
        print(f"All values for {asw_signal} during the error range match the RCF value.")
        return {asw_signal: original_rcf_value}, True
    else:
        print(f"Not all values are equal to the RCF value during the error range.")
        for idx in mismatch_indices:
            print(f"Mismatch at timestamp {error_timestamps[idx]:.6f}: "
                  f"{asw_signal} value = {error_values[idx]}, expected RCF value = {original_rcf_value}")
    
    return {asw_signal: original_rcf_value}, False  # Return False if mismatches exist

def process_error(known_template,start_ts, end_ts, frame_name, frame_id, error_signal, asw_signal,list_of_signals, is_gateway_analysis, DTC_matrix_path,mf4_file_path,
                  CAN_Matrix_path,gateway_sheet,fid_mapping_sheet,fid_check=None,gateway_substitution=None):
    print(f"Processing Error Signal: {error_signal}")
    rcf_cal_values = {}
    error_status = False  # Default to False unless proven otherwise

    try:
        inhibit_df = pd.read_excel(fid_mapping_sheet, sheet_name='DFC_Inhibit_HEX_Overview')
        inhibit_df['DFC Name'] = inhibit_df['DFC Name'].astype(str).str.strip()

        cleaned_signal = error_signal[7:] if error_signal.startswith("DFC_st.") else error_signal
        matching_rows = inhibit_df[inhibit_df['DFC Name'] == cleaned_signal]
        if matching_rows.empty:
            print(f"No matching entries found for {cleaned_signal} in 'DFC Name' column.")
            return {}, error_status

        if is_gateway_analysis:
            print("Gateway Signal Analysis")
            if fid_check is None:
                print("fid_check is None, skipping Gateway analysis.")
                return {}, error_status

            fid_check_string = str(fid_check)
            filtered_rows = matching_rows[matching_rows['Inhibit'].astype(str).str.contains(fid_check_string, na=False)]

            if not filtered_rows.empty:
                print(f"Gateway Analysis: FId Check String '{fid_check_string}' found.")
                calibration_signal = gateway_substitution
                rcf_cal_values = extract_rcf_cal_values(known_template,mf4_file_path, frame_id, CAN_Matrix_path,None, calibration_signal)
                # Capture the return status
                normalized_rcf_values,error_status = compare_asw_to_rcf(asw_signal, rcf_cal_values, start_ts, end_ts, list_of_signals, mode="gateway")

                return normalized_rcf_values, error_status

            print(f"Gateway Analysis: FId Check String '{fid_check_string}' NOT found.")
            return {}, error_status


        frame_based_rcf_string = f"FId_bInh_Pdu_{frame_name}_Rcf"
        filtered_rows = matching_rows[matching_rows['Inhibit'].astype(str).str.contains(frame_based_rcf_string, na=False)]

        if not filtered_rows.empty:
            print(f"Frame-based RCF found for {error_signal}. Processing all ASW signals of the frame:")
            rcf_cal_values = extract_rcf_cal_values(known_template,mf4_file_path, frame_id,CAN_Matrix_path, asw_signal, None)
            normalized_rcf_values,error_status = compare_asw_to_rcf(asw_signal, rcf_cal_values, start_ts, end_ts, list_of_signals, mode="normal")

            return normalized_rcf_values, error_status

        print(f"No frame-based RCF found for {cleaned_signal} in frame {frame_name}.")
        frame_id_clean = frame_id[2:]
        frame_index = error_signal.find(frame_id_clean)
        if frame_index != -1:
            dbc_signal_search = error_signal[frame_index + len(frame_id):]
            if dbc_signal_search.startswith('h_'):
                dbc_signal_search = dbc_signal_search[2:]
        else:
            dbc_signal_search = None  

        try:
            signal_based_rcf_string = f"FId_bInh_Sig{frame_id[2:]}h_{asw_signal}_Rcf"
            filtered_rows = matching_rows[matching_rows['Inhibit'].astype(str).str.contains(signal_based_rcf_string, na=False)]

            if not filtered_rows.empty:
                print(f"Signal-based RCF found for {error_signal}. Processing particular ASW signals of the frame:")
                rcf_cal_values = extract_rcf_cal_values(known_template,mf4_file_path, frame_id,CAN_Matrix_path, asw_signal, None)
                normalized_rcf_values,error_status = compare_asw_to_rcf(asw_signal,rcf_cal_values, start_ts, end_ts, list_of_signals, mode="normal")

                return normalized_rcf_values, error_status

            print(f"Signal-based RCF is not found.")
            normalized_rcf_values,error_status = process_temporary_error(start_ts, end_ts, asw_signal, list_of_signals)
            return normalized_rcf_values, error_status  

        except ValueError:
            print(f"DBC Signal '{dbc_signal_search}' not found in the list.")

    except Exception as e:
        print(f"Error reading Inhibit Overview Excel sheet: {str(e)}")

    return {}, error_status



def analyze_error_for_range(known_template,start_ts, end_ts, frame_name_input, frame_id,signal_list_info, error_signals, error_values, 
                            asw_signal, synchronized_signals_mf4,DTC_matrix_path,mf4_file_path,CAN_Matrix_path,
                            gateway_sheet,fid_mapping_sheet,fid_check,gateway_substitution,flag):
    error_status_mapping = {
        0: "Not tested before",
        40: "No error",
        71: "Temporary error",
        111: "Temporary error",
        239: "Temporary error",
        168: "Healed error",
        219: "Permanent error",
        251: "Permanent error",
        252: "Temporary healed error"
    }
    
    error_results = {}  # Dictionary to store error signals and their statuses
    last_valid_value = None  # This will be updated from process_temporary_error
    rcf_value = None  
    
    # Identify error statuses for each signal
    for error_signal, error_value in zip(error_signals, error_values):
        error_status = error_status_mapping.get(error_value, "Unknown status")
        error_results[error_signal] = error_status
    
    # Print out all the error statuses
    for error_signal, error_status in error_results.items():
        print(f"Error signal: {error_signal}, Error status: {error_status}")
    
    # Process errors based on type
    for error_signal, error_status in error_results.items():
        print(f"Processing {error_status}")  # Print which error status is being processed

        if error_status == "Temporary error":
            if flag:
                last_valid_value,error_status = process_temporary_error(start_ts=start_ts, end_ts=end_ts, asw_signal=asw_signal, 
                                                           list_of_signals=signal_list_info)  
            else:
                last_valid_value,error_status = process_temporary_error(start_ts=start_ts, end_ts=end_ts, asw_signal=asw_signal, 
                                                           list_of_signals=synchronized_signals_mf4)   
            return error_results, last_valid_value,error_status  # Return last valid value from process_temporary_error
        
        elif error_status in ["Permanent error", "Temporary healed error"]:
            if flag:
                rcf_value,error_status = process_error(known_template=known_template,start_ts=start_ts, end_ts=end_ts, frame_name=frame_name_input, frame_id=frame_id, 
                                       error_signal=error_signal, asw_signal=asw_signal,list_of_signals=signal_list_info,
                                       is_gateway_analysis=True,DTC_matrix_path = DTC_matrix_path,mf4_file_path= mf4_file_path,CAN_Matrix_path=CAN_Matrix_path,
                                       gateway_sheet = gateway_sheet,fid_mapping_sheet= fid_mapping_sheet,
                                       fid_check=fid_check,gateway_substitution=gateway_substitution)
            else:
                rcf_value,error_status = process_error(known_template=known_template,start_ts=start_ts, end_ts=end_ts, frame_name=frame_name_input, frame_id=frame_id, 
                                       error_signal=error_signal, asw_signal=asw_signal, list_of_signals=synchronized_signals_mf4,
                                        is_gateway_analysis=False,DTC_matrix_path = DTC_matrix_path,mf4_file_path= mf4_file_path,CAN_Matrix_path=CAN_Matrix_path,
                                        gateway_sheet = gateway_sheet,fid_mapping_sheet= fid_mapping_sheet,
                                        fid_check=None, gateway_substitution=None)
            return error_results, rcf_value,error_status 

    return error_results, None ,None # Default return if no errors



def plot_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, signal_data_list, mismatch_info_dict,synchronized_signals_mf4,mode):
    """
    Plots DBC and ASW signals within the provided mismatch time range (with buffer),
    highlighting error periods and generating zoomed-in plots for each error region.
    """
    # Add buffer for better visualization
    plot_start = mismatch_start_time - 5
    plot_end = mismatch_end_time + 5
    figures = []
    # Extract all mismatch entries for the given ASW signal
    error_periods = []
    error_timestamps = []
    error_values = []
    for key, value in mismatch_info_dict.items():
        if value["asw_signal"] == asw_signal:
            for error_signal in value["activated_error_signals"]:
                error_name = error_signal["error_signal"]
                error_value = error_signal.get("error_value", None)
                error_type = value["error_type"].get(error_name, "Unknown Error Type")

                compare_error_value_dict = value.get("compare_error_value", {})
                
                # Ensure compare_error_value_dict is not empty and has a valid value
                if not compare_error_value_dict:
                    continue  # Skip this iteration if compare_error_value_dict is empty
                
                compare_error_value = next(iter(compare_error_value_dict.values()), None)

                if compare_error_value is not None:
                    try:
                        compare_error_value = float(compare_error_value)
                    except ValueError:
                        continue  # Skip this iteration if the value cannot be converted to float

                    error_periods.append({
                        "start_ts": value["start_ts"],
                        "end_ts": value["end_ts"],
                        "error_name": error_name,
                        "error_value": error_value,
                        "error_type": error_type,
                        "compare_error_value": compare_error_value
                    })
                    error_timestamps.append(value["start_ts"])
                    error_values.append(error_value)

    # Count unique error names
    unique_error_names = set(error["error_name"] for error in error_periods)

    if not error_periods:
        print(f"No mismatch data found for ASW Signal: {asw_signal}")
        return

    # Find the relevant signal data
    signal_info = next((item for item in signal_data_list if item['asw_signal'] == asw_signal), None)
    
    if not signal_info:
        print(f"No data found for ASW Signal: {asw_signal}")
        return

    # Extract timestamps and values for DBC and ASW signals
    dbc_signal_name = signal_info['dbc_signal']
    asw_signal_name = signal_info['asw_signal']
    dbc_timestamps = np.array(signal_info['dbc_timestamps'])
    dbc_values = np.array(signal_info['dbc_data'])
    asw_timestamps = np.array(signal_info['asw_timestamps'])
    asw_values = np.array(signal_info['asw_data'])

    # ---- INTERPOLATE MISSING DBC DATA ----
    error_signal_data = {}
    for error_name in unique_error_names:
        for sync_signal in synchronized_signals_mf4:
            if sync_signal["asw_signal"] == error_name:
                timestamps = np.array(sync_signal["asw_timestamps"])
                values = np.array(sync_signal["asw_data"])
                
                for error in error_periods:
                    mask = (timestamps >= error["start_ts"]) & (timestamps <= error["end_ts"])
                    if np.any(mask):
                        if error_name not in error_signal_data:
                            error_signal_data[error_name] = {"timestamps": [], "values": []}
                        error_signal_data[error_name]["timestamps"].extend(timestamps[mask])
                        error_signal_data[error_name]["values"].extend(values[mask])

    # ---- PLOT 1: FULL SIGNAL COMPARISON WITH ERROR SIGNAL ----
    fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    if mode == "Gateway":
        axes[0].plot(dbc_timestamps, dbc_values, label=f"Upstream Signal : {dbc_signal_name}", 
                 color='blue', linestyle='-', linewidth=1.5, alpha=0.8)
        axes[0].plot(asw_timestamps, asw_values, label=f"Downstream Signal : {asw_signal_name}", 
                 color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    elif mode == "Basic":
        axes[0].plot(dbc_timestamps, dbc_values, label=f"DBC Signal : {dbc_signal_name}", 
                 color='blue', linestyle='-', linewidth=1.5, alpha=0.8)
        axes[0].plot(asw_timestamps, asw_values, label=f"ASW Interface : {asw_signal_name}", 
                 color='red', linestyle='-', linewidth=1.5, alpha=0.7)

    for error in error_periods:
        if error["start_ts"] >= plot_start and error["end_ts"] <= plot_end:
            axes[0].axvspan(error["start_ts"], error["end_ts"], color='red', alpha=0.3)

    axes[0].set_ylabel("Signal Value")
    axes[0].set_title(f"{dbc_signal_name} vs {asw_signal_name}")
    axes[0].legend(loc='upper right', prop={'size': 6})
    axes[0].grid(True)
    color_cycle = itertools.cycle(plt.cm.get_cmap("tab10").colors) 
    # Second Strip: Error Signal Values
    if error_signal_data:
        for error_name, data in error_signal_data.items():
            color = next(color_cycle)

            # Search for the error_name in synchronized_signals
            sync_signal = next((signal for signal in synchronized_signals_mf4 if signal["asw_signal"] == error_name), None)
    
            if sync_signal:
                # Extract timestamps and values for the matching asw_signal
                timestamps = np.array(sync_signal["asw_timestamps"])
                values = np.array(sync_signal["asw_data"])
        
                # Extract the time range for the error signal
                mask = (timestamps >= plot_start) & (timestamps <= plot_end)
        
                # Plot the values within the range of the error as a line plot
                axes[1].plot(timestamps[mask], values[mask], label=f"{error_name} (Error Signal)", color=color, alpha=0.7)


    axes[1].set_ylabel("Error Signal Value")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Error Signal Over Time")
    axes[1].legend(loc='lower right', prop={'size': 6}) 
    axes[1].grid(True)

    plt.xlim(plot_start, plot_end)
    plt.tight_layout()
    figures.append(fig)

    # ---- PLOT 2: ZOOMED-IN ERROR REGIONS ----
    for idx, error in enumerate(error_periods):
        error_start = error["start_ts"] - 3
        error_end = error["end_ts"] + 3
        compare_error_value = error["compare_error_value"]
        error_value = error["error_value"]
       
        # Skip plotting if compare_error_value is None
        if compare_error_value is None:
            continue  # Skip this iteration if compare_error_value is None
        
        dbc_error_mask = (dbc_timestamps >= error_start) & (dbc_timestamps <= error_end)
        asw_error_mask = (asw_timestamps >= error_start) & (asw_timestamps <= error_end)

        dbc_error_timestamps, dbc_error_values = dbc_timestamps[dbc_error_mask], dbc_values[dbc_error_mask]
        asw_error_timestamps, asw_error_values = asw_timestamps[asw_error_mask], asw_values[asw_error_mask]

        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

        ax1 = plt.subplot(gs[0])
        ax1.plot(dbc_error_timestamps, dbc_error_values, label=f"{dbc_signal_name}", linestyle='-', color='blue', linewidth=1.5)
        ax1.axvspan(error["start_ts"], error["end_ts"], color='red', alpha=0.3)
        ax1.set_ylabel("DBC Value")
        ax1.set_xticklabels([])
        ax1.legend(loc='upper right', fontsize=5, frameon=False)
        ax1.grid(True)

        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(asw_error_timestamps, asw_error_values, label=f"{asw_signal_name} ", linestyle='--', color='red', linewidth=1.5)
        if mode == "Gateway":
            ax2.axhline(y=compare_error_value, color='green', linestyle='--', linewidth=1.5, label=f"Gateway Substitution Value: {compare_error_value}")
        elif mode == "Basic":
            ax2.axhline(y=compare_error_value, color='green', linestyle='--', linewidth=1.5, label=f"Replacement Value: {compare_error_value}")
        ax2.axvspan(error["start_ts"], error["end_ts"], color='red', alpha=0.3)
        ax2.set_ylabel("ASW Value")
        ax2.set_xticklabels([])
        ax2.legend(loc='upper right', fontsize=5, frameon=False)
        ax2.grid(True)
        
        error_data_start = error["start_ts"] - 3
        error_data_end = error["end_ts"]
        error_signal_info = next((item for item in synchronized_signals_mf4 if item['asw_signal'] == error['error_name']), None)
        if error_signal_info is None:
            print(f"No matching signal data found for error name: {error['error_name']}")
            continue  # Skip this iteration if no matching signal data is found
        error_timestamps = np.array(error_signal_info['asw_timestamps'])
        error_value = np.array(error_signal_info['asw_data'])
        error_data_mask = (error_timestamps >= error_data_start) & (error_timestamps <= error_data_end)
        error_data_timestamps, error_data_values = error_timestamps[error_data_mask], error_value[error_data_mask]
        if 252 in error_data_values:
            error_data_start = error["start_ts"] - 1  # Change start time
            error_data_end = error["end_ts"] + 3  # Change end time
            plot_color = 'green'  # Change plot color to green
            error_data_mask = (error_timestamps >= error_data_start) & (error_timestamps <= error_data_end)
            error_data_timestamps, error_data_values = error_timestamps[error_data_mask], error_value[error_data_mask]
        else:
            plot_color = 'red'  # Default color

        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(error_data_timestamps, error_data_values, label=f"{error['error_name']} ", linestyle='--', color=plot_color, linewidth=1.5)
        ax3.axvspan(error["start_ts"], error["end_ts"], color=plot_color, alpha=0.3)

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Error Value")
        if plot_color == 'green':
            ax3.set_yticks([168, 251])
        ax3.legend(loc='upper right', fontsize=5, frameon=False)
        ax3.grid(True)

        plt.suptitle(f"Zoomed-In Region {idx+1}: {error['start_ts']} s - {error['end_ts']} s\n{error['error_name']} ({error['error_type']})", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        figures.append(fig)

    return figures



def analysis_basic_receiving_signal(basic_receiving_signals, synchronized_signals, enabled_error_signals, frame_id, frame_name_input,synchronized_signals_mf4,
                                    DTC_matrix_path,mf4_file_path,CAN_Matrix_path,gateway_sheet,fid_mapping_sheet,mdf_file_path,calculated_offset,known_template):
    
    error_info_dict,ini_region_dict = compare_signal1_with_signal2(basic_receiving_signals, synchronized_signals, enabled_error_signals, frame_id,synchronized_signals_mf4,
                                                   mdf_file_path,CAN_Matrix_path,calculated_offset,mf4_file_path,mode="Basic")
    error_status_list = []
    figures = []
    if not error_info_dict:
        if ini_region_dict:
            figures = plot_ini_calculation(ini_region_dict,basic_receiving_signals,synchronized_signals) 
        else:
            return error_status_list,figures 
                 
    else:
        
        # Dictionary to store complete mismatch time periods
        mismatch_time_ranges = {}
        error_status_list = []
        # Group mismatches by ASW signal
        grouped_mismatches = {}
        for range_key, group in error_info_dict.items():
            asw_signal = group['asw_signal']
            if asw_signal not in grouped_mismatches:
                grouped_mismatches[asw_signal] = []
            grouped_mismatches[asw_signal].append(group)

        # Iterate over each ASW signal, print mismatches first, then summary
        for asw_signal, mismatches in grouped_mismatches.items():
            mismatch_time_ranges[asw_signal] = {'start_ts': float('inf'), 'end_ts': float('-inf')}

            # Print all possible mismatch ranges first
            for group in mismatches:
                print(f"Range: {group['range']}")
                print(f"  DBC Signal: {group['dbc_signal']}")
                print(f"  ASW Signal: {group['asw_signal']}")
                print(f"  DBC Value: {group['dbc_value']}")
                print(f"  ASW Value: {group['asw_value']}")
                print(f"  Error Signals in this range:")
                
                for error in group['activated_error_signals']:
                    print(f"    - {error['error_signal']} , Value: {error['error_value']})")
                
                print("Analyzing error signals for the range")
                error_type, compare_error_value,error_status  = analyze_error_for_range(known_template,
                    start_ts=group['start_ts'], end_ts=group['end_ts'], frame_name_input=frame_name_input,
                    frame_id=frame_id,signal_list_info = synchronized_signals_mf4,
                    error_signals=[err['error_signal'] for err in group['activated_error_signals']],
                    error_values=[err['error_value'] for err in group['activated_error_signals']],
                    asw_signal=group['asw_signal'],synchronized_signals_mf4=synchronized_signals_mf4,
                    DTC_matrix_path=DTC_matrix_path,mf4_file_path=mf4_file_path,CAN_Matrix_path=CAN_Matrix_path,
                    gateway_sheet = gateway_sheet , fid_mapping_sheet=fid_mapping_sheet,
                    fid_check=None, gateway_substitution=None, flag=False)
                error_status_list.append(error_status)

                group['error_type'] = error_type
                group['compare_error_value'] = compare_error_value
                group['error_status'] = error_status
                print("-" * 60)

                # Track the complete mismatch time period for each ASW signal
                mismatch_time_ranges[asw_signal]['start_ts'] = min(mismatch_time_ranges[asw_signal]['start_ts'], group['start_ts'])
                mismatch_time_ranges[asw_signal]['end_ts'] = max(mismatch_time_ranges[asw_signal]['end_ts'], group['end_ts'])
                mismatch_start_time = mismatch_time_ranges[asw_signal]['start_ts']
                mismatch_end_time = mismatch_time_ranges[asw_signal]['end_ts']

            # Print the summary after all mismatch ranges for this ASW signal
            print(f"\nASW Signal: {asw_signal}, Mismatch Start Time: {mismatch_start_time}, Mismatch End Time: {mismatch_end_time}")
            figures = plot_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, synchronized_signals, error_info_dict,synchronized_signals_mf4,mode="Basic")
            print("=" * 80)
    
    return error_status_list,figures


def multiplexor_signals_analysis(multiplexor_signals, selector_signals, synchronized_signals, enabled_error_signals,frame_id,frame_name_input,
                                 synchronized_signals_mf4,DTC_matrix_path,mf4_file_path,CAN_Matrix_path,gateway_sheet,fid_mapping_sheet,
                                 mdf_file_path,calculated_offset,known_template):
    signal_info = []  # List to store signal info

    for multiplexor_signal in multiplexor_signals:
        for signal_data in synchronized_signals:
            if signal_data.get('dbc_signal') == multiplexor_signal:
                dbc_data = signal_data.get('dbc_data')
                asw_signal = signal_data.get('asw_signal')
                asw_data = signal_data.get('asw_data')
                dbc_timestamps = signal_data.get('dbc_timestamps')
                asw_timestamps = signal_data.get('asw_timestamps')

        if dbc_data is not None and asw_data is not None:
            grouped_ranges = []
            current_group = None

            for i, dbc_time in enumerate(dbc_timestamps):
                closest_asw_index = np.argmin(np.abs(np.array(asw_timestamps) - dbc_time))
                closest_asw_time = asw_timestamps[closest_asw_index]
                closest_asw_value = asw_data[closest_asw_index]

                if dbc_data[i] != closest_asw_value:
                    mismatch = {
                        'start_timestamp': dbc_time,
                        'end_timestamp': dbc_time,
                        'dbc_value': dbc_data[i],
                        'asw_signal': asw_signal,
                        'asw_timestamp': closest_asw_time,
                        'asw_value': closest_asw_value,
                        'selector_data': []
                    }

                    if current_group is None:
                        current_group = mismatch
                    else:
                        if (dbc_data[i] == current_group['dbc_value'] and closest_asw_value == current_group['asw_value'] and dbc_time - current_group['end_timestamp'] < 0.5): 
                            current_group['end_timestamp'] = dbc_time
                        else:
                            if current_group['start_timestamp'] != current_group['end_timestamp']:
                                grouped_ranges.append(current_group)
                            current_group = mismatch

            if current_group and current_group['start_timestamp'] != current_group['end_timestamp']:
                grouped_ranges.append(current_group)
            
            if grouped_ranges:
                print(f"\n⚠️ Mismatches found for multiplexor signal '{multiplexor_signal}' and ASW signal '{asw_signal}':")
                
                for group in grouped_ranges:
                    filtered_selector_timestamps = []
                    filtered_selector_values = []
                    filtered_selector_asw_timestamps = []
                    filtered_selector_asw_values = []
                    print(f"  - Range: {group['start_timestamp']} to {group['end_timestamp']}")
                    print(f"    - DBC Signal Mismatch: DBC Value: {group['dbc_value']}")
                    print(f"    - ASW Signal Mismatch: ASW Value: {group['asw_value']}")

                    matched_selector = [signal for signal, value in selector_signals.items() if value == group['asw_value']]
                    if matched_selector:
                        selector_signal = matched_selector[0]
                        print(f"    - Multiplexed Message: {selector_signal}")
                        
                        # Ensure the selector data is reset and properly filtered for the current range
                        selector_asw_signal = None
                        selector_timestamps = []
                        selector_values = []
                        selector_asw_timestamps = []
                        selector_asw_values = []
                        
                        for signal_data in synchronized_signals:
                            if signal_data.get('dbc_signal') == selector_signal:
                                selector_asw_signal = signal_data.get('asw_signal')
                                selector_timestamps = signal_data.get('dbc_timestamps')
                                selector_values = signal_data.get('dbc_data')
                                selector_asw_timestamps = signal_data.get('asw_timestamps')
                                selector_asw_values = signal_data.get('asw_data')
                       
                        
                        # Step 1: Filter the selector timestamps that fall within the error range (start_time to end_time)
                        
                        filtered_selector_timestamps = [t for t in selector_timestamps if group['start_timestamp'] <= t <= group['end_timestamp']]
                        filtered_selector_values = [selector_values[i] for i, t in enumerate(selector_timestamps) if group['start_timestamp'] <= t <= group['end_timestamp']]

                        # Step 2: Filter timestamps for selector ASW signal that fall within the error range
                        filtered_selector_asw_timestamps = [t for t in selector_asw_timestamps if group['start_timestamp'] <= t <= group['end_timestamp']]
                        filtered_selector_asw_values = [selector_asw_values[i] for i, t in enumerate(selector_asw_timestamps) if group['start_timestamp'] <= t <= group['end_timestamp']]
                        
                        range_info = {
                            'dbc_signal': selector_signal,
                            'asw_signal': selector_asw_signal,
                            'dbc_data': filtered_selector_values,
                            'asw_data': filtered_selector_asw_values,
                            'dbc_timestamps': filtered_selector_timestamps,
                            'asw_timestamps': filtered_selector_asw_timestamps
                        }
                        signal_info.append(range_info)
                        selector_signal_names = [selector_signal]
                        mismatch_info_dict,ini_region_dict = compare_signal1_with_signal2(selector_signal_names, [range_info], enabled_error_signals,frame_id,
                                                                                          synchronized_signals_mf4,mdf_file_path,CAN_Matrix_path,
                                                                                          calculated_offset,mf4_file_path,mode= "Basic")
                        error_status_list = []
                        # Check if mismatch_info_dict is empty
                        if not mismatch_info_dict:
                            print(f"✅ Communication Successful for {selector_signal} and {selector_asw_signal}.DFC_st.DFC_ComInvld_592h_CALIBRE_IBATT is set and the replacemnet reaction is the last valid value \n")
                            error_status_list.append(True)
                            
                        else:
                            print("Analysis of the ranges:")

                            mismatch_time_ranges = {}
                            grouped_mismatches = {}

                            for range_key, group in mismatch_info_dict.items():
                                asw_signal = group['asw_signal']
                                if asw_signal not in grouped_mismatches:
                                    grouped_mismatches[asw_signal] = []
                                grouped_mismatches[asw_signal].append(group)

                            for asw_signal, mismatches in grouped_mismatches.items():
                                mismatch_time_ranges[asw_signal] = {'start_ts': float('inf'), 'end_ts': float('-inf')}

                                for group in mismatches:
                                    print(f"Range: {group['range']}")
                                    print(f"  DBC Signal: {group['dbc_signal']}")
                                    print(f"  ASW Signal: {group['asw_signal']}")
                                    print(f"  DBC Value: {group['dbc_value']}")
                                    print(f"  ASW Value: {group['asw_value']}")
                                    print(f"  Error Signals in this range:")

                                    for error in group['activated_error_signals']:
                                        print(f"    - {error['error_signal']} , Value: {error['error_value']}")
                                    
                                    print("Analyzing error signals for the range")
                                    error_type, compare_error_value,error_status  = analyze_error_for_range(known_template,start_ts=group['start_ts'], end_ts=group['end_ts'], frame_name_input=frame_name_input,
                                                                                                frame_id=frame_id,signal_list_info = synchronized_signals_mf4,
                                                                                                error_signals=[err['error_signal'] for err in group['activated_error_signals']],
                                                                                                error_values=[err['error_value'] for err in group['activated_error_signals']],
                                                                                                asw_signal=group['asw_signal'],synchronized_signals_mf4=synchronized_signals_mf4,
                                                                                                DTC_matrix_path=DTC_matrix_path,mf4_file_path=mf4_file_path,
                                                                                                CAN_Matrix_path=CAN_Matrix_path,gateway_sheet=gateway_sheet,fid_mapping_sheet=fid_mapping_sheet,
                                                                                                fid_check=None, gateway_substitution=None, flag=False)
                                    error_status_list.append(error_status)
                                    group['error_type'] = error_type
                                    group['compare_error_value'] = compare_error_value
                                    group['error_status'] = error_status
                                    print("-" * 60)

                                    # Track the complete mismatch time period for each ASW signal
                                    mismatch_time_ranges[asw_signal]['start_ts'] = min(mismatch_time_ranges[asw_signal]['start_ts'], group['start_ts'])
                                    mismatch_time_ranges[asw_signal]['end_ts'] = max(mismatch_time_ranges[asw_signal]['end_ts'], group['end_ts'])
                                    mismatch_start_time = mismatch_time_ranges[asw_signal]['start_ts']
                                    mismatch_end_time = mismatch_time_ranges[asw_signal]['end_ts']

                                # Print the summary after all mismatch ranges for this ASW signal
                                print(f"\nASW Signal: {asw_signal}, Mismatch Start Time: {mismatch_start_time}, Mismatch End Time: {mismatch_end_time}")
                                figures = plot_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, synchronized_signals, mismatch_info_dict,
                                                               synchronized_signals_mf4,mode="Basic")
                                print("=" * 80)
                    else:
                        print(f"    - No matching selector found for ASW Value: {group['asw_value']} \n")
            else:
                print(f"✅ Multiplexor signal '{multiplexor_signal}' matches ASW signal '{asw_signal}' perfectly.")
            
    return error_status_list,figures

def signal_gateway_analysis (upstream,downstream,gateway_sheet,synchronized_signals_mdf,
                             enabled_error_signals, frame_id,synchronized_signals_mf4,
                                mdf_file_path,CAN_Matrix_path,calculated_offset,mf4_file_path,DTC_Matrix_path,frame_name_input,
                                fid_mapping_sheet,time_shift,known_template):
    
    mda_data = mdfreader.Mdf(mdf_file_path)
    signal_gateway_df = pd.read_excel(gateway_sheet, sheet_name='Signal Gateway', header=1)
    signal_data_list =[]
    if upstream and downstream:
        matched_rows = signal_gateway_df[signal_gateway_df["Signal upstream network"] == upstream]
        if matched_rows.empty:
            print(f"No match found for upstream signal: {upstream}")
            return None
        downstream_from_sheet = matched_rows["Signal downstream network"].values[0]
        if downstream_from_sheet in downstream:
            fid = matched_rows["Gateway FIDs for fault detection"].values[0]
            gateway = matched_rows["Gateway Substitution Calibration"].values[0]
        print(f"{upstream}  ----->  {downstream} -----> {fid} -------> {gateway}")
        if upstream in mda_data.keys():
            upstream_data = np.array(mda_data.get_channel_data(upstream))
            master_channel = mda_data.get_channel_master(upstream)  # Get master channel (time channel)
            upstream_timestamps = np.array(mda_data.get_channel_data(master_channel))
            valid_indices = np.where(upstream_timestamps >= time_shift)[0]
            if len(valid_indices) == 0:
                return None  # Skip storing this signal if no valid timestamps
        
            upstream_timestamps = upstream_timestamps[valid_indices]
            upstream_data = upstream_data[valid_indices]
            periodicity_upstream = round(np.min(np.diff(upstream_timestamps)),2)
            interpolated_upstream_timestamps = np.arange(upstream_timestamps[0], upstream_timestamps[-1], periodicity_upstream)
            upstream_data = np.interp(interpolated_upstream_timestamps, upstream_timestamps, upstream_data)
            if not np.issubdtype(upstream_data.dtype, np.integer):
                upstream_data = np.round(upstream_data + 0.00001).astype(int)

        if downstream in mda_data.keys():
            downstream_data = np.array(mda_data.get_channel_data(downstream))
            master_channel = mda_data.get_channel_master(downstream)  # Get master channel (time channel)
            downstream_timestamps = np.array(mda_data.get_channel_data(master_channel))
            valid_indices = np.where(downstream_timestamps >= time_shift)[0]
            if len(valid_indices) == 0:
                return None  # Skip storing this signal if no valid timestamps
        
            downstream_timestamps = downstream_timestamps[valid_indices]
            downstream_data = downstream_data[valid_indices]
            periodicity_downstream = round(np.min(np.diff(downstream_timestamps)),2)
            interpolated_downstream_timestamps = np.arange(downstream_timestamps[0], downstream_timestamps[-1], periodicity_downstream)
            downstream_data = np.interp(interpolated_downstream_timestamps, downstream_timestamps, downstream_data)
            if not np.issubdtype(downstream_data.dtype, np.integer):
                downstream_data = np.round(downstream_data + 0.00001).astype(int)
        
        signal_info = {'dbc_signal': upstream,
                        'asw_signal': downstream,
                        'dbc_data': upstream_data,
                        'asw_data': downstream_data,
                        'dbc_timestamps': interpolated_upstream_timestamps,
                        'asw_timestamps': interpolated_downstream_timestamps,
                        'fid_check': fid,
                        'gateway_substitution': gateway}
        signal_data_list.append(signal_info)
    else:
        print(f"Communication Failed for {upstream} and {downstream}. ")  # if either upstream_signal or downstream_signal is not present. 
        return [False]
    mismatch_info_dict = {}
    ini_region_dict ={}
    gateway_signals = [upstream]
    error_status_list = []
    mismatch_info_dict,ini_region_dict = compare_signal1_with_signal2(gateway_signals, signal_data_list,
                                                                       enabled_error_signals, frame_id,synchronized_signals_mf4,
                                                          mdf_file_path,CAN_Matrix_path,calculated_offset,mf4_file_path,mode="Gateway")
    print(mismatch_info_dict)
    print(ini_region_dict)
    if not mismatch_info_dict:
        figures = plot_ini_calculation(ini_region_dict,gateway_signals,signal_data_list)
       
    else:
        # Dictionary to store complete mismatch time periods
        mismatch_time_ranges = {}
        error_status_list = []
        # Group mismatches by ASW signal
        grouped_mismatches = {}
        for range_key, group in mismatch_info_dict.items():
            asw_signal = group['asw_signal']
            if asw_signal not in grouped_mismatches:
                grouped_mismatches[asw_signal] = []
            grouped_mismatches[asw_signal].append(group)

        # Iterate over each ASW signal, print mismatches first, then summary
        for asw_signal, mismatches in grouped_mismatches.items():
            mismatch_time_ranges[asw_signal] = {'start_ts': float('inf'), 'end_ts': float('-inf')}

            # Print all possible mismatch ranges first
            for group in mismatches:
                print(f"Range: {group['range']}")
                print(f"  DBC Signal: {group['dbc_signal']}")
                print(f"  ASW Signal: {group['asw_signal']}")
                print(f"  DBC Value: {group['dbc_value']}")
                print(f"  ASW Value: {group['asw_value']}")
                print(f"  Error Signals in this range:")
                    
                for error in group['activated_error_signals']:
                    print(f"    - {error['error_signal']} , Value: {error['error_value']})")

                # Extract correct fid_check and gateway_substitution
                signal_info = next((item for item in signal_data_list if item['asw_signal'] == group['asw_signal']), None)
                fid_check = signal_info['fid_check'] if signal_info else None
                gateway_substitution = signal_info['gateway_substitution'] if signal_info else None
                    
                print("Analyzing error signals for the range")
                error_type, compare_error_value,error_status  = analyze_error_for_range(known_template,
                        start_ts=group['start_ts'], end_ts=group['end_ts'], frame_name_input=frame_name_input,
                        frame_id=frame_id,signal_list_info = signal_data_list,
                        error_signals=[err['error_signal'] for err in group['activated_error_signals']],
                        error_values=[err['error_value'] for err in group['activated_error_signals']],
                        asw_signal=group['asw_signal'],synchronized_signals_mf4=synchronized_signals_mf4,
                        DTC_matrix_path=DTC_Matrix_path,mf4_file_path=mf4_file_path,CAN_Matrix_path=CAN_Matrix_path,
                        gateway_sheet=gateway_sheet,fid_mapping_sheet=fid_mapping_sheet,
                        fid_check=fid_check, gateway_substitution=gateway_substitution, flag=True)
                error_status_list.append(error_status)

                group['error_type'] = error_type
                group['compare_error_value'] = compare_error_value
                group['error_status'] = error_status
                print("-" * 60)

                # Track the complete mismatch time period for each ASW signal
                mismatch_time_ranges[asw_signal]['start_ts'] = min(mismatch_time_ranges[asw_signal]['start_ts'], group['start_ts'])
                mismatch_time_ranges[asw_signal]['end_ts'] = max(mismatch_time_ranges[asw_signal]['end_ts'], group['end_ts'])
                mismatch_start_time = mismatch_time_ranges[asw_signal]['start_ts']
                mismatch_end_time = mismatch_time_ranges[asw_signal]['end_ts']

            # Print the summary after all mismatch ranges for this ASW signal
            print(f"\nASW Signal: {asw_signal}, Mismatch Start Time: {mismatch_start_time}, Mismatch End Time: {mismatch_end_time}")
            figures = plot_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, signal_data_list, 
                                           mismatch_info_dict,synchronized_signals_mf4,mode="Gateway")
            print("=" * 80)
    return error_status_list,figures


def plot_communication_sucessful(dbc_signal, asw_signal, synchronized_signals, mode,offset,mdf_file_path=None, CAN_Matrix_path=None):
    """Function to plot DBC and ASW signals in both basic and signal_gateway modes."""
    figure = plt.figure(figsize=(10, 5))
    dbc_data, dbc_timestamps = None, None
    asw_data, asw_timestamps = None, None
    
    if mode == "basic":
        for signal_data in synchronized_signals:
            if signal_data.get('dbc_signal') == dbc_signal:
                dbc_timestamps = np.array(signal_data.get('dbc_timestamps', []))
                asw_timestamps = np.array(signal_data.get('asw_timestamps', []))
                dbc_data = np.array(signal_data.get('dbc_data', []))
                asw_data = np.array(signal_data.get('asw_data', []))
                break

    elif mode in ("signal_gateway", "frame_gateway"):
        for signal_data in synchronized_signals:
            current_signal = signal_data.get('dbc_signal')
            
            if current_signal == dbc_signal:
                dbc_timestamps = np.array(signal_data.get('dbc_timestamps', []))
                dbc_data = np.array(signal_data.get('dbc_data', []))
            
            if current_signal == asw_signal:
                asw_timestamps = np.array(signal_data.get('dbc_timestamps', []))
                asw_data = np.array(signal_data.get('dbc_data', []))
   
    # Plotting if data is available
    if dbc_timestamps is not None and dbc_data is not None:
        plt.plot(dbc_timestamps, dbc_data, label=f"DBC Signal: {dbc_signal}", linestyle='-', color='red')

    if asw_timestamps is not None and asw_data is not None:
        plt.plot(asw_timestamps, asw_data, label=f"ASW Signal: {asw_signal}", linestyle='-', color='blue')
    
    if mode == "frame_gateway" and mdf_file_path and CAN_Matrix_path:
        try:
            missing_timestamps_dbc = raw_all_signals_mdf(mdf_file_path, CAN_Matrix_path, dbc_signal)
            if missing_timestamps_dbc and isinstance(missing_timestamps_dbc, dict):
                dbc_ranges = missing_timestamps_dbc.get(dbc_signal, [])
                if dbc_ranges:
                    first_entry = dbc_ranges[0]
                    start_dbc_missing = first_entry["Start"] - offset
                    end_dbc_missing = first_entry["End"] - offset
                    plt.axvspan(start_dbc_missing, end_dbc_missing, color='yellow', alpha=0.3)
        except Exception as e:
            print(f"Warning: Failed to fetch or highlight missing DBC range - {e}")

    plt.xlabel("Timestamps")
    plt.ylabel("Signal Values")
    plt.title(f"Synchronization of {dbc_signal} and {asw_signal}")
    plt.legend()
    plt.grid(True)
    return figure

    
def quality_status_of_signal(frame_id, dbc_signals, asw_signals, synchronized_signals_mf4, enabled_error_signals, gateway_signals,
                             signal_info_dict,gateway_types,synchronized_signals_mdf,basic_receiving_signal,multiplexor_signals,selector_signals,transmitter_signals,
                             frame_name,dtc_matrix_path,mf4_file_path,can_matrix_path,gateway_sheet_path,fid_mapping_path,mdf_file_path,offset,time_shift,known_templates):
    """Function to create an interactive summary table with only necessary columns (5 columns)."""
    display(f"📡 Quality Status of Signals for Frame {frame_id}")

    headers = [
        "Node", "Network Signal", "ASW Signal" , "Frame Gateway\nStatus",
        "Basic Communication\n Status", "Basic Replacement\nStatus",  "Signal Gateway\nStatus",
        "Signal Gateway \nSubstitution Status"]
    row = []
    color_dict = {}
    figure_dict = {}
    unique_dbc_signals = list(set(dbc_signals))
    frame_gateway_status = "--"
    error_status_list_basic = None
    error_status_list_gateway = None
    figures = None
    gx_figures = None
    for dbc_signal in unique_dbc_signals:
        
        if dbc_signal in gateway_signals:
            gateway_type = gateway_types.get(dbc_signal, "").strip().lower()
            
            if gateway_type in ["frame gateway"]:
                signal_info = signal_info_dict.get(dbc_signal, {})
                from_node = signal_info.get("from_node", "N/A")
                to_node = signal_info.get("to_node", "N/A")
                upstream_signal = signal_info.get("upstream_signal", dbc_signal)
                downstream_signal = signal_info.get("downstream_signal", "N/A")

                upstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == upstream_signal), None)
                downstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == downstream_signal), None)

                if upstream_data and downstream_data:
                    up_values = np.array(upstream_data.get("dbc_data", []))
                    down_values = np.array(downstream_data.get("dbc_data", []))

                    if len(up_values) == len(down_values) and np.array_equal(up_values, down_values):
                        frame_gateway_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝" if not enabled_error_signals else "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    else:
                        frame_gateway_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"
                node = f"Upstream Node: {from_node}\nDownstream Node: {to_node}"
                network_signal = f"Upstream Signal: {upstream_signal}\nDownstream Signal: {downstream_signal}"
                asw_signal = "--"
                basic_comm_status = "--"
                basic_replacement_status = "--"
                signal_gateway_status = "--"
                signal_gateway_sub_status = "--"
                
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status,basic_replacement_status, signal_gateway_status,signal_gateway_sub_status])
            
            elif gateway_type in ["used+frame gateway"]:
                signal_info = signal_info_dict.get(dbc_signal, {})
                from_node = signal_info.get("from_node", "N/A")
                to_node = signal_info.get("to_node", "N/A")
                upstream_signal = signal_info.get("upstream_signal", dbc_signal)
                downstream_signal = signal_info.get("downstream_signal", "N/A")
                asw_signal = signal_info.get("asw_signal", "N/A")
                
                upstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == upstream_signal), None)
                downstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == downstream_signal), None)
                dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
                #asw_data = next((s for s in synchronized_signals_mf4 if s.get('asw_signal') == asw_signal), None)
                
                
                if upstream_data and downstream_data:
                    up_values = np.array(upstream_data.get("dbc_data", []))
                    down_values = np.array(downstream_data.get("dbc_data", []))

                    if len(up_values) == len(down_values) and np.array_equal(up_values, down_values):
                        frame_gateway_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝" if not enabled_error_signals else "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    else:
                        frame_gateway_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"
                
                if not dbc_value:
                    print(f"Skipping {dbc_signal}: dbc_value is None")
                    continue
                
                if dbc_value :
                    dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                    asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                    dbc_data = np.array(dbc_value.get('dbc_data', []))
                    asw_data = np.array(dbc_value.get('asw_data', []))
                   
                    if dbc_timestamps.size == 0 or asw_timestamps.size == 0 or dbc_data.size == 0 or asw_data.size == 0:
                        print(f"Skipping signal {dbc_signal} due to missing timestamps or data.")
                        continue
                
                basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

                if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                    # Determine common end timestamp
                    common_end_time = min(dbc_timestamps[-1], asw_timestamps[-1])

                    # Filter DBC data up to common end time
                    valid_dbc_indices = dbc_timestamps <= common_end_time
                    dbc_times_to_compare = dbc_timestamps[valid_dbc_indices]
                    dbc_vals_to_compare = dbc_data[valid_dbc_indices]

                    # Now compare each value with the nearest timestamp in ASW
                    matched_values = [
                        dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]
                        for dbc_time, dbc_val in zip(dbc_times_to_compare, dbc_vals_to_compare)
                    ]

                    if all(matched_values):
                        basic_comm_status = (
                            "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
                            if not enabled_error_signals else
                            "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                        )
                
                
                error_status_list_basic, figures = analysis_basic_receiving_signal([upstream_signal], synchronized_signals_mf4, 
                                                        enabled_error_signals, frame_id, frame_name,
                                                        synchronized_signals_mf4, dtc_matrix_path, mf4_file_path, 
                                                        can_matrix_path,gateway_sheet_path,fid_mapping_path,mdf_file_path,offset,known_templates)
                if error_status_list_basic is None:
                    error_status_list_basic = [True]
                else:
                    error_status_list_basic = [x if x is not None else True for x in error_status_list_basic]
                
                node = f"Upstream Node: {from_node}\nDownstream Node: {to_node}"
                network_signal = f"Upstream Signal: {upstream_signal}\nDownstream Signal: {downstream_signal}"
                asw_signal = asw_signal
                basic_replacement_status = "#c6f6c3" if all(error_status_list_basic) else "#ffc6c3"
                signal_gateway_status = "--"
                signal_gateway_sub_status = "--"
                color_dict[upstream_signal] = {'basic_replacement_status': basic_replacement_status,
                                               'signal_gateway_sub_status': None}
                figure_dict[upstream_signal] = {'basic_communication_figure': [figures],
                                                'signal_gateway_figure': []}
                
                
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status," ", signal_gateway_status,signal_gateway_sub_status])
                
            
            elif gateway_type in ["used+signal gateway"]:
                signal_info = signal_info_dict.get(dbc_signal, {})
                from_node = signal_info.get("from_node", "N/A")
                to_node = signal_info.get("to_node", "N/A")
                upstream_signal = signal_info.get("upstream_signal", dbc_signal)
                downstream_signal = signal_info.get("downstream_signal", "N/A")
                asw_signal = signal_info.get("asw_signal", "N/A")
                
                upstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == upstream_signal), None)
                downstream_data = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == downstream_signal), None)
                dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
                #asw_data = next((s for s in synchronized_signals_mf4 if s.get('asw_signal') == asw_signal), None)
                
                if upstream_data and downstream_data:
                    up_values = np.array(upstream_data.get("dbc_data", []))
                    down_values = np.array(downstream_data.get("dbc_data", []))

                    if len(up_values) == len(down_values) and np.array_equal(up_values, down_values):
                        signal_gateway_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝" if not enabled_error_signals else "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    else:
                        signal_gateway_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"
                if dbc_value :
                    dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                    if dbc_timestamps.size == 0:
                        continue 
                    asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                    if asw_timestamps.size == 0:
                        continue 
                    dbc_data = np.array(dbc_value.get('dbc_data', []))
                    asw_data = np.array(dbc_value.get('asw_data', []))

                basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

                if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                    # Determine common end timestamp
                    common_end_time = min(dbc_timestamps[-1], asw_timestamps[-1])

                    # Filter DBC data up to common end time
                    valid_dbc_indices = dbc_timestamps <= common_end_time
                    dbc_times_to_compare = dbc_timestamps[valid_dbc_indices]
                    dbc_vals_to_compare = dbc_data[valid_dbc_indices]

                    # Now compare each value with the nearest timestamp in ASW
                    matched_values = [
                        dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]
                        for dbc_time, dbc_val in zip(dbc_times_to_compare, dbc_vals_to_compare)
                    ]

                    if all(matched_values):
                        basic_comm_status = (
                            "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
                            if not enabled_error_signals else
                            "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                        )
                
                error_status_list_basic, figures = analysis_basic_receiving_signal([upstream_signal], synchronized_signals_mf4, 
                                                        enabled_error_signals, frame_id, frame_name,
                                                        synchronized_signals_mf4, dtc_matrix_path, mf4_file_path, 
                                                        can_matrix_path,gateway_sheet_path,fid_mapping_path,mdf_file_path,offset,known_templates)
                if error_status_list_basic is None:
                    error_status_list_basic = [True]
                else:
                    error_status_list_basic = [x if x is not None else True for x in error_status_list_basic]

                error_status_list_gateway,gx_figures = signal_gateway_analysis(upstream_signal,downstream_signal,gateway_sheet_path,synchronized_signals_mdf,
                                                 enabled_error_signals,frame_id,synchronized_signals_mf4,
                                                 mdf_file_path,can_matrix_path,offset,mf4_file_path,dtc_matrix_path,
                                                 frame_name,fid_mapping_path,time_shift,known_templates)
                        
                if error_status_list_gateway is None:
                    error_status_list_gateway = [True]

                else:
                    error_status_list_gateway = [x if x is not None else True for x in error_status_list_gateway]

                node = f"Upstream Node: {from_node}\nDownstream Node: {to_node}"
                network_signal = f"Upstream Signal: {upstream_signal}\nDownstream Signal: {downstream_signal}"
                asw_signal = asw_signal
                frame_gateway_status = "--"
                basic_replacement_status = "#c6f6c3" if all(error_status_list_basic) else "#ffc6c3"
                signal_gateway_sub_status = "#c6f6c3" if all(error_status_list_gateway) else "#ffc6c3"
                color_dict[upstream_signal] = {'basic_replacement_status': basic_replacement_status,
                                               'signal_gateway_sub_status': signal_gateway_sub_status}
                figure_dict[upstream_signal] = {'basic_communication_figure': [figures],
                                                'signal_gateway_figure': [gx_figures]}
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status," ",signal_gateway_status," "])
            
            elif gateway_type in ["used+frame+signal gateway"]:
                signal_info = signal_info_dict.get(dbc_signal, {})
                from_node = signal_info.get("from_node", "N/A")
                to_node = signal_info.get("to_node", "N/A")
                upstream_signal = signal_info.get("upstream_signal", dbc_signal)
                downstream_signal_signal_gateway = signal_info.get("downstream_signal_signal_gateway", "N/A")
                downstream_signal_frame = signal_info.get("downstream_signal_frame", "N/A")
                asw_signal = signal_info.get("asw_signal", "N/A")
                upstream_data_frame = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == upstream_signal), None)
                downstream_data_frame = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == downstream_signal_frame), None)
                if upstream_data_frame and downstream_data_frame:
                    up_values_frame = np.array(upstream_data_frame.get("dbc_data", []))
                    down_values_frame = np.array(downstream_data_frame.get("dbc_data", []))

                    if len(up_values_frame) == len(down_values_frame) and np.array_equal(up_values_frame, down_values_frame):
                        frame_gateway_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝" if not enabled_error_signals else "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    else:
                        frame_gateway_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"
                
                upstream_data_signal = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == upstream_signal), None)
                downstream_data_signal = next((s for s in synchronized_signals_mdf if s.get('dbc_signal') == downstream_signal_signal_gateway), None)
                if upstream_data_signal and downstream_data_signal:
                    up_values_signal = np.array(upstream_data_signal.get("dbc_data", []))
                    down_values_signal = np.array(downstream_data_signal.get("dbc_data", []))

                    if len(up_values_signal) == len(down_values_signal) and np.array_equal(up_values_signal, down_values_signal):
                        signal_gateway_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝" if not enabled_error_signals else "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    else:
                        signal_gateway_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"
                dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
                if dbc_value :
                    dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                    if dbc_timestamps.size == 0:
                        continue 
                    asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                    if asw_timestamps.size == 0:
                        continue 
                    dbc_data = np.array(dbc_value.get('dbc_data', []))
                    asw_data = np.array(dbc_value.get('asw_data', []))

                basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

                if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                    # Determine common end timestamp
                    common_end_time = min(dbc_timestamps[-1], asw_timestamps[-1])

                    # Filter DBC data up to common end time
                    valid_dbc_indices = dbc_timestamps <= common_end_time
                    dbc_times_to_compare = dbc_timestamps[valid_dbc_indices]
                    dbc_vals_to_compare = dbc_data[valid_dbc_indices]

                    # Now compare each value with the nearest timestamp in ASW
                    matched_values = [
                        dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]
                        for dbc_time, dbc_val in zip(dbc_times_to_compare, dbc_vals_to_compare)
                    ]

                    if all(matched_values):
                        basic_comm_status = (
                            "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
                            if not enabled_error_signals else
                            "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                        )

                if basic_comm_status == "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝":
                    error_status_list_basic, figures = analysis_basic_receiving_signal([upstream_signal], synchronized_signals_mf4, 
                                                            enabled_error_signals, frame_id, frame_name,
                                                            synchronized_signals_mf4, dtc_matrix_path, mf4_file_path, 
                                                            can_matrix_path,gateway_sheet_path,fid_mapping_path,mdf_file_path,offset,known_templates)
                    if error_status_list_basic is None:
                        error_status_list_basic = [True]
                    else:
                        error_status_list_basic = [x if x is not None else True for x in error_status_list_basic]


                if signal_gateway_status == "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝":
                    print("detected mismatch ..going for analysis")
                    error_status_list_gateway,gx_figures = signal_gateway_analysis(upstream_signal,downstream_signal_signal_gateway,gateway_sheet_path,synchronized_signals_mdf,
                                                    enabled_error_signals,frame_id,synchronized_signals_mf4,
                                                    mdf_file_path,can_matrix_path,offset,mf4_file_path,dtc_matrix_path,
                                                    frame_name,fid_mapping_path,time_shift,known_templates)
                            
                    if error_status_list_gateway is None:
                        error_status_list_gateway = [True]

                    else:
                        error_status_list_gateway = [x if x is not None else True for x in error_status_list_gateway]

                node = f"Upstream Node: {from_node}\nDownstream Node: {to_node}"
                network_signal = f"Upstream Signal: {upstream_signal}\nDownstream Signal: {downstream_signal_signal_gateway}"
                asw_signal = asw_signal
                basic_replacement_status = "#c6f6c3" if all(error_status_list_basic) else "#ffc6c3"
                signal_gateway_sub_status = "#c6f6c3" if all(error_status_list_gateway) else "#ffc6c3"
                color_dict[upstream_signal] = {'basic_replacement_status': basic_replacement_status,
                                               'signal_gateway_sub_status': signal_gateway_sub_status}
                figure_dict[upstream_signal] = {'basic_communication_figure': [figures],
                                                'signal_gateway_figure': [gx_figures]}
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status," ",signal_gateway_status," "])

        elif dbc_signal in basic_receiving_signal:
            signal_info = signal_info_dict.get(dbc_signal, {})
            node = signal_info.get("node", "N/A")
            network_signal = signal_info.get("network_signal", dbc_signal)
            asw_signal = signal_info.get("asw_signal", "N/A")
            if asw_signal == "Not Required":
                continue
            frame_gateway_status = "--"
            signal_gateway_status = "--"
            signal_gateway_sub_status = "--"
            dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
            if not dbc_value:
                continue
            else:
                dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                if dbc_timestamps.size == 0:
                        continue 
                asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                if asw_timestamps.size == 0:
                        continue 
                dbc_data = np.array(dbc_value.get('dbc_data', []))
                asw_data = np.array(dbc_value.get('asw_data', []))

            basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

            if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                # Determine common end timestamp
                common_end_time = min(dbc_timestamps[-1], asw_timestamps[-1])

                # Filter DBC data up to common end time
                valid_dbc_indices = dbc_timestamps <= common_end_time
                dbc_times_to_compare = dbc_timestamps[valid_dbc_indices]
                dbc_vals_to_compare = dbc_data[valid_dbc_indices]

                # Now compare each value with the nearest timestamp in ASW
                matched_values = [
                    dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]
                    for dbc_time, dbc_val in zip(dbc_times_to_compare, dbc_vals_to_compare)
                ]

                if all(matched_values):
                    basic_comm_status = (
                        "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
                        if not enabled_error_signals else
                        "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    )

            # Check if all required paths for analysis are present
            if dtc_matrix_path and gateway_sheet_path and fid_mapping_path and basic_comm_status == "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝":
                error_status_list_basic, figures = analysis_basic_receiving_signal(
                    [network_signal], synchronized_signals_mf4, enabled_error_signals,
                    frame_id, frame_name, synchronized_signals_mf4, dtc_matrix_path,
                    mf4_file_path, can_matrix_path, gateway_sheet_path,
                    fid_mapping_path, mdf_file_path, offset,known_templates)
                if error_status_list_basic is None:
                    error_status_list_basic = [True]
                else:
                    error_status_list_basic = [x if x is not None else True for x in error_status_list_basic]

                basic_replacement_status = "#c6f6c3" if all(error_status_list_basic) else "#ffc6c3"
                color_dict[network_signal] = {'basic_replacement_status': basic_replacement_status,
                                                'signal_gateway_sub_status': None}
                figure_dict[network_signal] = {
                    'basic_communication_figure': [figures],
                    'signal_gateway_figure': []
                }

            row.append([
                node, network_signal, asw_signal,
                frame_gateway_status, basic_comm_status, " ",
                signal_gateway_status, signal_gateway_sub_status
            ])

        
        elif dbc_signal in multiplexor_signals or dbc_signal in selector_signals:
            signal_info = signal_info_dict.get(dbc_signal, {})
            node = signal_info.get("node", "N/A")
            network_signal = signal_info.get("network_signal", dbc_signal)
            asw_signal = signal_info.get("asw_signal", "N/A")
            frame_gateway_status = "--"
            signal_gateway_status = "--"
            signal_gateway_sub_status = "--"
            dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
            if not dbc_value:
                continue
            else:
                dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                if dbc_timestamps.size == 0:
                        continue 
                asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                if asw_timestamps.size == 0:
                        continue 
                dbc_data = np.array(dbc_value.get('dbc_data', []))
                asw_data = np.array(dbc_value.get('asw_data', []))

            basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

            if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                # Determine common end timestamp
                common_end_time = min(dbc_timestamps[-1], asw_timestamps[-1])

                # Filter DBC data up to common end time
                valid_dbc_indices = dbc_timestamps <= common_end_time
                dbc_times_to_compare = dbc_timestamps[valid_dbc_indices]
                dbc_vals_to_compare = dbc_data[valid_dbc_indices]

                # Now compare each value with the nearest timestamp in ASW
                matched_values = [
                    dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]
                    for dbc_time, dbc_val in zip(dbc_times_to_compare, dbc_vals_to_compare)
                ]

                if all(matched_values):
                    basic_comm_status = (
                        "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
                        if not enabled_error_signals else
                        "       ╔═══╗\n       ║   – ║\n       ╚═══╝"
                    )
            
            if dtc_matrix_path and gateway_sheet_path and fid_mapping_path and basic_comm_status == "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝":
                error_status_list_basic, figures = analysis_basic_receiving_signal([network_signal], synchronized_signals_mf4, 
                                                            enabled_error_signals, frame_id, frame_name,
                                                            synchronized_signals_mf4, dtc_matrix_path, mf4_file_path, 
                                                            can_matrix_path,gateway_sheet_path,fid_mapping_path,mdf_file_path,offset,known_templates)
                if error_status_list_basic is None:
                    error_status_list_basic = [True]
                else:
                    error_status_list_basic = [x if x is not None else True for x in error_status_list_basic]
                basic_replacement_status = "#c6f6c3" if all(error_status_list_basic) else "#ffc6c3"
                color_dict[network_signal] = {'basic_replacement_status': basic_replacement_status,
                                                'signal_gateway_sub_status': None}
                figure_dict[network_signal] = {'basic_communication_figure': [figures],
                                                    'signal_gateway_figure': []}
            row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status," ",signal_gateway_status,signal_gateway_sub_status])
            
        elif dbc_signal in transmitter_signals:
            signal_info = signal_info_dict.get(dbc_signal, {})
            node = signal_info.get("node", "N/A")
            network_signal = signal_info.get("network_signal", dbc_signal)
            asw_signal = signal_info.get("asw_signal", "N/A")
            frame_gateway_status = "--"
            basic_replacement_status = ""
            signal_gateway_status = "--"
            signal_gateway_sub_status = "--"
            dbc_value = next((s for s in synchronized_signals_mf4 if s.get('dbc_signal') == dbc_signal), None)
            if not dbc_value:
                continue
            else:
                dbc_timestamps = np.array(dbc_value.get('dbc_timestamps', []))
                if dbc_timestamps.size == 0:
                        continue 
                asw_timestamps = np.array(dbc_value.get('asw_timestamps', []))
                if asw_timestamps.size == 0:
                        continue 
                dbc_data = np.array(dbc_value.get('dbc_data', []))
                asw_data = np.array(dbc_value.get('asw_data', []))

            basic_comm_status = "       ╔═══╗\n       ║   ✘ ║\n       ╚═══╝"

            if dbc_timestamps.size > 0 and asw_timestamps.size > 0 and dbc_data.size > 0 and asw_data.size > 0:
                matched_values = [dbc_val == asw_data[np.abs(asw_timestamps - dbc_time).argmin()]for dbc_time, dbc_val in zip(dbc_timestamps, dbc_data)]
                if all(matched_values):
                    basic_comm_status = "       ╔═══╗\n       ║   ✔ ║\n       ╚═══╝"
            row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status,basic_replacement_status, 
                            signal_gateway_status,signal_gateway_sub_status])


    table_df = pd.DataFrame(row, columns=headers)
    return table_df,color_dict,figure_dict
    




def load_l2_asw_signals(file_path):
    try:
        l2_asw_df = pd.read_excel(file_path, sheet_name='L2ASW')

        if 'Signal name' not in l2_asw_df.columns or 'L2 ASW Interface' not in l2_asw_df.columns:
            return {}

        l2_asw_signals_dict = pd.Series(
            l2_asw_df['L2 ASW Interface'].values,
            index=l2_asw_df['Signal name']
        ).to_dict()

        return l2_asw_signals_dict

    except Exception:
        return {}

# Function to check communication success, with fallback to L2 ASW if primary ASW fails
def check_transmitter_communication_success(transmitter_signals, synchronized_signals, CAN_Matrix_path):
    print("Function called")
    l2_asw_signals_dict = load_l2_asw_signals(CAN_Matrix_path)

    for dbc_signal in transmitter_signals:
        print(f"Checking signal: {dbc_signal}")
        primary_match = True

        # Get primary ASW data
        signal_entry = next((s for s in synchronized_signals if s.get('dbc_signal') == dbc_signal), None)

        if signal_entry:
            dbc_data = signal_entry.get('dbc_data')
            asw_data = signal_entry.get('asw_data')
            dbc_timestamps = signal_entry.get('dbc_timestamps')
            asw_timestamps = signal_entry.get('asw_timestamps')

            if dbc_data is not None and asw_data is not None:
                # Trim to shortest length
                min_len = min(len(dbc_data), len(asw_data))
                mismatch_timestamps = []
                dbc_data, asw_data = dbc_data[:min_len], asw_data[:min_len]

                # Compare DBC and ASW data
                for i in range(min_len):
                    if dbc_data[i] != asw_data[i]:
                        mismatch_timestamps.append(dbc_timestamps[i])
            
                print(mismatch_timestamps)
                if not mismatch_timestamps:
                    print(f"Communication Success for '{dbc_signal}'")
                    print("-" * 40)
                    continue  # no need to check L2 ASW
                else:
                    print(f"Mismatch in ASW for '{dbc_signal}', checking L2 ASW...")
                    return False

            else:
                print(f"Missing DBC or ASW data for '{dbc_signal}', checking L2 ASW...")
                

            # --- L2 ASW CHECK ---
            l2_asw_signal_name = l2_asw_signals_dict.get(dbc_signal)
            if l2_asw_signal_name:
                l2_signal_entry = next((s for s in synchronized_signals if s.get('dbc_signal') == l2_asw_signal_name), None)
                if l2_signal_entry:
                    l2_asw_data = l2_signal_entry.get('dbc_data')
                    l2_asw_timestamps = l2_signal_entry.get('dbc_timestamps')

                    if l2_asw_data is not None and len(dbc_data) == len(l2_asw_data):
                        l2_asw_match = True
                        for i in range(len(dbc_data)):
                            if dbc_data[i] != l2_asw_data[i]:
                                l2_asw_match = False
                                break

                        if l2_asw_match:
                            print(f"L2 ASW Success for '{dbc_signal}' vs '{l2_asw_signal_name}'")
                            print("-" * 40)
                        else:
                            print(f"Final mismatch for '{dbc_signal}'. No match found in both ASW and L2 ASW.")
                            print("-" * 40)
                    else:
                        print(f"L2 ASW data not found or length mismatch for '{l2_asw_signal_name}'")
                        print("-" * 40)
                else:
                    print(f"L2 ASW signal '{l2_asw_signal_name}' not found in synchronized signals.")
                    print("-" * 40)
            else:
                print(f"No L2 ASW mapping found for '{dbc_signal}' in CAN Matrix.")
                print("-" * 40)
        else:
            print(f"DBC signal '{dbc_signal}' not found in synchronized signals.")
            print("-" * 40)
