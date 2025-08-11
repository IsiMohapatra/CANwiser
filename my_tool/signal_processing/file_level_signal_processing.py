import mdfreader
import numpy as np
import pandas as pd
from my_tool.data_loading import load_signals_from_excel


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

# ---------------------------------------------------------------------
# RAW MDF Signal Processing Functions (without interpolation to know the exact missing timestamps)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# MDF Signal Processing Functions (with interpolation based on periodicity)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# RAW MF4 Signal Processing Functions (without interpolation to kow the exact missing timestamps)
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
#  MF4 Signal Processing Functions (shifting with the calculated offset)
# ---------------------------------------------------------------------

def shift_all_signals_mf4(mf4_file_path, offset):
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


# ---------------------------------------------------------------------
# RESAMPLING THE SIGNALS FOR COMPARISION
# ---------------------------------------------------------------------

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
    

