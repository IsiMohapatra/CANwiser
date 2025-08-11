import numpy as np
import pandas as pd
from my_tool.data_loading import load_signals_from_excel
from my_tool.utils import find_corresponding_signal
from my_tool.signal_processing.file_level_signal_processing import (raw_all_signals_mdf,raw_all_signals_mf4,process_and_resample_signal)
from my_tool.visualization.plotting_signals_graphs import (plot_individual_signals,plot_synchronized_signals)


# -------------------------HELPER FUNCTIONS FOR SYNCHING-------------------------------------------------------------------
#    CALCULATING THE OFFSET BETWEEN BOTH THE SIGNALS (STATE CHANGE METHOD)
#    SHIFTING THE MF4 SIGNALS WITH THE OFFSET TO SYNC 
#    SHIFTING THE RAW ASW TIMESTAMPS WITHOUT INTERPOLATION TO SPOT THE EXACT MISSING TIMESTAMPS
# ---------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------
#    MAIN SYNCHRONIZATION FUNCTION FOR SYNCING TWO SIGNALS
# ---------------------------------------------------------------------------------------------

def synching_of_two_signals(mdf_file_path, mf4_file_path, CAN_Matrix_path, synchronized_signals_mdf, input_signal):
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