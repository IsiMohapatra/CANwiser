import matplotlib.pyplot as plt
import numpy as np
from my_tool.signal_processing.file_level_signal_processing import raw_all_signals_mdf

# --------------------------------------------------------------------------------------------
#           PLOT INDIVIDUAL DBC AND ASW SIGNAL PLOT
#--------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------
#           PLOT MERGED DBC AND ASW SIGNAL PLOT (ON TOP OF EACH OTHER)
#--------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------
#           PLOT INI REGION PLOTS FOR THE SIGNALS (IF FOUND)
#--------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------
#           PLOT SUCCESSFUL COMMUNICATION PLOT OF TWO SIGNALS
#--------------------------------------------------------------------------------------------

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