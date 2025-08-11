import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.gridspec as gridspec


# --------------------------------------------------------------------------------------------
#           PLOT DETAILED SIGNAL MISMATCH WITH ZOOMED IN ERROR REGIONS
#--------------------------------------------------------------------------------------------

def plot_detailed_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, signal_data_list,
                          mismatch_info_dict,synchronized_signals_mf4,mode):
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