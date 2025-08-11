import mdfreader
import pandas as pd
import numpy as np
from my_tool.deep_signal_analysis.error_analysis import (compare_signal1_with_signal2,analyze_error_for_range)
from my_tool.visualization.plotting_signals_graphs import plot_ini_calculation
from my_tool.visualization.detailed_mismatch_plot import plot_detailed_signal_mismatch


#   ----------------HELPER FUNCTIONS TO CREATE QUALITY STATUS TABLE OF SIGNAL-------------------
#   BASIC RX SIGNAL ANALYSIS
#   -----------------------------------------------------------------------------------

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
            figures = plot_detailed_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, synchronized_signals, error_info_dict,synchronized_signals_mf4,mode="Basic")
            print("=" * 80)
    
    return error_status_list,figures



#   -------------------------------------------------------------------------------------
#   SIGNAL GATEWAY ANALYSIS
#   -----------------------------------------------------------------------------------


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
            figures = plot_detailed_signal_mismatch(mismatch_start_time, mismatch_end_time, asw_signal, signal_data_list, 
                                           mismatch_info_dict,synchronized_signals_mf4,mode="Gateway")
            print("=" * 80)
    return error_status_list,figures