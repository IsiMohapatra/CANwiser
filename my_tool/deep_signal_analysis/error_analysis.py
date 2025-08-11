import numpy as np
from my_tool.signal_processing.file_level_signal_processing import raw_all_signals_mdf, raw_all_signals_mf4
from my_tool.utils import shift_raw_mf4_data_dict,check_error_signals_at_mismatch_timestamp
import mdfreader
from my_tool.data_loading import load_signals_from_excel
import pandas as pd


# ---------------------------------------------------------------------
# COMPARE TWO SIGNALS TO DETECT MISMATCH 
# ---------------------------------------------------------------------

def compare_signal1_with_signal2(signal_list, signal_list_structure, enabled_error_signals, frame_id, synchronized_signals_mf4,
                                 mdf_file_path, CAN_Matrix_path, calculated_offset, mf4_file_path, mode):
    
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
            missing_timestamps_asw = shift_raw_mf4_data_dict(missing_timestamps_asw, calculated_offset)
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
                    error_status = check_error_signals_at_mismatch_timestamp(dbc_timestamp, synchronized_signals_mf4, enabled_error_signals, 
                                                                    frame_id, asw_signal)
                else:  # Basic mode
                    error_status = check_error_signals_at_mismatch_timestamp(dbc_timestamp, synchronized_signals_mf4, enabled_error_signals, 
                                                                    frame_id, dbc_signal)

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


# ---------------------------------------------------------------------
# ANALYZE THE TYPE OF ERROR AT THE TIMESTAMP
# ---------------------------------------------------------------------

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