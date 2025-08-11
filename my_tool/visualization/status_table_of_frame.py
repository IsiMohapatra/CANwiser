import pandas as pd
import mdfreader 
from my_tool.data_loading import load_signals_from_excel

# ------------------------------------------------------------------
# HELPER FUNCTIONS TO GENERATE THE STATUS TABLE
# -------------------------------------------------------------------

def normalize_frame(frame):
    frame = frame.lower().replace('0x', '').replace('h', '')
    return frame 

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


# ------------------------------------------------------------------
# MAIN FUNCTION TO GENERATE DETAILED STATUS TABLE OF THE FRAME 
# -------------------------------------------------------------------

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