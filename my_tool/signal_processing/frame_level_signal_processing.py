import pandas as pd
from my_tool.signal_processing.file_level_signal_processing import process_and_resample_signal
from my_tool.data_loading import load_signals_from_excel
from my_tool.utils import search_error_signals_in_mf4



# ----------------------------------------------------------------------
# FETCHING ALL THE DBC AND THEIR CORRESPONDING ASW SIGNALS
# ---------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# PROCESS ALL THE SIGNALS OF THE PARTICULAR FRAME
# ---------------------------------------------------------------------

def process_all_signals_of_frame(dbc_signals, asw_signals, offset,mdf_file_path,mf4_file_path,CAN_Matrix_path,synchronized_signals_mdf,
                                 synchronized_signals_mf4):
    for dbc_signal, asw_signal in zip(dbc_signals, asw_signals):
        if mdf_file_path is None:
            dbc_timestamps, dbc_data, dbc_tracking_flag = process_and_resample_signal(
                mf4_file_path, dbc_signal, CAN_Matrix_path, synchronized_signals_mdf)
            asw_timestamps, asw_data, asw_tracking_flag = process_and_resample_signal(
                mf4_file_path, asw_signal, CAN_Matrix_path, synchronized_signals_mdf)
        else:
            dbc_timestamps, dbc_data, dbc_tracking_flag = process_and_resample_signal(
                mdf_file_path, dbc_signal, CAN_Matrix_path, synchronized_signals_mdf)
            asw_timestamps, asw_data, asw_tracking_flag = process_and_resample_signal(
                mf4_file_path, asw_signal, CAN_Matrix_path, synchronized_signals_mdf)

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


# ----------------------------------------------------------------------
# FIND THE TYPE OF SIGNALS (RECEIVER OR TRANSMITTER)
# ---------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# FIND THE ERROR SIGNALS OF THE PARTICULAR FRAME
# ---------------------------------------------------------------------

def find_error_signals_for_frame(frame_name_input, mf4_file_path, frame_id):
    if frame_id is None:
        print(f"No Frame ID found for frame '{frame_name_input}'.")
        return [], None
    frame_id_str = str(frame_id).strip()
    frame_id_numeric = frame_id_str.replace('0x', '') if frame_id_str.startswith('0x') else frame_id_str
    search_string = f"DFC_st.DFC_Com{frame_id_numeric}"
    matching_signals, mdf = search_error_signals_in_mf4(mf4_file_path, search_string)
    return matching_signals, mdf


# ----------------------------------------------------------------------
# FILTER ACTIVATED ERROR SIGNALS OF THE FRAME FROM DTC MATRIX
# ---------------------------------------------------------------------

def filter_activated_error_signals(matrix_df, matching_signals):
    activated_signals = []
    for signal in matching_signals:
        signal = str(signal).strip()
        signal_full = signal
        signal_clean = signal[7:] if signal.startswith('DFC_st.') else signal

        signal_info = matrix_df[matrix_df['Supplier label'] == signal_clean]
        if not signal_info.empty:
            activation_status = signal_info['Activation target'].values[0]
            if activation_status == "Activated":
                activated_signals.append(signal_full)
    return activated_signals


# ----------------------------------------------------------------------
# FILTER ENABLED ERROR SIGNALS OF THE FRAME FROM DTC MATRIX
# ---------------------------------------------------------------------

def filter_enabled_signals_from_dsm(disable_mask_df, activated_signals, label_column='Label name', cal_column='CAL'):
    enabled_signals = []
    for signal_full in activated_signals:
        signal_full = str(signal_full).strip()
        signal_clean = signal_full[7:] if signal_full.startswith('DFC_st.') else signal_full
        modified_signal_name = f"DFC_DisblMsk2.{signal_clean}_C"

        signal_info = disable_mask_df[disable_mask_df[label_column] == modified_signal_name]
        if not signal_info.empty:
            cal_value = signal_info[cal_column].values[0]
            if cal_value != 65535:
                enabled_signals.append(signal_full)
        else:
            print(f"{modified_signal_name} not found in Disable_Mask sheet.")
    return enabled_signals


# ----------------------------------------------------------------------
# FIND THE TYPE OF GATEWAY (IF THE FRAME IS USED AS GATEWAY)
# ---------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# FIND IF THE FRAME IS MULTIPLEXOR
# ---------------------------------------------------------------------

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
    return  multiplexor_signals, selector_signals

# ----------------------------------------------------------------------
# FIND BASIC RECEIVER SIGNALS OF THE FRAME
# ---------------------------------------------------------------------

def extract_basic_receiving_signal(receiver_signals, multiplexor_signals, selector_signals):
    if isinstance(multiplexor_signals, dict):
        multiplexor_signals = list(multiplexor_signals.keys())
    if isinstance(selector_signals, dict):
        selector_signals = list(selector_signals.keys())

    signals_to_remove = set(multiplexor_signals + selector_signals)
    basic_receiving_signals = [signal for signal in receiver_signals if signal not in signals_to_remove]
    
    return basic_receiving_signals

