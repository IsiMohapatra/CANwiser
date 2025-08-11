from datetime import datetime
import mdfreader
import re
import numpy as np

# ---------------------------------------------------------------------
# 🕒 Timestamp Extraction Utilities
# ---------------------------------------------------------------------

def extract_date_and_time_of_mf4(file_path):
    data = mdfreader.Mdf(file_path)

    if hasattr(data, 'fileMetadata'):
        metadata = str(data.fileMetadata)

        # Match date in multiple formats: 17/04/2025, 04/17/2025, 28-jul-2025
        date_match = re.search(
    r"Date:\s*((?:\d{1,2}-[A-Za-z]{3}-\d{2,4})|(?:\d{1,2}/\d{1,2}/\d{2,4}))",
    metadata)

        # Match time (English: Time, French: Temps)
        time_match = re.search(r"(?:Time|Temps)[.:]*\s*([\d:]+\s*[APMapm]*)", metadata)

        if date_match and time_match:
            datetime_str = f"{date_match.group(1)} {time_match.group(1)}"
            print(f"mf4_date_time: {datetime_str}")

            possible_formats = [
                "%m/%d/%Y %I:%M:%S %p",  # 04/17/2025 03:23:41 PM
                "%d/%m/%Y %H:%M:%S",     # 17/04/2025 15:23:41
                "%d-%b-%Y %H:%M:%S",     # 28-Jul-2025 15:23:41
                "%d-%b-%Y %I:%M:%S %p",  # 28-Jul-2025 03:23:41 PM
                "%d-%b-%y %H:%M:%S",     # 28-Jul-25 15:23:41
                "%d-%b-%y %I:%M:%S %p"   # 28-Jul-25 03:23:41 PM
            ]

            for fmt in possible_formats:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    return dt.strftime("%m/%d/%Y"), dt.strftime("%I:%M:%S.%f %p")
                except ValueError:
                    continue  # Try the next format

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


# ---------------------------------------------------------------------
#  DBC ↔ ASW Signal Finder 
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# SEARCHING ERROR SIGNAL IN MF4 FILE UTILITY FUNCTION
# ---------------------------------------------------------------------

def search_error_signals_in_mf4(mf4_file_path, search_string):
    try:
        mdf = mdfreader.Mdf(mf4_file_path)
        signal_names = mdf.keys()
        matching_signals = [signal for signal in signal_names if search_string in signal]
        return matching_signals, mdf
    except Exception as e:
        print(f"Error reading MF4 file: {str(e)}")
        return [], None


# ---------------------------------------------------------------------
# SHIFT RAW MF4 SIGNAL DICT WITH CALCULATED OFFSET
# IT FINDS THE EXACT PLACE WHERE THERE IS MISSING WITHOUT ANY INTERPOLATED VALUE
# ---------------------------------------------------------------------


def shift_raw_mf4_data_dict(missing_data_dict, offset):
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


# ---------------------------------------------------------------------
# CHECK ERROR SIGNALS AT MISMATCH TIMESTAMP
# IT HELPS US TO FIND IF THE ANY ERROR IS ACTIVATED IN TIME OF MISMATCH
# ---------------------------------------------------------------------

def search_additional_error_signals(synchronized_signals, frame_id, dbc_signal_name, timestamp):
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


def check_error_signals_at_mismatch_timestamp(timestamp, synchronized_signals, enabled_error_signals, frame_id, signal_name_check):
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
    additional_error_values = search_additional_error_signals(synchronized_signals, frame_id, signal_name_check, timestamp)
    
    if additional_error_values:
        # Append the additional error signals and their values to error_status
        for key, value in additional_error_values.items():
            error_status.append({
                'error_signal': key,
                'error_value': value[1],  # value[1] is the error value at the closest timestamp
            })

    return error_status