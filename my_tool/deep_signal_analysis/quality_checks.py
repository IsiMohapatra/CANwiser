import numpy as np
from IPython.display import display
import pandas as pd
from my_tool.deep_signal_analysis.quality_utils import (analysis_basic_receiving_signal,signal_gateway_analysis)


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
                
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status,basic_replacement_status, 
                            signal_gateway_status,signal_gateway_sub_status])
                
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
                
                row.append([node, network_signal,asw_signal,frame_gateway_status, basic_comm_status," ",
                             signal_gateway_status,signal_gateway_sub_status])
            
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