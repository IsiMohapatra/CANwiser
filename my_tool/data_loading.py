import pandas as pd


# ------------------ LOADING DATA FROM EXCEL (CAN MATRIX) ----------------------

def load_signals_from_excel(CAN_Matrix_path):
    """Loads Excel data once to avoid repeated file access."""
    try:
        tx_df = pd.read_excel(CAN_Matrix_path, sheet_name='VCU_TX_SIGNALS')
        rx_df = pd.read_excel(CAN_Matrix_path, sheet_name='VCU_RX_SIGNALS')
    except Exception as e:
        print(f"Failed to load sheets: {str(e)}")
        return None, None
    return tx_df, rx_df

# ------------------ LOADING DATA FROM DTC MATRIX ----------------------

def load_dtc_matrix_and_disable_mask(dtc_excel_path, matrix_header_row=5):
    """
    Loads both the 'Matrix' and 'Disable_Mask' sheets from the DTC Excel file.

    Returns:
        tuple: (matrix_df, disable_mask_df)
    """
    matrix_df = pd.DataFrame()
    disable_mask_df = pd.DataFrame()

    try:
        # Load both sheets at once
        excel_data = pd.read_excel(dtc_excel_path, sheet_name=['Matrix', 'Disable_Mask'], header=None)

        # Extract 'Matrix' sheet and set proper header
        if 'Matrix' in excel_data:
            raw_matrix_df = excel_data['Matrix']
            raw_matrix_df.columns = raw_matrix_df.iloc[matrix_header_row]
            matrix_df = raw_matrix_df.iloc[matrix_header_row + 1:].copy()
            matrix_df['Supplier label'] = matrix_df['Supplier label'].astype(str).str.strip()

        # Extract 'Disable_Mask' sheet
        if 'Disable_Mask' in excel_data:
            disable_mask_df = excel_data['Disable_Mask']
            disable_mask_df.columns = disable_mask_df.iloc[0]
            disable_mask_df = disable_mask_df.iloc[1:].copy()
            disable_mask_df['Label name'] = disable_mask_df['Label name'].astype(str).str.strip()

    except Exception as e:
        print(f"Error loading DTC sheets: {e}")

    return matrix_df, disable_mask_df