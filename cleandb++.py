#Developed by ODAT project
#please see https://odat.info
#please see https://github.com/ODAT-Project
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
import re
import os
import threading

def extract_numeric(value):

    if isinstance(value, str):
        #handle potential percentage signs or other common non-numeric chars before extraction
        value = value.replace('%', '').strip()
        match = re.search(r"[-+]?\d*\.?\d+", value)
        return float(match.group()) if match else pd.NA
    elif isinstance(value, (int, float)):
        return float(value)
    return pd.NA

def clean_mixed_columns(df, logger_func=print):

    for column in df.columns:
        #attempt conversion only if the column is of object type and not entirely NA
        if df[column].dtype == "object" and not df[column].isna().all():
            try:
                #create a new series for cleaned values
                cleaned_series = df[column].apply(extract_numeric)

                # if no numbers are found.
                if cleaned_series.notna().any() or df[column].apply(lambda x: isinstance(x, str) and extract_numeric(x) is pd.NA).all():
                     df[column] = cleaned_series
                # else:
                #    logger_func(f"Column '{column}' contained no extractable numeric data and was left as is.")

            except Exception as e:
                logger_func(f"Error processing column {column} for numeric extraction: {e}")
    return df

def remove_fully_non_numeric_columns(df, logger_func=print):

    cols_to_drop = []
    for col in df.columns:
        try:
            #attempt to convert to numeric. If all are NaN, it's non-numeric.
            if pd.to_numeric(df[col], errors='coerce').isna().all():
              
                if df[col].dtype == 'object':

                     pass #handled by the numeric_cols logic below
        except Exception as e:
            logger_func(f"Could not assess numeric nature of column {col}: {e}")
    
    #more direct way to find numeric columns after cleaning attempts
    numeric_cols_mask = []
    for col_name in df.columns:
        #check if at least one value in the column can be considered numeric
        is_potentially_numeric = False
        try:
            if df[col_name].map(lambda x: isinstance(x, (int, float))).any(): #check if any are already numeric
                 is_potentially_numeric = True
            elif pd.to_numeric(df[col_name], errors='coerce').notna().any(): #check if any can be converted
                 is_potentially_numeric = True
        except Exception: #handle mixed types that pd.to_numeric might struggle with before apply
            pass #will be caught by the numeric_cols_mask being False

        numeric_cols_mask.append(is_potentially_numeric)

    if not any(numeric_cols_mask):
        logger_func("Warning: No numeric columns found or all columns became NA. The DataFrame might be empty or contain only non-convertible text.")

    
    return df.loc[:, numeric_cols_mask]


def clean_headers(df):

    new_columns = df.columns.astype(str) #ensure all column names are strings
    new_columns = (
        new_columns.str.replace(r"[（）]", "()", regex=True)  #replace Chinese-style parentheses
                  .str.replace(r"\(.*?\)", "", regex=True)  #remove content inside parentheses
                  .str.replace(r"[^a-zA-Z0-9_\s-]", "", regex=True)  #keep spaces for now
                  .str.replace(" ", "_")                   #replace spaces with underscores
                  .str.replace(r"_+", "_", regex=True)     #replace multiple underscores with single
                  .str.strip('_')                          #remove leading/trailing underscores
                  .str.encode('ascii', errors='ignore').str.decode('ascii')
    )
    #handle empty column names or duplicates by appending a number
    final_columns = []
    counts = {}
    for col in new_columns:
        if not col: #if column name became empty
            col = "unnamed_column"
        if col in counts:
            counts[col] += 1
            final_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            final_columns.append(col)
    df.columns = final_columns
    return df

def report_missing_values(df, output_file, logger_func=print):
    try:
        #total datapoints in the dataset
        total_datapoints = df.size

        #total missing values in the dataset
        total_missing = df.isna().sum().sum()

        #percentage of missing values in the dataset
        missing_percentage_dataset = (total_missing / total_datapoints) * 100 if total_datapoints > 0 else 0

        #percentage of missing values per column
        missing_percentage_columns = (df.isna().mean() * 100) if not df.empty else pd.Series(dtype=float)

        report = [
            f"Missing Values Report for: {os.path.basename(output_file).replace('_report.txt', '.csv')}",
            f"Timestamp: {pd.Timestamp.now()}",
            "---",
            f"Total data points: {total_datapoints}",
            f"Total missing values: {total_missing}",
            f"Overall missing data percentage: {missing_percentage_dataset:.2f}%",
            "\nMissing values percentage per column:"
        ]
        if not missing_percentage_columns.empty:
            report.append(missing_percentage_columns.to_string())
        else:
            report.append("No columns to report or DataFrame is empty.")

        with open(output_file, "w", encoding='utf-8') as file:
            file.write("\n".join(report))
        logger_func(f"Missing value report saved to: {output_file}")
    except Exception as e:
        logger_func(f"Error generating missing values report: {e}")


def mean_imputation(df, logger_func=print):
    numeric_cols_df = df.select_dtypes(include=["number"])
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

    if not numeric_cols_df.empty:
        for col in numeric_cols_df.columns:
            if df[col].isna().any():
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col] = df[col].fillna(mean_val).round(3)
                    logger_func(f"Imputed missing values in column '{col}' with mean: {mean_val:.3f}")
                else:
                    logger_func(f"Could not calculate mean for column '{col}' (all NA after cleaning?). Leaving NAs.")
            else:
                #ensure numeric columns are rounded even if no imputation occurs
                df[col] = df[col].round(3)


    for col in non_numeric_cols:
        if df[col].dtype == 'object':
             df[col] = df[col].fillna(pd.NA)
    return df

def process_single_csv_file(file_path, report_folder, cleaned_folder, logger_func):

    base_filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(base_filename)[0]

    logger_func(f"--- Starting processing for: {base_filename} ---")
    try:
        #load the CSV file
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logger_func(f"Error reading CSV {base_filename}: {e}. Trying with different encoding.")
            try:
                df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            except Exception as e_enc:
                logger_func(f"Failed to read {base_filename} with fallback encoding: {e_enc}")
                return

        if df.empty:
            logger_func(f"File {base_filename} is empty or could not be read properly. Skipping.")
            return

        logger_func(f"Initial shape of {base_filename}: {df.shape}")
        
        #clean headers
        df = clean_headers(df)
        logger_func(f"Headers cleaned for {base_filename}.")

        #clean mixed numeric and non-numeric columns (attempt to extract numerics)
        df = clean_mixed_columns(df.copy(), logger_func) # Use .copy() to avoid SettingWithCopyWarning
        logger_func(f"Mixed-type columns processed for {base_filename}.")
        
        #remove columns that are fully non-numeric (or became all NA after cleaning)
        original_cols = df.columns.tolist()
        df = remove_fully_non_numeric_columns(df.copy(), logger_func)
        removed_cols = set(original_cols) - set(df.columns.tolist())
        if removed_cols:
            logger_func(f"Removed fully non-numeric columns from {base_filename}: {', '.join(removed_cols)}")
        
        if df.empty or df.shape[1] == 0:
            logger_func(f"DataFrame for {base_filename} became empty after removing non-numeric columns. No further processing.")
            empty_cleaned_file = os.path.join(cleaned_folder, f"{filename_no_ext}_clean_empty.csv")
            pd.DataFrame().to_csv(empty_cleaned_file, index=False)
            logger_func(f"Saved an empty placeholder: {empty_cleaned_file}")
            return

        #generate a report on missing values *before* imputation
        report_file = os.path.join(report_folder, f"{filename_no_ext}_report.txt")
        report_missing_values(df, report_file, logger_func)

        #perform mean imputation
        df = mean_imputation(df.copy(), logger_func)
        logger_func(f"Mean imputation completed for {base_filename}.")

        #save cleaned file
        cleaned_file = os.path.join(cleaned_folder, f"{filename_no_ext}_clean.csv")
        df.to_csv(cleaned_file, index=False, na_rep='NA') # Save NA as 'NA' string
        logger_func(f"Cleaned file saved: {cleaned_file}")
        logger_func(f"Final shape of {base_filename}: {df.shape}")

    except Exception as e:
        logger_func(f"!!! Critical error processing file {base_filename}: {e}")
        import traceback
        logger_func(traceback.format_exc())
    finally:
        logger_func(f"--- Finished processing for: {base_filename} ---\n")


def main_processing_logic(input_folder, report_folder, cleaned_folder, logger_func):

    logger_func("Starting CSV processing...")
    os.makedirs(report_folder, exist_ok=True)
    os.makedirs(cleaned_folder, exist_ok=True)

    found_csv_files = False
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                found_csv_files = True
                file_path = os.path.join(root, file)
                process_single_csv_file(file_path, report_folder, cleaned_folder, logger_func)
    
    if not found_csv_files:
        logger_func("No CSV files found in the input folder.")
    logger_func("All CSV processing finished.")

#main GUI
class CSVCleanerApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Data Cleaner Pro")
        master.geometry("700x550")

        #configure style
        self.style = ttk.Style()
        self.style.theme_use('clam') # or 'alt', 'default', 'classic'
        
        self.style.configure("TLabel", padding=5, font=('Helvetica', 10))
        self.style.configure("TButton", padding=5, font=('Helvetica', 10, 'bold'))
        self.style.configure("TEntry", padding=5, font=('Helvetica', 10))
        self.style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))

        #folder selection frames
        self.input_folder_path = tk.StringVar()
        self.report_folder_path = tk.StringVar()
        self.cleaned_folder_path = tk.StringVar()

        #set default folder names (optional, can be relative)
        self.input_folder_path.set("input_csv")
        self.report_folder_path.set("reports")
        self.cleaned_folder_path.set("cleaned_csv")

        #input Folder
        input_frame = ttk.Frame(master, padding=10)
        input_frame.pack(fill=tk.X)
        ttk.Label(input_frame, text="Input CSV Folder:").pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.input_folder_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(input_frame, text="Browse...", command=lambda: self._browse_directory(self.input_folder_path, "Select Input CSV Folder")).pack(side=tk.LEFT)

        #report Folder
        report_frame = ttk.Frame(master, padding=10)
        report_frame.pack(fill=tk.X)
        ttk.Label(report_frame, text="Report Output Folder:").pack(side=tk.LEFT)
        ttk.Entry(report_frame, textvariable=self.report_folder_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(report_frame, text="Browse...", command=lambda: self._browse_directory(self.report_folder_path, "Select Report Output Folder")).pack(side=tk.LEFT)

        #cleaned CSV Folder
        cleaned_frame = ttk.Frame(master, padding=10)
        cleaned_frame.pack(fill=tk.X)
        ttk.Label(cleaned_frame, text="Cleaned CSV Output Folder:").pack(side=tk.LEFT)
        ttk.Entry(cleaned_frame, textvariable=self.cleaned_folder_path, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(cleaned_frame, text="Browse...", command=lambda: self._browse_directory(self.cleaned_folder_path, "Select Cleaned CSV Output Folder")).pack(side=tk.LEFT)

        #process Button
        self.process_button = ttk.Button(master, text="Start Processing", command=self._start_processing)
        self.process_button.pack(pady=15, ipadx=10, ipady=5)

        #status Area
        ttk.Label(master, text="Processing Log:", style="Header.TLabel").pack(pady=(10,0))
        self.status_area = scrolledtext.ScrolledText(master, height=15, width=80, state=tk.DISABLED, relief=tk.SUNKEN, borderwidth=1, font=('Courier New', 9))
        self.status_area.pack(pady=5, padx=10, expand=True, fill=tk.BOTH)

        #bottom buttons frame
        bottom_frame = ttk.Frame(master, padding=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.quit_button = ttk.Button(bottom_frame, text="Quit", command=master.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5)
        
        self.about_button = ttk.Button(bottom_frame, text="About", command=self._show_about)
        self.about_button.pack(side=tk.RIGHT)


    def _browse_directory(self, path_var, title):
        directory = filedialog.askdirectory(title=title)
        if directory:
            path_var.set(directory)

    def _log_message(self, message):
        self.master.after(0, self._update_gui_log, message)

    def _update_gui_log(self, message):
        self.status_area.config(state=tk.NORMAL)
        self.status_area.insert(tk.END, str(message) + "\n")
        self.status_area.see(tk.END) #scroll to the end
        self.status_area.config(state=tk.DISABLED)
        self.master.update_idletasks() #process pending GUI events, helps with responsiveness

    def _show_about(self):
        messagebox.showinfo("About CSV Data Cleaner++",
                            "CSV Data Cleaner++ v1.0\n\n"
                            "This application processes CSV files to:\n"
                            "- Clean and standardize headers.\n"
                            "- Extract numeric data from mixed-type columns.\n"
                            "- Remove columns that are entirely non-numeric.\n"
                            "- Generate missing value reports.\n"
                            "- Perform mean imputation for numeric columns.\n\n"
                            "Developed by ODAT project.")

    def _start_processing(self):
        input_f = self.input_folder_path.get()
        report_f = self.report_folder_path.get()
        cleaned_f = self.cleaned_folder_path.get()

        if not all([input_f, report_f, cleaned_f]):
            messagebox.showerror("Error", "All folder paths must be specified.")
            return

        if not os.path.isdir(input_f):
            messagebox.showerror("Error", f"Input folder does not exist: {input_f}")
            return

        #clear status area before starting
        self.status_area.config(state=tk.NORMAL)
        self.status_area.delete('1.0', tk.END)
        self.status_area.config(state=tk.DISABLED)

        self._log_message("User initiated processing...")
        self._log_message(f"Input Folder: {input_f}")
        self._log_message(f"Report Folder: {report_f}")
        self._log_message(f"Cleaned Folder: {cleaned_f}")
        
        self.process_button.config(state=tk.DISABLED)
        self.quit_button.config(state=tk.DISABLED) 

        #run processing in a separate thread
        self.processing_thread = threading.Thread(target=self._processing_thread_target,
                                                  args=(input_f, report_f, cleaned_f))
        self.processing_thread.daemon = True 
        self.processing_thread.start()

    def _processing_thread_target(self, input_f, report_f, cleaned_f):
        try:
            main_processing_logic(input_f, report_f, cleaned_f, self._log_message)
        except Exception as e:
            self._log_message(f"An unexpected error occurred in the processing thread: {e}")
            import traceback
            self._log_message(traceback.format_exc())
        finally:
            #ensure buttons are re-enabled in the main thread
            self.master.after(0, self._finalize_processing)

    def _finalize_processing(self):
        self.process_button.config(state=tk.NORMAL)
        self.quit_button.config(state=tk.NORMAL)
        self._log_message("------------------------------------")
        self._log_message("Processing complete. Ready for new task.")
        messagebox.showinfo("Processing Finished", "All CSV files have been processed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVCleanerApp(root)
    root.mainloop()
