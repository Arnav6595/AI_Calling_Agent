import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm
import glob

# --- SETUP ---
# Load environment variables from .env file.
# The script will look for the .env file in its own directory, or parent directories.
load_dotenv()
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

def discover_csv_files_and_map_tables(script_dir: str) -> dict:
    """
    Automatically discovers all .csv files in the script's directory
    and creates a mapping to database table names.
    """
    mapping = {}
    # Use glob to find all files ending with .csv in the script's directory
    csv_files = glob.glob(os.path.join(script_dir, '*.csv'))
    
    for csv_path in csv_files:
        # Get the filename without the extension (e.g., "Ongoing_Projects")
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # Create a clean table name (e.g., "ongoing_projects_source")
        table_name = f"{base_name.lower().replace(' ', '_')}_source"
        mapping[csv_path] = table_name
        
    return mapping

def upload_csvs_to_database(db_url: str, mapping: dict):
    """
    Connects to the database, reads each CSV, and uploads it to a specified table.
    """
    if not db_url:
        print("Error: NEON_DATABASE_URL not found in the .env file.")
        return
    
    if not mapping:
        print("No CSV files found to upload in the script's directory.")
        return

    print("Connecting to the database...")
    try:
        engine = create_engine(db_url)
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return

    # Use tqdm to create a progress bar for the files
    for csv_path, table_name in tqdm(mapping.items(), desc="Uploading CSVs to Database"):
        try:
            # Read the local CSV file into a pandas DataFrame
            df = pd.read_csv(csv_path)
            
            # Use pandas' to_sql method to upload the DataFrame to the database.
            # 'if_exists="replace"' will drop the table if it already exists and create a new one.
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            
            tqdm.write(f"- Successfully uploaded {csv_path} to table '{table_name}' ({len(df)} rows).")

        except FileNotFoundError:
            # This case is less likely with the new discovery method but kept for safety
            tqdm.write(f"- Warning: File not found, skipping: {csv_path}")
        except Exception as e:
            tqdm.write(f"- Error processing {csv_path}: {e}")
            
    print("\n--- CSV upload process complete. ---")


if __name__ == "__main__":
    # Get the directory where this script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Automatically discover CSV files and create the mapping
    csv_mapping = discover_csv_files_and_map_tables(script_directory)
    
    # Run the upload process with the dynamically generated mapping
    upload_csvs_to_database(NEON_DATABASE_URL, csv_mapping)

