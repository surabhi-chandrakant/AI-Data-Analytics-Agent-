import pandas as pd
import numpy as np
import json
from pathlib import Path
import io
import chardet

class DataLoader:
    """Handle loading of various data formats with robust error handling"""
    
    @staticmethod
    def load(file_obj):
        """Load data from uploaded file with robust error handling"""
        file_extension = Path(file_obj.name).suffix.lower()
        
        try:
            if file_extension == '.csv':
                return DataLoader._load_csv(file_obj)
            elif file_extension in ['.xlsx', '.xls']:
                return DataLoader._load_excel(file_obj)
            elif file_extension == '.json':
                return DataLoader._load_json(file_obj)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Please upload CSV, Excel, or JSON files.")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def _load_csv(file_obj):
        """Load CSV with multiple encoding and format attempts"""
        # Read the file content
        content = file_obj.read()
        
        # Try to detect encoding
        try:
            detected = chardet.detect(content)
            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        except:
            encoding = 'utf-8'
        
        # Reset file pointer
        file_obj.seek(0)
        
        # Try different approaches to load CSV
        approaches = [
            # Approach 1: Default pandas read_csv
            lambda: pd.read_csv(file_obj, encoding=encoding),
            
            # Approach 2: Try different encoding
            lambda: pd.read_csv(file_obj, encoding='latin1'),
            
            # Approach 3: Try with error handling
            lambda: pd.read_csv(file_obj, encoding=encoding, on_bad_lines='skip'),
            
            # Approach 4: Try with different separator
            lambda: pd.read_csv(file_obj, encoding=encoding, sep=';'),
            
            # Approach 5: Try with engine='python'
            lambda: pd.read_csv(file_obj, encoding=encoding, engine='python'),
            
            # Approach 6: Try reading as text and parsing manually
            lambda: DataLoader._parse_malformed_csv(content, encoding),
        ]
        
        for i, approach in enumerate(approaches):
            try:
                # Reset file pointer for each attempt
                file_obj.seek(0)
                df = approach()
                
                # Check if we got a valid dataframe
                if len(df) > 0 and len(df.columns) > 0:
                    return df
            except Exception as e:
                continue
        
        raise Exception("Could not parse CSV file. Please check the file format.")
    
    @staticmethod
    def _parse_malformed_csv(content, encoding):
        """Parse malformed CSV by reading line by line"""
        try:
            # Decode content
            text_content = content.decode(encoding, errors='ignore')
            lines = text_content.split('\n')
            
            # Find the line with most commas (likely the header)
            max_commas = 0
            header_line = 0
            for i, line in enumerate(lines[:20]):  # Check first 20 lines
                commas = line.count(',')
                if commas > max_commas:
                    max_commas = commas
                    header_line = i
            
            # Parse using StringIO
            clean_lines = []
            for line in lines[header_line:]:
                if line.strip():
                    # Ensure consistent number of columns
                    parts = line.split(',')
                    if len(parts) >= max_commas - 1:  # Allow some variation
                        clean_lines.append(','.join(parts[:max_commas + 1]))
            
            clean_content = '\n'.join(clean_lines)
            return pd.read_csv(io.StringIO(clean_content))
        except Exception as e:
            raise Exception(f"Could not parse malformed CSV: {str(e)}")
    
    @staticmethod
    def _load_excel(file_obj):
        """Load Excel file with multiple engine attempts"""
        engines = ['openpyxl', 'xlrd']
        
        for engine in engines:
            try:
                return pd.read_excel(file_obj, engine=engine)
            except:
                continue
        
        # Try reading all sheets and combining
        try:
            excel_file = pd.ExcelFile(file_obj)
            dfs = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_obj, sheet_name=sheet_name)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        except Exception as e:
            raise Exception(f"Could not parse Excel file: {str(e)}")
    
    @staticmethod
    def _load_json(file_obj):
        """Load JSON file with different formats"""
        try:
            # Try standard JSON
            return pd.read_json(file_obj)
        except:
            try:
                # Try lines format
                file_obj.seek(0)
                return pd.read_json(file_obj, lines=True)
            except:
                try:
                    # Try reading as string and parsing
                    content = file_obj.read()
                    data = json.loads(content)
                    return pd.json_normalize(data)
                except Exception as e:
                    raise Exception(f"Could not parse JSON file: {str(e)}")
    
    @staticmethod
    def validate_dataset(df):
        """Validate dataset for analysis"""
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if len(df) < 2:
            raise ValueError("Dataset has too few rows for analysis (need at least 2 rows)")
        
        if len(df.columns) < 1:
            raise ValueError("Dataset has no columns")
        
        return True