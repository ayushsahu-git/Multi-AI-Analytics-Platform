# import pandas as pd
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from typing import List, Dict, Any, Union, Tuple, Optional
# import os
# import json


# class DataLoader:
#     def __init__(self):
#         self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
#         self.supported_text_formats = ['.txt', '.csv', '.json', '.xlsx', '.xls']

#     def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
#         return pd.read_csv(file_path)

#     def load_excel(self, file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
#         return pd.read_excel(file_path, sheet_name=sheet_name)

#     def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
#         return pd.read_json(file_path)

#     def load_image(self, file_path: Union[str, Path]) -> Image.Image:
#         return Image.open(file_path).convert('RGB')

#     def load_images_from_folder(self, folder_path: Union[str, Path]) -> List[Tuple[str, Image.Image]]:
#         folder = Path(folder_path)
#         images = []
#         for ext in self.supported_image_formats:
#             for file_path in folder.glob(f"*{ext}"):
#                 try:
#                     img = Image.open(file_path).convert('RGB')
#                     images.append((str(file_path), img))
#                 except Exception as e:
#                     print(f"Error loading {file_path}: {e}")
#         return images

#     def load_text_file(self, file_path: Union[str, Path]) -> str:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()

#     def detect_file_type(self, file_path: Union[str, Path]) -> str:
#         path = Path(file_path)
#         suffix = path.suffix.lower()
#         if suffix in self.supported_image_formats:
#             return "image"
#         elif suffix == '.csv':
#             return "csv"
#         elif suffix in ['.xlsx', '.xls']:
#             return "excel"
#         elif suffix == '.json':
#             return "json"
#         elif suffix == '.txt':
#             return "text"
#         else:
#             return "unknown"

#     def auto_load(self, file_path: Union[str, Path]) -> Tuple[Any, str]:
#         file_type = self.detect_file_type(file_path)
#         if file_type == "csv":
#             return self.load_csv(file_path), "dataframe"
#         elif file_type == "excel":
#             return self.load_excel(file_path), "dataframe"
#         elif file_type == "json":
#             return self.load_json(file_path), "dataframe"
#         elif file_type == "image":
#             return self.load_image(file_path), "image"
#         elif file_type == "text":
#             return self.load_text_file(file_path), "text"
#         else:
#             raise ValueError(f"Unsupported file type: {file_type}")

#     def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
#         summary = {
#             "row_count": int(len(df)),
#             "columns": df.columns.tolist(),
#             "features": int(len(df.columns)),
#             "dtypes": df.dtypes.astype(str).to_dict(),
#             "missing_values": df.isnull().sum().to_dict(),
#             "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
#             "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
#             "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
#             "duplicate_rows": int(df.duplicated().sum()),
#         }

#         numeric_df = df.select_dtypes(include=[np.number])
#         if not numeric_df.empty:
#             summary["numeric_summary"] = {
#                 "mean": numeric_df.mean().round(4).to_dict(),
#                 "std": numeric_df.std().round(4).to_dict(),
#                 "min": numeric_df.min().to_dict(),
#                 "max": numeric_df.max().to_dict(),
#                 "median": numeric_df.median().to_dict(),
#             }

#         return summary

#     def preprocess_dataframe(
#         self,
#         df: pd.DataFrame,
#         drop_non_numeric: bool = True,
#         fill_strategy: str = "median"
#     ) -> pd.DataFrame:
#         df = df.copy()

#         df = df.dropna(axis=1, how='all')

#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 try:
#                     df[col] = pd.to_numeric(df[col])
#                 except (ValueError, TypeError):
#                     if drop_non_numeric:
#                         df = df.drop(columns=[col])
#                     else:
#                         df = pd.get_dummies(df, columns=[col], drop_first=True)

#         for col in df.columns:
#             if df[col].isnull().any():
#                 if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
#                     if fill_strategy == "median":
#                         df[col] = df[col].fillna(df[col].median())
#                     elif fill_strategy == "mean":
#                         df[col] = df[col].fillna(df[col].mean())
#                     else:
#                         df[col] = df[col].fillna(0)
#                 elif df[col].dtype == 'bool':
#                     df[col] = df[col].fillna(False)
#                 else:
#                     mode_val = df[col].mode()
#                     df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")

#         return df

#     def split_features_target(
#         self, df: pd.DataFrame, target_column: str
#     ) -> Tuple[pd.DataFrame, pd.Series]:
#         if target_column not in df.columns:
#             raise ValueError(f"Target column '{target_column}' not found in dataframe")
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
#         return X, y

#     def get_class_distribution(self, series: pd.Series) -> Dict[str, int]:
#         return series.value_counts().to_dict()

#     def detect_task_type(self, series: pd.Series) -> str:
#         """Auto-detect whether classification or regression is appropriate."""
#         if series.dtype == 'object' or series.nunique() <= 20:
#             return "classification"
#         return "regression"



import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import os
import json


class DataLoader:
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        self.supported_text_formats = ['.txt', '.csv', '.json', '.xlsx', '.xls']

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def load_excel(self, file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_json(file_path)

    def load_image(self, file_path: Union[str, Path]) -> Image.Image:
        return Image.open(file_path).convert('RGB')

    def load_images_from_folder(self, folder_path: Union[str, Path]) -> List[Tuple[str, Image.Image]]:
        folder = Path(folder_path)
        images = []
        for ext in self.supported_image_formats:
            for file_path in folder.glob(f"*{ext}"):
                try:
                    img = Image.open(file_path).convert('RGB')
                    images.append((str(file_path), img))
                except Exception:
                    continue
        return images

    def load_text_file(self, file_path: Union[str, Path]) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in self.supported_image_formats:
            return "image"
        elif suffix == '.csv':
            return "csv"
        elif suffix in ['.xlsx', '.xls']:
            return "excel"
        elif suffix == '.json':
            return "json"
        elif suffix == '.txt':
            return "text"
        else:
            return "unknown"

    def auto_load(self, file_path: Union[str, Path]) -> Tuple[Any, str]:
        file_type = self.detect_file_type(file_path)
        if file_type == "csv":
            return self.load_csv(file_path), "dataframe"
        elif file_type == "excel":
            return self.load_excel(file_path), "dataframe"
        elif file_type == "json":
            return self.load_json(file_path), "dataframe"
        elif file_type == "image":
            return self.load_image(file_path), "image"
        elif file_type == "text":
            return self.load_text_file(file_path), "text"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary = {
            "row_count": int(len(df)),
            "columns": df.columns.tolist(),
            "features": int(len(df.columns)),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary["numeric_summary"] = {
                "mean": numeric_df.mean().round(4).to_dict(),
                "std": numeric_df.std().round(4).to_dict(),
                "min": numeric_df.min().to_dict(),
                "max": numeric_df.max().to_dict(),
                "median": numeric_df.median().to_dict(),
            }

        return summary

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        drop_non_numeric: bool = True,
        fill_strategy: str = "median"
    ) -> pd.DataFrame:
        df = df.copy()

        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    if drop_non_numeric:
                        df = df.drop(columns=[col])
                    else:
                        df = pd.get_dummies(df, columns=[col], drop_first=True)

        # Fill missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    if fill_strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif fill_strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(0)
                elif df[col].dtype == 'bool':
                    df[col] = df[col].fillna(False)
                else:
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")

        return df

    def split_features_target(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def get_class_distribution(self, series: pd.Series) -> Dict[str, int]:
        return series.value_counts().to_dict()

    def detect_task_type(self, series: pd.Series) -> str:
        """Auto-detect whether classification or regression is appropriate."""
        if series.dtype == 'object' or series.nunique() <= 20:
            return "classification"
        return "regression"
