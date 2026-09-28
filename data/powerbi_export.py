# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# from pathlib import Path
# from typing import Dict, Any, List, Union
# import json
# from datetime import datetime


# class PowerBIExporter:
#     def __init__(self, output_dir: Union[str, Path]):
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.exported_files = []
        
#     def export_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
#         output_path = self.output_dir / f"{filename}.csv"
#         df.to_csv(output_path, index=False)
#         self.exported_files.append(output_path)
#         return output_path
    
#     def export_to_parquet(self, df: pd.DataFrame, filename: str) -> Path:
#         output_path = self.output_dir / f"{filename}.parquet"
#         df.to_parquet(output_path, index=False, engine='pyarrow')
#         self.exported_files.append(output_path)
#         return output_path
    
#     def export_to_json(self, data: Any, filename: str) -> Path:
#         output_path = self.output_dir / f"{filename}.json"
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, default=str)
            
#         self.exported_files.append(output_path)
#         return output_path
    
#     def create_data_model(self, tables: Dict[str, pd.DataFrame], relationships: List[Dict[str, str]] = None) -> Dict[str, Any]:
#         data_model = {
#             "tables": {},
#             "relationships": relationships or [],
#             "created_at": datetime.now().isoformat()
#         }
        
#         for table_name, df in tables.items():
#             data_model["tables"][table_name] = {
#                 "columns": df.columns.tolist(),
#                 "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
#                 "row_count": len(df),
#                 "primary_key": df.columns[0] if len(df.columns) > 0 else None
#             }
            
#         model_path = self.export_to_json(data_model, "powerbi_data_model")
#         return data_model
    
#     def create_analysis_results(self, ml_results: Dict[str, Any], dl_results: Dict[str, Any], 
#                                data_summary: Dict[str, Any]) -> pd.DataFrame:
#         results_df = pd.DataFrame([
#             {
#                 "metric_category": "Machine Learning",
#                 "metric_name": "accuracy" if "accuracy" in ml_results else "mse",
#                 "metric_value": ml_results.get("accuracy", ml_results.get("mse", 0)),
#                 "timestamp": datetime.now()
#             },
#             {
#                 "metric_category": "Deep Learning",
#                 "metric_name": "device",
#                 "metric_value": dl_results.get("device", "unknown"),
#                 "timestamp": datetime.now()
#             },
#             {
#                 "metric_category": "Data Summary",
#                 "metric_name": "row_count",
#                 "metric_value": data_summary.get("row_count", 0),
#                 "timestamp": datetime.now()
#             }
#         ])
        
#         return results_df
    
#     def export_predictions(self, df: pd.DataFrame, predictions: List[Any], 
#                            probabilities: List[List[float]] = None, filename: str = "predictions") -> Path:
#         result_df = df.copy()
#         result_df["prediction"] = predictions
        
#         if probabilities:
#             for i, probs in enumerate(zip(*probabilities)):
#                 result_df[f"prob_class_{i}"] = probs
                
#         return self.export_to_csv(result_df, filename)
    
#     def create_dashboard_data(self, analysis_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
#         dashboard_data = {}
        
#         if "feature_importance" in analysis_results:
#             dashboard_data["feature_importance"] = pd.DataFrame(analysis_results["feature_importance"])
            
#         if "predictions" in analysis_results:
#             dashboard_data["predictions"] = pd.DataFrame(analysis_results["predictions"])
            
#         if "metrics" in analysis_results:
#             metrics_list = []
#             for key, value in analysis_results["metrics"].items():
#                 if isinstance(value, (int, float)):
#                     metrics_list.append({"metric": key, "value": value})
#             if metrics_list:
#                 dashboard_data["metrics_summary"] = pd.DataFrame(metrics_list)
                
#         return dashboard_data
    
#     def export_all(self, dataframes: Dict[str, pd.DataFrame], include_parquet: bool = True) -> List[Path]:
#         exported = []
        
#         for name, df in dataframes.items():
#             csv_path = self.export_to_csv(df, name)
#             exported.append(csv_path)
            
#             if include_parquet:
#                 parquet_path = self.export_to_parquet(df, name)
#                 exported.append(parquet_path)
                
#         return exported
    
#     def get_exported_files(self) -> List[Path]:
#         return self.exported_files
    
#     def generate_powerbi_instructions(self) -> str:
#         instructions = """
#         Power BI Integration Instructions:
#         ================================
        
#         1. Open Power BI Desktop
        
#         2. Get Data:
#            - Click "Get Data" > "More..."
#            - Select "Text/CSV" for CSV files
#            - Select "Parquet" for Parquet files
        
#         3. Load the exported data:
#            - Navigate to the 'output' folder
#            - Select the relevant CSV/Parquet files
        
#         4. Create relationships:
#            - Open "Model" view
#            - Drag columns to create relationships between tables
        
#         5. Build visualizations:
#            - Use the "Visualizations" pane
#            - Create charts, tables, and KPIs
        
#         Exported files are located in: {output_dir}
#         """.format(output_dir=str(self.output_dir))
        
#         return instructions






import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import json
import re
from datetime import datetime


class PowerBIExporter:
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exported_files: List[Path] = []

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", filename):
            raise ValueError("Invalid export name")

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        self._validate_filename(filename)
        output_path = self.output_dir / f"{filename}.csv"
        df.to_csv(output_path, index=False)
        self.exported_files.append(output_path)
        return output_path

    def export_to_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        self._validate_filename(filename)
        try:
            import pyarrow  # noqa
            output_path = self.output_dir / f"{filename}.parquet"
            df.to_parquet(output_path, index=False, engine='pyarrow')
            self.exported_files.append(output_path)
            return output_path
        except ImportError:
            # Fallback to CSV if pyarrow not installed
            return self.export_to_csv(df, filename + "_parquet_fallback")

    def export_to_json(self, data: Any, filename: str) -> Path:
        self._validate_filename(filename)
        output_path = self.output_dir / f"{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        self.exported_files.append(output_path)
        return output_path

    def create_data_model(
        self,
        tables: Dict[str, pd.DataFrame],
        relationships: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        data_model: Dict[str, Any] = {
            "tables": {},
            "relationships": relationships or [],
            "created_at": datetime.now().isoformat(),
        }
        for table_name, df in tables.items():
            data_model["tables"][table_name] = {
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "primary_key": df.columns[0] if len(df.columns) > 0 else None,
            }
        self.export_to_json(data_model, "powerbi_data_model")
        return data_model

    def create_analysis_results(
        self,
        ml_results: Dict[str, Any],
        dl_results: Dict[str, Any],
        data_summary: Dict[str, Any],
    ) -> pd.DataFrame:
        rows = [
            {
                "metric_category": "Machine Learning",
                "metric_name": "accuracy" if "accuracy" in ml_results else "mse",
                "metric_value": ml_results.get("accuracy", ml_results.get("mse", 0)),
                "timestamp": datetime.now(),
            },
            {
                "metric_category": "Deep Learning",
                "metric_name": "device",
                "metric_value": str(dl_results.get("device", "unknown")),
                "timestamp": datetime.now(),
            },
            {
                "metric_category": "Data Summary",
                "metric_name": "row_count",
                "metric_value": data_summary.get("row_count", 0),
                "timestamp": datetime.now(),
            },
        ]
        return pd.DataFrame(rows)

    def export_predictions(
        self,
        df: pd.DataFrame,
        predictions: List[Any],
        probabilities: Optional[List[List[float]]] = None,
        filename: str = "predictions",
    ) -> Path:
        result_df = df.copy()
        result_df["prediction"] = predictions
        if probabilities is not None:
            prob_array = list(zip(*probabilities))
            for i, probs in enumerate(prob_array):
                result_df[f"prob_class_{i}"] = probs
        return self.export_to_csv(result_df, filename)

    def create_dashboard_data(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        dashboard_data: Dict[str, pd.DataFrame] = {}
        if "feature_importance" in analysis_results:
            dashboard_data["feature_importance"] = pd.DataFrame(
                analysis_results["feature_importance"]
            )
        if "predictions" in analysis_results:
            dashboard_data["predictions"] = pd.DataFrame(
                analysis_results["predictions"]
            )
        if "metrics" in analysis_results:
            metrics_list = [
                {"metric": k, "value": v}
                for k, v in analysis_results["metrics"].items()
                if isinstance(v, (int, float))
            ]
            if metrics_list:
                dashboard_data["metrics_summary"] = pd.DataFrame(metrics_list)
        return dashboard_data

    def export_all(
        self,
        dataframes: Dict[str, pd.DataFrame],
        include_parquet: bool = True,
    ) -> List[Path]:
        exported: List[Path] = []
        for name, df in dataframes.items():
            exported.append(self.export_to_csv(df, name))
            if include_parquet:
                exported.append(self.export_to_parquet(df, name))
        return exported

    def get_exported_files(self) -> List[Path]:
        return self.exported_files

    def generate_powerbi_instructions(self) -> str:
        return f"""
Power BI Integration Instructions
===================================

1. Open Power BI Desktop

2. Get Data:
   - Click "Get Data" → "More..."
   - Select "Text/CSV" for CSV files
   - Select "Parquet" for Parquet files

3. Load the exported data:
   - Navigate to: {self.output_dir}
   - Select the relevant CSV/Parquet files

4. Create relationships (Model view):
   - Drag shared columns between tables to link them

5. Build visualizations:
   - Use the "Visualizations" pane to create charts, KPIs, tables

Exported files location: {self.output_dir}
Total files exported: {len(self.exported_files)}
"""
