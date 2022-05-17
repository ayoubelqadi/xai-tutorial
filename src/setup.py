import os 

needed_libraries = [
    "pandas",
    "numpy",
    "shap",
    "seaborn",
    "sklearn",
    "category_encoders",
    "xgboost",
    "dice-ml"
]

for pack in needed_libraries: 
    os.system(f"pip install {pack}")