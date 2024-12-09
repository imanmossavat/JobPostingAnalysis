### **Team Design Rules for Job Posting Analysis Integration**

#### **Specific Design Choices**
1. **Configuration Handling**:
   - Each contributor must implement a `Config` class that contains all default values and settings.
   - Configurations should be separated from logic, allowing easy modification.
   - Example structure for `Config`:
     ```python
     class Config:
         def __init__(self):
             self.default_setting_1 = "value"
             self.default_setting_2 = 42
     ```

2. **Interface Classes**:
   - Each contributor must implement an `Interface` class as the single entry point to their code.
   - Interfaces should adhere to a predefined API defined in an abstract base class (ABC).
   - Example using an ABC:
     ```python
     from abc import ABC, abstractmethod

     class InterfaceBase(ABC):
         @abstractmethod
         def process_data(self, data):
             pass
     ```

3. **Data Formatting**:
   - A shared `DataFormatter` class must be used to standardize data between contributors.
   - The formatter should handle missing columns (e.g., filling with `NA` or obfuscating).
   - Contributors must agree on a standard format (e.g., column names, data types).
   - Example:
     ```python
     class DataFormatter:
         def format_data(self, data):
             # Standardize column names
             standardized_data = data.rename(columns={"colA": "standard_col"})
             # Fill missing columns
             for col in ["standard_col", "extra_col"]:
                 if col not in standardized_data:
                     standardized_data[col] = "NA"
             return standardized_data
     ```

4. **Mother Interface**:
   - A central `MotherInterface` class should mediate between individual interfaces and the UI.
   - It must handle:
     - Aggregating configurations.
     - Standardizing data via `DataFormatter`.
     - Dynamically enabling/disabling features based on data availability.

5. **Selective Feature Availability**:
   - Functions that depend on specific data must dynamically check for required fields and disable themselves gracefully.
   - Example:
     ```python
     class FeatureManager:
         def __init__(self, data):
             self.available_features = {
                 "feature_x": "column_x" in data.columns,
                 "feature_y": "column_y" in data.columns,
             }

         def is_feature_available(self, feature_name):
             return self.available_features.get(feature_name, False)
     ```

6. **Testing**:
   - All modules must include:
     - Unit tests to verify their components.
     - Integration tests for compatibility with the shared framework.
   - Use Python's `unittest` or `pytest`.

---

#### **Generic Rules**

1. **Follow PEP 8**:
   - Write clean, readable, and standardized Python code.
   - Examples:
     - Use `snake_case` for function and variable names.
     - Limit lines to 79 characters.
     - Include docstrings for all classes and methods.

2. **Type Annotations**:
   - Use type annotations for all method arguments and return values to improve code clarity.
   - Example:
     ```python
     def process_data(data: pd.DataFrame) -> pd.DataFrame:
         pass
     ```

3. **Documentation**:
   - Each module must include:
     - A README explaining its purpose, usage, and configuration.
     - Docstrings for all methods and classes using Google-style or NumPy-style.

4. **Error Handling**:
   - Use Python’s built-in error classes to raise meaningful exceptions.
   - Avoid bare `except` statements; always catch specific exceptions.

5. **Extensibility**:
   - Write modular, reusable components with future contributors in mind.
   - Avoid hardcoding logic; instead, use configuration or input arguments.

6. **Version Control**:
   - Commit frequently with meaningful messages.
   - Use feature branches for individual work and merge via pull requests after code review.

7. **Collaboration**:
   - Hold regular sync-ups to agree on shared design decisions like standard column names.
   - Maintain clear communication about changes that affect shared components.

8. **Use Specialized Python Classes When Appropriate**:
   - Prefer `dataclasses` for lightweight data storage classes:
     ```python
     from dataclasses import dataclass

     @dataclass
     class JobPost:
         title: str
         description: str
         salary: float = 0.0
     ```
   - Use `NamedTuple` for immutable, structured data:
     ```python
     from typing import NamedTuple

     class JobPost(NamedTuple):
         title: str
         description: str
         salary: float
     ```

