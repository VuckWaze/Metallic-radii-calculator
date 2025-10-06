"""
Utility functions for Enhanced Metallic Radii Calculator.

Contains helper functions for file operations, data validation,
mathematical calculations, and formatting.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from config import (
    CRYSTAL_SYSTEM_ALIASES, 
    CALCULATION_TOLERANCES,
    DISPLAY_CONFIG,
    ERROR_MESSAGES
)

class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_crystallographic_data(data: Dict[str, Any]) -> bool:
    """
    Validate crystallographic data for completeness and reasonableness.
    
    Args:
        data: Dictionary containing crystallographic parameters
        
    Returns:
        True if data is valid
        
    Raises:
        ValidationError: If data is invalid
    """
    required_fields = ['a', 'volume', 'z']
    
    # Check required fields
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate numeric ranges
    if data['volume'] < CALCULATION_TOLERANCES['minimum_volume']:
        raise ValidationError(f"Volume too small: {data['volume']:.2f} Ų")
    
    if not (CALCULATION_TOLERANCES['minimum_z'] <= data['z'] <= CALCULATION_TOLERANCES['maximum_z']):
        raise ValidationError(f"Unreasonable Z value: {data['z']}")
    
    if data['a'] <= 0:
        raise ValidationError(f"Invalid lattice parameter a: {data['a']}")
    
    return True

def validate_radius(radius: float, element: str = "Unknown") -> bool:
    """
    Validate calculated radius for reasonableness.
    
    Args:
        radius: Calculated metallic radius
        element: Element symbol for context
        
    Returns:
        True if radius is reasonable
        
    Raises:
        ValidationError: If radius is unreasonable
    """
    min_radius = CALCULATION_TOLERANCES['minimum_radius']
    max_radius = CALCULATION_TOLERANCES['maximum_radius']
    
    if not (min_radius <= radius <= max_radius):
        raise ValidationError(
            ERROR_MESSAGES['invalid_radius'].format(radius=radius)
        )
    
    return True

def clean_crystal_system_name(crystal_system: str) -> str:
    """
    Clean and standardize crystal system names.
    
    Args:
        crystal_system: Raw crystal system string
        
    Returns:
        Standardized crystal system name
    """
    if not crystal_system:
        return 'unknown'
    
    # Handle multiple structures
    if 'Multiple:' in crystal_system:
        systems = crystal_system.replace('Multiple: ', '').split(', ')
        return systems[0].strip()
    
    # Clean the name
    cleaned = crystal_system.strip().lower()
    
    # Check aliases
    for alias, standard in CRYSTAL_SYSTEM_ALIASES.items():
        if alias.lower() in cleaned:
            return standard
    
    return crystal_system.strip()

def extract_element_from_filename(filename: str) -> str:
    """
    Extract element symbol from CIF filename.
    
    Args:
        filename: CIF filename
        
    Returns:
        Element symbol or 'Unknown'
    """
    # Try standard pattern: Element_(structure)_...
    element_match = re.search(r'^([A-Z][a-z]?)(?:[0-9]*|_)', filename)
    if element_match:
        return element_match.group(1)
    
    # Try chemical formula pattern
    formula_match = re.search(r'([A-Z][a-z]?)', filename)
    if formula_match:
        return formula_match.group(1)
    
    return 'Unknown'

def calculate_unit_cell_volume(a: float, b: float, c: float, 
                             alpha: float, beta: float, gamma: float) -> float:
    """
    Calculate unit cell volume from lattice parameters.
    
    Args:
        a, b, c: Lattice parameters in Angstroms
        alpha, beta, gamma: Lattice angles in degrees
        
    Returns:
        Unit cell volume in ų
    """
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # General formula for unit cell volume
    volume = a * b * c * np.sqrt(
        1 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad) -
        np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
    )
    
    return volume

def format_radius(radius: Optional[float]) -> str:
    """
    Format radius for display with appropriate precision.
    
    Args:
        radius: Radius value or None
        
    Returns:
        Formatted radius string
    """
    if radius is None:
        return "N/A"
    
    precision = DISPLAY_CONFIG['radius_precision']
    return f"{radius:.{precision}f}"

def format_volume(volume: Optional[float]) -> str:
    """
    Format volume for display with appropriate precision.
    
    Args:
        volume: Volume value or None
        
    Returns:
        Formatted volume string
    """
    if volume is None:
        return "N/A"
    
    precision = DISPLAY_CONFIG['volume_precision']
    return f"{volume:.{precision}f}"

def format_angle(angle: Optional[float]) -> str:
    """
    Format angle for display with appropriate precision.
    
    Args:
        angle: Angle value or None
        
    Returns:
        Formatted angle string
    """
    if angle is None:
        return "N/A"
    
    precision = DISPLAY_CONFIG['angle_precision']
    return f"{angle:.{precision}f}"

def load_json_file(filepath: Union[str, Path]) -> Dict:
    """
    Safely load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON file or empty dict if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return {}

def save_json_file(data: Dict, filepath: Union[str, Path]) -> bool:
    """
    Safely save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error saving {filepath}: {e}")
        return False

def safe_float_conversion(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert value to float with error handling.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Safely convert value to int with error handling.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def create_directory_if_not_exists(directory: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_file_list(directory: Union[str, Path], pattern: str = "*.cif") -> List[Path]:
    """
    Get list of files matching pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    return list(dir_path.glob(pattern))

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}
    
    values_array = np.array(values)
    
    return {
        'count': len(values),
        'mean': np.mean(values_array),
        'median': np.median(values_array),
        'std': np.std(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array),
        'range': np.max(values_array) - np.min(values_array)
    }

def compare_values(value1: float, value2: float, tolerance: float = 0.001) -> bool:
    """
    Compare two values within a tolerance.
    
    Args:
        value1, value2: Values to compare
        tolerance: Absolute tolerance for comparison
        
    Returns:
        True if values are within tolerance
    """
    return abs(value1 - value2) <= tolerance

def is_metallic_element_category(category: str) -> bool:
    """
    Check if element category represents metallic bonding.
    
    Args:
        category: Element category string
        
    Returns:
        True if category is metallic
    """
    from config import METALLIC_ELEMENT_CATEGORIES
    return category in METALLIC_ELEMENT_CATEGORIES

def is_outlier_element(element: str) -> bool:
    """
    Check if element is known to be an outlier.
    
    Args:
        element: Element symbol
        
    Returns:
        True if element is a known outlier
    """
    from config import OUTLIER_ELEMENTS
    return element in OUTLIER_ELEMENTS

def progress_indicator(current: int, total: int, interval: int = 10) -> None:
    """
    Print progress indicator at specified intervals.
    
    Args:
        current: Current item number
        total: Total number of items
        interval: Progress reporting interval
    """
    if current % interval == 0 or current == total:
        percentage = (current / total) * 100
        print(f"   Progress: {current}/{total} ({percentage:.1f}%)")

class CrystalSystemDetector:
    """Helper class for detecting crystal systems from various sources."""
    
    @staticmethod
    def from_filename(filename: str) -> str:
        """Extract crystal system from filename."""
        patterns = [
            r'_\(([^)]+)\)_',  # Standard pattern (element)_(system)_
            r'_(α-[^_]+)_',    # Alpha phases
            r'_(β-[^_]+)_',    # Beta phases  
            r'_(γ-[^_]+)_',    # Gamma phases
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return 'unknown'
    
    @staticmethod
    def from_space_group(space_group: str, space_group_number: Optional[int] = None) -> str:
        """Infer crystal system from space group."""
        if not space_group:
            return 'unknown'
        
        sg = space_group.strip().lower()
        
        # Common space group patterns
        if any(pattern in sg for pattern in ['fm3m', 'fm-3m', 'f m 3 m']):
            return 'fcc'
        elif any(pattern in sg for pattern in ['im3m', 'im-3m', 'i m 3 m']):
            return 'bcc'
        elif any(pattern in sg for pattern in ['p63/mmc', 'p 63/m m c']):
            return 'hcp'
        elif any(pattern in sg for pattern in ['fd3m', 'fd-3m']):
            return 'diamond'
        
        # Use space group number if available
        if space_group_number:
            if space_group_number == 225:
                return 'fcc'
            elif space_group_number == 229:
                return 'bcc'
            elif space_group_number == 194:
                return 'hcp'
            elif space_group_number == 227:
                return 'diamond'
        
        return 'unknown'
    
    @staticmethod
    def from_lattice_parameters(a: float, b: Optional[float], c: Optional[float],
                              alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0) -> str:
        """Infer crystal system from lattice parameters."""
        tolerance = CALCULATION_TOLERANCES['length_tolerance']
        angle_tolerance = CALCULATION_TOLERANCES['angle_tolerance']
        
        b = b or a
        c = c or a
        
        # Check if angles are 90 degrees
        angles_90 = all(abs(angle - 90.0) < angle_tolerance for angle in [alpha, beta, gamma])
        
        # Cubic systems
        if angles_90 and compare_values(a, b, tolerance) and compare_values(b, c, tolerance):
            return 'cubic'  # Could be fcc, bcc, or simple cubic
        
        # Tetragonal
        if angles_90 and compare_values(a, b, tolerance) and not compare_values(a, c, tolerance):
            return 'tetragonal'
        
        # Orthorhombic
        if angles_90 and not any(compare_values(x, y, tolerance) for x, y in [(a, b), (b, c), (a, c)]):
            return 'orthorhombic'
        
        # Hexagonal (gamma = 120 degrees)
        if angles_90 and abs(gamma - 120.0) < angle_tolerance and compare_values(a, b, tolerance):
            return 'hexagonal'
        
        return 'unknown'