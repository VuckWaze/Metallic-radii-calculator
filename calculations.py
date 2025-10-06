"""
Calculation methods for metallic radii from crystal structures.

Contains traditional structure-specific calculation methods and
the universal volume-based approach with correction functions.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from utils import validate_radius, ValidationError

class RadiusCalculationMethods:
    """
    Collection of radius calculation methods for different crystal structures.
    
    Each method calculates the metallic radius based on the specific
    geometry and bonding in the crystal structure.
    """
    
    @staticmethod
    def face_centered_cubic(a: float, **kwargs) -> float:
        """
        Calculate radius for face-centered cubic (fcc) structure.
        
        In fcc, atoms touch along the face diagonal: 4r = a√2
        
        Args:
            a: Lattice parameter
            
        Returns:
            Metallic radius in Angstroms
        """
        return a / (2 * np.sqrt(2))
    
    @staticmethod
    def hexagonal_close_packed(a: float, c: Optional[float] = None, **kwargs) -> float:
        """
        Calculate radius for hexagonal close-packed (hcp) structure.
        
        In hcp, atoms touch along the basal plane: 2r = a
        
        Args:
            a: Basal plane lattice parameter
            c: c-axis parameter (not used in calculation)
            
        Returns:
            Metallic radius in Angstroms
        """
        return a / 2
    
    @staticmethod
    def body_centered_cubic(a: float, **kwargs) -> float:
        """
        Calculate radius for body-centered cubic (bcc) structure.
        
        In bcc, atoms touch along the body diagonal: 4r = a√3
        
        Args:
            a: Lattice parameter
            
        Returns:
            Metallic radius in Angstroms
        """
        return a * np.sqrt(3) / 4
    
    @staticmethod
    def alpha_lanthanum(a: float, c: Optional[float] = None, **kwargs) -> float:
        """
        Calculate radius for double hexagonal close-packed (α-La type) structure.
        
        Similar to hcp but with doubled c-axis. Radius calculation
        uses basal plane parameter.
        
        Args:
            a: Basal plane lattice parameter
            c: c-axis parameter (doubled compared to hcp)
            
        Returns:
            Metallic radius in Angstroms
        """
        return a / 2
    
    @staticmethod
    def diamond_cubic(a: float, **kwargs) -> float:
        """
        Calculate radius for diamond cubic structure.
        
        In diamond structure: 8r = a√3
        
        Args:
            a: Lattice parameter
            
        Returns:
            Covalent radius in Angstroms
        """
        return a * np.sqrt(3) / 8
    
    @staticmethod
    def tetragonal_protactinium(a: float, c: float, **kwargs) -> float:
        """
        Calculate radius for tetragonal protactinium structure.
        
        Fixed formula for Pa: r = √(2a² + c²) / 4
        
        Args:
            a: Basal plane lattice parameter
            c: c-axis parameter
            
        Returns:
            Metallic radius in Angstroms
        """
        return np.sqrt(2 * a**2 + c**2) / 4
    
    @staticmethod
    def orthorhombic_neptunium(a: float, b: float, c: float, **kwargs) -> float:
        """
        Calculate radius for orthorhombic neptunium structure.
        
        Uses geometric mean of nearest-neighbor distances.
        
        Args:
            a, b, c: Lattice parameters
            
        Returns:
            Metallic radius in Angstroms
        """
        # Approximate using average of shortest distances
        return min(a, b, c) / 2
    
    @staticmethod
    def uranium_structure(a: float, b: float, c: float, **kwargs) -> float:
        """
        Calculate radius for orthorhombic uranium structure.
        
        Complex structure with multiple nearest neighbors.
        
        Args:
            a, b, c: Lattice parameters
            
        Returns:
            Metallic radius in Angstroms
        """
        # Uranium has a complex structure with varying bond lengths
        # Use empirical relationship
        return 0.276 * (a * b * c)**(1/3)
    
    @staticmethod
    def alpha_manganese(a: float, **kwargs) -> float:
        """
        Calculate radius for complex cubic α-manganese structure.
        
        Very complex structure with 58 atoms per unit cell.
        Uses empirical value.
        
        Args:
            a: Lattice parameter
            
        Returns:
            Metallic radius in Angstroms
        """
        # Complex structure requires empirical approach
        return 1.286  # Empirical value for α-Mn
    
    @staticmethod
    def beta_tin(a: float, c: float, **kwargs) -> float:
        """
        Calculate radius for β-tin (white tin) structure.
        
        Tetragonal structure with 4 atoms per unit cell.
        
        Args:
            a: Basal plane lattice parameter
            c: c-axis parameter
            
        Returns:
            Metallic radius in Angstroms
        """
        # Distance to nearest neighbors in tetragonal structure
        return a / 2
    
    @staticmethod
    def arsenic_structure(a: float, c: Optional[float] = None, **kwargs) -> float:
        """
        Calculate radius for rhombohedral arsenic structure.
        
        Layered structure with covalent bonding.
        
        Args:
            a: Lattice parameter
            c: c-axis parameter
            
        Returns:
            Covalent radius in Angstroms
        """
        # Empirical relationship for As structure
        return a * 0.25  # Approximate based on As-As distances
    
    @staticmethod
    def selenium_chains(a: float, b: Optional[float] = None, c: Optional[float] = None, **kwargs) -> float:
        """
        Calculate radius for chain-like selenium structure.
        
        Monoclinic structure with Se-Se chains.
        
        Args:
            a, b, c: Lattice parameters
            
        Returns:
            Covalent radius in Angstroms
        """
        # Use minimum lattice parameter as approximation
        params = [p for p in [a, b, c] if p is not None]
        return min(params) / 2 if params else a / 2
    
    @staticmethod
    def molecular_crystal(a: float, **kwargs) -> float:
        """
        Calculate radius for simple molecular crystals.
        
        Van der Waals or weak molecular interactions.
        
        Args:
            a: Lattice parameter
            
        Returns:
            van der Waals radius in Angstroms
        """
        # Simple approximation for molecular crystals
        return a / 2
    
    @staticmethod
    def generic_structure(a: float, b: Optional[float] = None, c: Optional[float] = None, **kwargs) -> float:
        """
        Generic calculation for unknown structures.
        
        Uses simple geometric approximation.
        
        Args:
            a, b, c: Lattice parameters
            
        Returns:
            Approximate radius in Angstroms
        """
        # Use minimum lattice parameter as conservative estimate
        params = [p for p in [a, b, c] if p is not None]
        return min(params) / 2 if params else a / 2

class VolumeBasedCalculator:
    """
    Universal volume-based radius calculator with correction functions.
    """
    
    def __init__(self, correction_functions: Dict):
        """
        Initialize with correction functions.
        
        Args:
            correction_functions: Dictionary of correction factors
        """
        self.correction_functions = correction_functions
    
    def calculate_raw_radius(self, volume: float, z: int) -> float:
        """
        Calculate raw volume-based radius.
        
        Formula: r = (3V_atomic / 4π)^(1/3)
        where V_atomic = V_unit_cell / Z
        
        Args:
            volume: Unit cell volume in ų
            z: Number of formula units (atoms) per unit cell
            
        Returns:
            Raw volume-based radius in Angstroms
        """
        if volume <= 0 or z <= 0:
            raise ValidationError(f"Invalid volume ({volume}) or Z ({z})")
        
        atomic_volume = volume / z
        radius = (3 * atomic_volume / (4 * np.pi))**(1/3)
        
        return radius
    
    def apply_corrections(self, raw_radius: float, crystal_system: str, 
                         element_category: str = "", coordination: Optional[int] = None) -> float:
        """
        Apply correction factors to volume-based radius.
        
        Args:
            raw_radius: Raw volume-based radius
            crystal_system: Crystal structure type
            element_category: Element classification
            coordination: Coordination number
            
        Returns:
            Corrected radius in Angstroms
        """
        from config import METALLIC_ELEMENT_CATEGORIES
        from utils import clean_crystal_system_name
        
        # Clean crystal system name
        clean_system = clean_crystal_system_name(crystal_system)
        
        # Try crystal system correction (highest priority)
        crystal_corrections = self.correction_functions.get('metallic_crystal_system', {})
        if clean_system in crystal_corrections:
            corrected = raw_radius * crystal_corrections[clean_system]
            validate_radius(corrected)
            return corrected
        
        # Try element category correction
        if element_category in METALLIC_ELEMENT_CATEGORIES:
            category_corrections = self.correction_functions.get('element_category', {})
            if element_category in category_corrections:
                corrected = raw_radius * category_corrections[element_category]
                validate_radius(corrected)
                return corrected
        
        # Try coordination number correction
        if coordination:
            coord_corrections = self.correction_functions.get('metallic_coordination', {})
            coord_key = str(float(coordination))
            if coord_key in coord_corrections:
                corrected = raw_radius * coord_corrections[coord_key]
                validate_radius(corrected)
                return corrected
        
        # Use linear regression for metallic elements
        linear_params = self.correction_functions.get('metallic_linear', {})
        if linear_params and element_category in METALLIC_ELEMENT_CATEGORIES:
            slope = linear_params.get('slope', 0.882)
            intercept = linear_params.get('intercept', 0.020)
            corrected = slope * raw_radius + intercept
            validate_radius(corrected)
            return corrected
        
        # Default metallic correction
        corrected = raw_radius * 0.907
        validate_radius(corrected)
        return corrected
    
    def calculate_corrected_radius(self, volume: float, z: int, crystal_system: str,
                                 element_category: str = "", coordination: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate both raw and corrected volume-based radii.
        
        Args:
            volume: Unit cell volume
            z: Formula units per cell
            crystal_system: Crystal structure
            element_category: Element type
            coordination: Coordination number
            
        Returns:
            Tuple of (raw_radius, corrected_radius)
        """
        raw_radius = self.calculate_raw_radius(volume, z)
        corrected_radius = self.apply_corrections(raw_radius, crystal_system, element_category, coordination)
        
        return raw_radius, corrected_radius

class RadiusCalculatorFactory:
    """
    Factory class for selecting appropriate calculation methods.
    """
    
    # Mapping of crystal systems to calculation methods
    METHOD_MAPPING = {
        'fcc': RadiusCalculationMethods.face_centered_cubic,
        'hcp': RadiusCalculationMethods.hexagonal_close_packed,
        'bcc': RadiusCalculationMethods.body_centered_cubic,
        'α-La': RadiusCalculationMethods.alpha_lanthanum,
        'diamond': RadiusCalculationMethods.diamond_cubic,
        'α-Pa': RadiusCalculationMethods.tetragonal_protactinium,
        'α-Np': RadiusCalculationMethods.orthorhombic_neptunium,
        'U': RadiusCalculationMethods.uranium_structure,
        'α-Mn': RadiusCalculationMethods.alpha_manganese,
        'β-Sn': RadiusCalculationMethods.beta_tin,
        'α-As': RadiusCalculationMethods.arsenic_structure,
        'γ-Se': RadiusCalculationMethods.selenium_chains,
        'molecular': RadiusCalculationMethods.molecular_crystal,
        'cI4': RadiusCalculationMethods.molecular_crystal,
        'tetragonal': RadiusCalculationMethods.tetragonal_protactinium,
        'orthorhombic': RadiusCalculationMethods.orthorhombic_neptunium,
        'cubic': RadiusCalculationMethods.generic_structure,
    }
    
    @classmethod
    def get_calculation_method(cls, crystal_system: str):
        """
        Get appropriate calculation method for crystal system.
        
        Args:
            crystal_system: Crystal structure identifier
            
        Returns:
            Calculation method function
        """
        from utils import clean_crystal_system_name
        
        clean_system = clean_crystal_system_name(crystal_system)
        
        # Try exact match first
        if clean_system in cls.METHOD_MAPPING:
            return cls.METHOD_MAPPING[clean_system]
        
        # Try partial matches
        for system_key, method in cls.METHOD_MAPPING.items():
            if system_key in clean_system or clean_system in system_key:
                return method
        
        # Default to generic method
        return RadiusCalculationMethods.generic_structure
    
    @classmethod
    def calculate_traditional_radius(cls, crystal_system: str, a: float, 
                                   b: Optional[float] = None, c: Optional[float] = None, 
                                   **kwargs) -> Optional[float]:
        """
        Calculate radius using traditional structure-specific method.
        
        Args:
            crystal_system: Crystal structure type
            a, b, c: Lattice parameters
            **kwargs: Additional parameters
            
        Returns:
            Calculated radius or None if calculation fails
        """
        try:
            method = cls.get_calculation_method(crystal_system)
            radius = method(a=a, b=b, c=c, **kwargs)
            validate_radius(radius)
            return radius
        except (ValidationError, ValueError, TypeError) as e:
            print(f"⚠️  Traditional calculation failed for {crystal_system}: {e}")
            return None