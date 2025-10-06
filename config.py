"""
Configuration module for Enhanced Metallic Radii Calculator.

Contains constants, default values, and configuration settings used
throughout the calculator.
"""

from typing import Dict, Set

# Crystal structure definitions with coordination numbers and calculation methods
CRYSTAL_STRUCTURE_DATABASE = {
    # Close-packed structures (highest coordination)
    'fcc': {'coordination': 12, 'method': 'face_centered_cubic', 'description': 'Face-centered cubic'},
    'hcp': {'coordination': 12, 'method': 'hexagonal_close_packed', 'description': 'Hexagonal close-packed'},
    
    # Body-centered structures
    'bcc': {'coordination': 8, 'method': 'body_centered_cubic', 'description': 'Body-centered cubic'},
    
    # Lanthanide-type structures
    'α-La': {'coordination': 12, 'method': 'alpha_lanthanum', 'description': 'Double hexagonal close-packed'},
    
    # Covalent and semiconductor structures
    'diamond': {'coordination': 4, 'method': 'diamond_cubic', 'description': 'Diamond cubic structure'},
    'β-Sn': {'coordination': 6, 'method': 'beta_tin', 'description': 'White tin structure'},
    
    # Complex actinide structures
    'α-Pa': {'coordination': 8, 'method': 'tetragonal_protactinium', 'description': 'Tetragonal protactinium'},
    'α-Np': {'coordination': 8, 'method': 'orthorhombic_neptunium', 'description': 'Orthorhombic neptunium'},
    'U': {'coordination': 8, 'method': 'uranium_structure', 'description': 'Orthorhombic uranium'},
    
    # Complex transition metal structures
    'α-Mn': {'coordination': 8, 'method': 'alpha_manganese', 'description': 'Complex cubic manganese'},
    
    # Low-coordination structures
    'α-As': {'coordination': 3, 'method': 'arsenic_structure', 'description': 'Rhombohedral arsenic'},
    'γ-Se': {'coordination': 2, 'method': 'selenium_chains', 'description': 'Monoclinic selenium'},
    
    # Molecular crystals
    'cI4': {'coordination': 2, 'method': 'molecular_crystal', 'description': 'Simple molecular crystal'},
    'molecular': {'coordination': 2, 'method': 'molecular_crystal', 'description': 'General molecular crystal'},
}

# Default coordination numbers for structures not in main database
DEFAULT_COORDINATION_NUMBERS = {
    'fcc': 12, 'hcp': 12, 'bcc': 8, 'diamond': 4,
    'α-La': 12, 'α-As': 3, 'cI4': 2, 'molecular': 2,
    'tetragonal': 8, 'orthorhombic': 8, 'cubic': 8,
    'hexagonal': 6, 'monoclinic': 4, 'triclinic': 4
}

# Element categories that follow metallic bonding principles
METALLIC_ELEMENT_CATEGORIES: Set[str] = {
    'Transition Metals',
    'Alkali Metals', 
    'Alkaline Earth Metals',
    'Lanthanides',
    'Actinides',
    'Post-transition Metals',
    'Rare Earth Metals'
}

# Non-metallic element categories (excluded from corrections)
NON_METALLIC_CATEGORIES: Set[str] = {
    'Noble Gases',
    'Halogens', 
    'Nonmetals',
    'Metalloids',
    'Molecular Crystals',
    'Covalent Networks'
}

# Elements known to have unusual bonding or structural anomalies
OUTLIER_ELEMENTS: Set[str] = {
    # Group 14 covalent elements
    'C', 'Si', 'Ge',
    
    # Molecular crystals
    'O2', 'N2', 'F2', 'Cl2', 'Br2', 'I2',
    'H2', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',
    
    # Complex covalent structures
    'P', 'As', 'Sb', 'Bi',
    'S', 'Se', 'Te',
    
    # Van der Waals bonded
    'graphite', 'layered_structures'
}

# Default correction factors (fallback if file not found)
DEFAULT_CORRECTION_FACTORS = {
    "metallic_crystal_system": {
        "fcc": 0.905,
        "hcp": 0.909,
        "bcc": 0.867,
        "α-La": 0.908,
        "α-Pa": 0.912,
        "α-Mn": 0.885
    },
    "metallic_coordination": {
        "12.0": 0.907,  # Close-packed
        "8.0": 0.867,   # BCC-type
        "6.0": 0.842,   # Octahedral
        "4.0": 0.795    # Tetrahedral
    },
    "metallic_linear": {
        "slope": 0.882,
        "intercept": 0.020,
        "r2": 0.960
    },
    "element_category": {
        "Transition Metals": 0.895,
        "Alkali Metals": 0.920,
        "Alkaline Earth Metals": 0.910,
        "Lanthanides": 0.908,
        "Actinides": 0.905,
        "Post-transition Metals": 0.885
    }
}

# File and directory configuration
DEFAULT_PATHS = {
    'correction_functions': 'improved_correction_functions.json',
    'elements_directory': 'Elements',
    'output_directory': 'output',
    'backup_directory': 'backup'
}

# Output CSV column names and order
OUTPUT_COLUMNS = [
    'Element',
    'Crystal System', 
    'Space Group',
    'Space Group Number',
    'a (Å)', 'b (Å)', 'c (Å)',
    'alpha (°)', 'beta (°)', 'gamma (°)',
    'Volume (Å³)',
    'Z',
    'Density (g/cm³)',
    'Traditional Radius (Å)',
    'Volume Radius (Å)', 
    'Corrected Volume Radius (Å)',
    'Primary Radius (Å)',
    'Atomic Volume (Å³)',
    'Coordination Number',
    'Filename'
]

# Numeric fields for averaging multiple measurements
NUMERIC_FIELDS = [
    'a (Å)', 'b (Å)', 'c (Å)', 
    'alpha (°)', 'beta (°)', 'gamma (°)',
    'Volume (Å³)', 'Z', 'Density (g/cm³)',
    'Traditional Radius (Å)', 'Volume Radius (Å)', 
    'Corrected Volume Radius (Å)', 'Primary Radius (Å)',
    'Atomic Volume (Å³)', 'Coordination Number'
]

# Tolerance values for calculations
CALCULATION_TOLERANCES = {
    'minimum_volume': 1.0,        # Minimum unit cell volume (Ų)
    'minimum_z': 1,               # Minimum formula units
    'maximum_z': 100,             # Maximum reasonable formula units
    'minimum_radius': 0.5,        # Minimum reasonable metallic radius (Å)
    'maximum_radius': 5.0,        # Maximum reasonable metallic radius (Å)
    'angle_tolerance': 0.1,       # Tolerance for angle comparisons (°)
    'length_tolerance': 0.001     # Tolerance for length comparisons (Å)
}

# Display formatting
DISPLAY_CONFIG = {
    'radius_precision': 3,        # Decimal places for radii
    'volume_precision': 2,        # Decimal places for volumes
    'angle_precision': 1,         # Decimal places for angles
    'density_precision': 3,       # Decimal places for density
    'progress_interval': 10       # Show progress every N files
}

# Crystal system aliases and variations
CRYSTAL_SYSTEM_ALIASES = {
    'face-centered cubic': 'fcc',
    'f.c.c.': 'fcc',
    'cubic F': 'fcc',
    'Fm-3m': 'fcc',
    
    'body-centered cubic': 'bcc',
    'b.c.c.': 'bcc', 
    'cubic I': 'bcc',
    'Im-3m': 'bcc',
    
    'hexagonal close-packed': 'hcp',
    'h.c.p.': 'hcp',
    'P63/mmc': 'hcp',
    
    'double hexagonal close-packed': 'α-La',
    'd.h.c.p.': 'α-La'
}

# Error messages
ERROR_MESSAGES = {
    'file_not_found': "CIF file not found: {path}",
    'parse_error': "Failed to parse CIF file: {error}",
    'calculation_error': "Radius calculation failed: {error}",
    'invalid_structure': "Unknown crystal structure: {structure}",
    'missing_parameters': "Missing required crystallographic parameters",
    'invalid_radius': "Calculated radius outside reasonable range: {radius:.3f} Å"
}

# Success messages
SUCCESS_MESSAGES = {
    'calculation_complete': "✅ Radius calculation completed successfully",
    'file_processed': "📄 Processed file: {filename}",
    'results_saved': "💾 Results saved to: {path}",
    'analysis_complete': "🏆 Analysis complete! {count} elements processed"
}