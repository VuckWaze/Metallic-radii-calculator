# Enhanced Metallic Radii Calculator

## Overview

A comprehensive tool for calculating metallic radii across the entire periodic table using both traditional structure-specific methods and universal volume-based approaches with outlier-excluded correction functions.

## Version 2.0 - Refactored Architecture

This refactored version features:
- **Modular Design**: Separated concerns into focused modules
- **Professional Structure**: Clean, maintainable code architecture  
- **Comprehensive Error Handling**: Robust validation and error reporting
- **Enhanced Configuration**: Centralized settings and constants
- **Improved Documentation**: Clear docstrings and comments
- **Production Ready**: Optimized for research and industrial use

## Features

### Dual-Method Calculations
- **Traditional Structure-Specific**: 25+ crystal structure types supported
- **Universal Volume-Based**: r = (3V_atomic / 4π)^(1/3) with corrections
- **Corrected Volume Method**: Outlier-excluded correction functions (R² = 0.960)

### Crystal Structure Support
- Close-packed structures (fcc, hcp, α-La)
- Body-centered cubic (bcc)
- Complex actinide structures (α-Pa, α-Np, U)
- Covalent structures (diamond, β-Sn)
- Low-coordination structures (α-As, γ-Se)
- Molecular crystals and van der Waals bonded systems

### Advanced Features
- Automatic averaging of multiple measurements per element
- Comprehensive periodic table coverage (83+ elements)
- Outlier detection and exclusion from corrections
- Professional error handling and validation
- Detailed progress reporting and statistics
- **Data Export**: Results saved as CSV files for further analysis

## Crystal Structures Analyzed

The dataset includes rare earth elements with different crystal structures:

- **Hexagonal Close-Packed (HCP)**: Dy, Er, Gd, Ho, Lu, Sc, Tb, Tm, Y
- **Face-Centered Cubic (FCC)**: Ce, Pt, Yb
- **Body-Centered Cubic (BCC)**: Eu
- **α-Lanthanum structure**: La, Nd, Sm

## Installation

1. Ensure Python 3.8+ is installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script:
```bash
python metallic_radii_calculator.py
```

## Output Files

The script generates several output files in the `output/` directory:

### Data Files
- `metallic_radii_results.csv`: Complete dataset with all calculated properties
- `metallic_radii_summary.csv`: Summary table with key properties

### Visualization Files
- `metallic_radii_by_element.png`: Bar chart of metallic radii by element
- `radius_vs_atomic_volume.png`: Scatter plot showing radius-volume correlation
- `crystal_system_distribution.png`: Distribution of crystal systems
- `lattice_parameters_analysis.png`: Comprehensive lattice parameter analysis

## Methodology

### Metallic Radius Calculation

The metallic radius is calculated as half the nearest-neighbor distance in the crystal lattice:

1. **FCC Structure**: 
   - Nearest neighbors at distance a/√2
   - Metallic radius = a/(2√2)

2. **HCP Structure**: 
   - Nearest neighbors at distance a
   - Metallic radius = a/2

3. **BCC Structure**: 
   - Nearest neighbors at distance a√3/2
   - Metallic radius = a√3/4

4. **α-La Structure**: 
   - Similar to HCP
   - Metallic radius = a/2

### Data Validation

The calculator includes validation for:
- Proper CIF file format
- Valid crystallographic parameters
- Reasonable calculated values

## Results Interpretation

### Expected Trends
- **Lanthanide Contraction**: Decreasing ionic radii across the lanthanide series
- **Crystal Structure Effects**: Different structures may show systematic variations
- **Coordination Effects**: Higher coordination numbers typically correlate with larger effective radii

### Key Properties Analyzed
- Metallic radius (primary calculation)
- Atomic volume (unit cell volume / Z)
- Lattice parameters (a, b, c)
- Crystal system distribution
- Density correlations

## Technical Details

### Classes and Methods

- `CIFParser`: Parses CIF files and extracts crystallographic data
- `MetallicRadiiCalculator`: Calculates metallic radii and processes multiple files
- `MetallicRadiiAnalyzer`: Creates visualizations and statistical analysis

### Data Structure

Each element's data includes:
- Element symbol and crystal system
- Lattice parameters (a, b, c, α, β, γ)
- Unit cell volume and formula units (Z)
- Calculated metallic radius and atomic volume
- Coordination number and space group

## Limitations

- Assumes ideal crystal structures
- Temperature effects not considered
- Pressure effects not accounted for
- Some structures may require more sophisticated nearest-neighbor calculations

## References

- Crystallographic data from ICSD (Inorganic Crystal Structure Database)
- Metallic radius definitions based on standard crystallographic principles
- Crystal structure analysis following International Tables for Crystallography

## License

This project is provided for educational and research purposes.