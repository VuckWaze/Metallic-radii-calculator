# Metallic Radii Calculator for Rare Earth Elements

This project calculates metallic radii from crystallographic information files (CIF) of rare earth elements and provides comprehensive analysis through tables and visualizations.

## Overview

The metallic radius is a fundamental property that represents half the distance between the centers of two adjacent atoms in a metallic crystal. This calculator extracts crystallographic data from CIF files and computes metallic radii based on the crystal structure and nearest-neighbor distances.

## Features

- **CIF File Parsing**: Automatically extracts crystallographic data from CIF files
- **Metallic Radius Calculation**: Computes metallic radii based on crystal structure:
  - FCC (Face-Centered Cubic): r = a/(2√2)
  - HCP (Hexagonal Close-Packed): r = a/2
  - BCC (Body-Centered Cubic): r = a√3/4
  - α-La structure: r = a/2
- **Comprehensive Analysis**: 
  - Summary tables with key properties
  - Multiple visualization plots
  - Statistical analysis
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