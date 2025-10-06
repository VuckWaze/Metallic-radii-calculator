#!/usr/bin/env python3
"""
Enhanced Metallic Radii Calculator - Main Application

A comprehensive tool for calculating metallic radii across the entire periodic table
using both traditional structure-specific methods and universal volume-based approaches
with outlier-excluded correction functions.

This refactored version features:
- Modular architecture with separated concerns
- Comprehensive error handling and validation
- Professional code structure and documentation
- Enhanced configuration management
- Improved maintainability and extensibility

Author: Developed with GitHub Copilot
Version: 2.0 (Refactored Production)
Date: October 2025
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from config import (
    DEFAULT_PATHS, 
    DEFAULT_CORRECTION_FACTORS,
    OUTPUT_COLUMNS,
    NUMERIC_FIELDS,
    SUCCESS_MESSAGES,
    ERROR_MESSAGES,
    DISPLAY_CONFIG
)
from utils import (
    load_json_file,
    create_directory_if_not_exists,
    get_file_list,
    progress_indicator,
    ValidationError
)
from calculations import VolumeBasedCalculator, RadiusCalculatorFactory
from visualizations import PeriodicTableVisualizer

class CIFParser:
    """
    Enhanced parser for Crystallographic Information Files (CIF).
    
    Extracts crystallographic data including lattice parameters, space group,
    density, and atomic positions from CIF files.
    """
    
    def __init__(self, cif_path: str):
        """
        Initialize CIF parser.
        
        Args:
            cif_path: Path to the CIF file
            
        Raises:
            ValidationError: If file cannot be read or parsed
        """
        self.cif_path = Path(cif_path)
        self.data = {}
        
        if not self.cif_path.exists():
            raise ValidationError(ERROR_MESSAGES['file_not_found'].format(path=cif_path))
            
        self._parse_cif()
    
    def _parse_cif(self) -> None:
        """Parse CIF file and extract crystallographic data."""
        try:
            with open(self.cif_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValidationError(ERROR_MESSAGES['parse_error'].format(error=str(e)))
        
        self.data['filename'] = self.cif_path.name
        
        # Extract data using helper methods
        from utils import (
            extract_element_from_filename, 
            CrystalSystemDetector,
            calculate_unit_cell_volume,
            safe_float_conversion,
            safe_int_conversion
        )
        import re
        
        # Element identification
        self.data['element'] = extract_element_from_filename(self.data['filename'])
        
        # Crystal system from filename
        self.data['crystal system'] = CrystalSystemDetector.from_filename(self.data['filename'])
        
        # Lattice parameters
        for param in ['a', 'b', 'c']:
            pattern = rf'_cell_length_{param}\s+([\d.]+)'
            match = re.search(pattern, content)
            self.data[param] = safe_float_conversion(match.group(1) if match else None)
        
        # Lattice angles
        for angle in ['alpha', 'beta', 'gamma']:
            pattern = rf'_cell_angle_{angle}\s+([\d.]+)'
            match = re.search(pattern, content)
            self.data[angle] = safe_float_conversion(match.group(1) if match else None, 90.0)
        
        # Space group information
        self._extract_space_group(content)
        
        # Unit cell volume
        volume_match = re.search(r'_cell_volume\s+([\d.]+)', content)
        self.data['volume'] = safe_float_conversion(volume_match.group(1) if volume_match else None)
        
        # Calculate volume if not provided
        if not self.data['volume'] and all(self.data.get(p) for p in ['a', 'b', 'c']):
            self.data['volume'] = calculate_unit_cell_volume(
                self.data['a'], self.data['b'] or self.data['a'], self.data['c'] or self.data['a'],
                self.data['alpha'], self.data['beta'], self.data['gamma']
            )
        
        # Formula units (Z)
        z_match = re.search(r'_cell_formula_units_Z\s+(\d+)', content)
        self.data['z'] = safe_int_conversion(z_match.group(1) if z_match else None)
        
        # Density
        density_patterns = [
            r'_exptl_crystal_density_diffrn\s+([\d.]+)',
            r'_exptl_crystal_density_meas\s+([\d.]+)',
        ]
        for pattern in density_patterns:
            match = re.search(pattern, content)
            if match:
                self.data['density'] = safe_float_conversion(match.group(1))
                break
        else:
            self.data['density'] = None
    
    def _extract_space_group(self, content: str) -> None:
        """Extract space group information."""
        import re
        from utils import safe_int_conversion
        
        # Space group symbol
        sg_patterns = [
            r"_space_group_name_H-M_alt\s+'([^']+)'",
            r'_space_group_name_H-M_alt\s+([^\s]+)',
            r"_symmetry_space_group_name_H-M\s+'([^']+)'",
        ]
        
        for pattern in sg_patterns:
            match = re.search(pattern, content)
            if match:
                self.data['space group'] = match.group(1).strip()
                break
        else:
            self.data['space group'] = 'Unknown'
        
        # Space group number
        sg_number_match = re.search(r'_space_group_IT_number\s+(\d+)', content)
        self.data['space group number'] = safe_int_conversion(sg_number_match.group(1) if sg_number_match else None)

class MetallicRadiusCalculator:
    """
    Enhanced calculator for metallic radii with dual-method approach.
    
    Supports both traditional structure-specific calculations and universal
    volume-based calculations with outlier-excluded correction functions.
    """
    
    def __init__(self):
        """Initialize calculator with correction functions."""
        self.correction_functions = self._load_correction_functions()
        self.volume_calculator = VolumeBasedCalculator(self.correction_functions)
        
        print("🔬 Enhanced Metallic Radii Calculator initialized!")
        print(f"   ✅ Correction functions loaded")
        print(f"   📚 Supporting comprehensive crystal structure database")
    
    def calculate_all_methods(self, cif_data: Dict) -> Dict[str, Optional[float]]:
        """
        Calculate metallic radius using all available methods.
        
        Args:
            cif_data: Dictionary containing crystallographic data
            
        Returns:
            Dictionary with all calculation results
        """
        results = {}
        
        # Traditional structure-specific calculation
        results['traditional'] = self._calculate_traditional_radius(cif_data)
        
        # Volume-based calculations
        raw_volume, corrected_volume = self._calculate_volume_radii(cif_data)
        results['volume_raw'] = raw_volume
        results['volume_corrected'] = corrected_volume
        
        return results
    
    def _calculate_traditional_radius(self, cif_data: Dict) -> Optional[float]:
        """Calculate radius using traditional structure-specific methods."""
        crystal_system = cif_data.get('crystal system', '')
        a = cif_data.get('a')
        b = cif_data.get('b')
        c = cif_data.get('c')
        
        if not a:
            return None
        
        return RadiusCalculatorFactory.calculate_traditional_radius(
            crystal_system=crystal_system,
            a=a, b=b, c=c
        )
    
    def _calculate_volume_radii(self, cif_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate both raw and corrected volume-based radii.
        
        Returns:
            Tuple of (raw_volume_radius, corrected_volume_radius)
        """
        volume = cif_data.get('volume')
        z = cif_data.get('z')
        
        if not volume or not z or z <= 0:
            return None, None
        
        try:
            return self.volume_calculator.calculate_corrected_radius(
                volume=volume,
                z=z,
                crystal_system=cif_data.get('crystal system', ''),
                element_category=cif_data.get('element_category', ''),
                coordination=self._get_coordination_number(cif_data.get('crystal system', ''))
            )
        except ValidationError:
            return None, None
    
    def _get_coordination_number(self, crystal_system: str) -> Optional[int]:
        """Get coordination number for crystal system."""
        from config import CRYSTAL_STRUCTURE_DATABASE, DEFAULT_COORDINATION_NUMBERS
        from utils import clean_crystal_system_name
        
        clean_system = clean_crystal_system_name(crystal_system)
        
        if clean_system in CRYSTAL_STRUCTURE_DATABASE:
            return CRYSTAL_STRUCTURE_DATABASE[clean_system]['coordination']
        
        # Fallback to defaults
        for system, coord in DEFAULT_COORDINATION_NUMBERS.items():
            if system in clean_system:
                return coord
        
        return None
    
    def _load_correction_functions(self) -> Dict:
        """Load correction functions from file."""
        correction_file = DEFAULT_PATHS['correction_functions']
        corrections = load_json_file(correction_file)
        
        if not corrections:
            print("⚠️  Using default correction functions")
            corrections = DEFAULT_CORRECTION_FACTORS
        else:
            print(f"📁 Loaded correction functions from {correction_file}")
        
        return corrections

class PeriodicTableProcessor:
    """
    Process multiple CIF files and generate comprehensive periodic table analysis.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.calculator = MetallicRadiusCalculator()
        self.results = []
        
    def process_directory(self, cif_directory: str) -> None:
        """
        Process all CIF files in a directory.
        
        Args:
            cif_directory: Path to directory containing CIF files
        """
        cif_files = get_file_list(cif_directory, "*.cif")
        
        if not cif_files:
            raise ValueError(f"No CIF files found in {cif_directory}")
        
        print(f"📁 Processing {len(cif_files)} CIF files...")
        
        for i, cif_file in enumerate(cif_files, 1):
            try:
                result = self._process_single_file(cif_file)
                if result:
                    self.results.append(result)
                
                progress_indicator(i, len(cif_files), DISPLAY_CONFIG['progress_interval'])
                    
            except Exception as e:
                print(f"   ❌ Error processing {cif_file.name}: {e}")
        
        self._process_multiple_measurements()
        print(SUCCESS_MESSAGES['analysis_complete'].format(count=len(self.results)))
    
    def _process_single_file(self, cif_file: Path) -> Optional[Dict]:
        """Process a single CIF file."""
        try:
            parser = CIFParser(cif_file)
            calculations = self.calculator.calculate_all_methods(parser.data)
            
            # Get primary radius (corrected volume preferred)
            primary_radius = calculations['volume_corrected']
            if not primary_radius:
                primary_radius = calculations['traditional']
            
            if not primary_radius:
                return None
            
            # Calculate atomic volume
            atomic_volume = None
            if parser.data.get('volume') and parser.data.get('z'):
                atomic_volume = parser.data['volume'] / parser.data['z']
            
            # Get coordination number
            coordination = self.calculator._get_coordination_number(
                parser.data.get('crystal system', '')
            )
            
            return {
                'Element': parser.data.get('element'),
                'Crystal System': parser.data.get('crystal system'),
                'Space Group': parser.data.get('space group'),
                'Space Group Number': parser.data.get('space group number'),
                'a (Å)': parser.data.get('a'),
                'b (Å)': parser.data.get('b'),
                'c (Å)': parser.data.get('c'),
                'alpha (°)': parser.data.get('alpha'),
                'beta (°)': parser.data.get('beta'), 
                'gamma (°)': parser.data.get('gamma'),
                'Volume (Å³)': parser.data.get('volume'),
                'Z': parser.data.get('z'),
                'Density (g/cm³)': parser.data.get('density'),
                'Traditional Radius (Å)': calculations['traditional'],
                'Volume Radius (Å)': calculations['volume_raw'],
                'Corrected Volume Radius (Å)': calculations['volume_corrected'],
                'Primary Radius (Å)': primary_radius,
                'Atomic Volume (Å³)': atomic_volume,
                'Coordination Number': coordination,
                'Filename': parser.data.get('filename'),
            }
            
        except Exception as e:
            print(f"   ⚠️  Error processing {cif_file.name}: {e}")
            return None
    
    def _process_multiple_measurements(self) -> None:
        """Process and average multiple measurements for the same element."""
        # Group by element
        element_groups = {}
        for result in self.results:
            element = result['Element']
            if element not in element_groups:
                element_groups[element] = []
            element_groups[element].append(result)
        
        # Find and process duplicates
        duplicates = {elem: data for elem, data in element_groups.items() if len(data) > 1}
        
        if duplicates:
            print(f"\n🔄 Found {len(duplicates)} elements with multiple measurements:")
            
            for element, measurements in duplicates.items():
                self._average_element_measurements(element, measurements)
    
    def _average_element_measurements(self, element: str, measurements: List[Dict]) -> None:
        """Average multiple measurements for a single element."""
        print(f"   • {element}: {len(measurements)} measurements")
        
        # Calculate averages for numeric fields
        averaged_result = measurements[0].copy()  # Start with first measurement
        
        for field in NUMERIC_FIELDS:
            values = [m[field] for m in measurements if m[field] is not None]
            if values:
                averaged_result[field] = np.mean(values)
        
        # Combine text fields
        crystal_systems = list(set(m['Crystal System'] for m in measurements))
        if len(crystal_systems) > 1:
            averaged_result['Crystal System'] = f"Multiple: {', '.join(crystal_systems)}"
        
        filenames = [m['Filename'] for m in measurements]
        averaged_result['Filename'] = f"Averaged from {len(measurements)} files: {', '.join(filenames[:3])}"
        if len(filenames) > 3:
            averaged_result['Filename'] += "..."
        
        # Remove original measurements and add averaged result
        self.results = [r for r in self.results if r['Element'] != element]
        self.results.append(averaged_result)
    
    def save_results(self, output_path: str) -> None:
        """Save results to CSV file."""
        df = pd.DataFrame(self.results)
        df = df.sort_values('Element')
        
        # Reorder columns according to configuration
        available_columns = [col for col in OUTPUT_COLUMNS if col in df.columns]
        df = df[available_columns]
        
        output_file = Path(output_path)
        create_directory_if_not_exists(output_file.parent)
        
        df.to_csv(output_file, index=False)
        print(SUCCESS_MESSAGES['results_saved'].format(path=output_file))
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        from utils import calculate_statistics
        
        primary_radii = [r for r in df['Primary Radius (Å)'] if r is not None]
        
        stats = {
            'total_elements': len(df),
            'elements_with_radii': len(primary_radii),
            'radius_statistics': calculate_statistics(primary_radii) if primary_radii else {},
            'crystal_systems': df['Crystal System'].value_counts().to_dict(),
            'coverage_percentage': (len(primary_radii) / len(df)) * 100 if len(df) > 0 else 0
        }
        
        return stats

def main():
    """Main execution function."""
    print("🌟 ENHANCED Metallic Radii Calculator - Production Version 2.0!")
    print("=" * 80)
    
    # Check for CIF directory
    elements_dir = DEFAULT_PATHS['elements_directory']
    if 'Elements' in str(Path.cwd()).lower():
        elements_dir = "."  # Use current directory if we're in a Elements folder
    
    if not Path(elements_dir).exists():
        # Try alternative locations
        for alt_dir in ['Elements', 'cif', '../Elements', '../cif']:
            if Path(alt_dir).exists():
                elements_dir = alt_dir
                break
        else:
            print(f"❌ No CIF directory found. Tried: {elements_dir}")
            print("   Please ensure CIF files are in 'Elements' or 'cif' directory")
            return 1
    
    try:
        # Initialize processor
        processor = PeriodicTableProcessor()
        
        # Process all elements
        processor.process_directory(elements_dir)
        
        if not processor.results:
            print("❌ No valid results obtained from CIF files")
            return 1
        
        # Save results
        output_dir = DEFAULT_PATHS['output_directory']
        output_file = f"{output_dir}/enhanced_metallic_radii_results.csv"
        processor.save_results(output_file)
        
        # Generate and display summary
        stats = processor.generate_summary_statistics()
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total CIF files processed: {stats['total_elements']}")
        print(f"   Elements with calculated radii: {stats['elements_with_radii']}")
        print(f"   Success rate: {stats['coverage_percentage']:.1f}%")
        
        if stats.get('radius_statistics'):
            radius_stats = stats['radius_statistics']
            print(f"   Average metallic radius: {radius_stats['mean']:.3f} Å")
            print(f"   Radius range: {radius_stats['min']:.3f} - {radius_stats['max']:.3f} Å")
            print(f"   Standard deviation: {radius_stats['std']:.3f} Å")
        
        print(f"\n🏆 ANALYSIS COMPLETE!")
        print(f"   Results saved to: {output_file}")
        
        # Generate beautiful visualizations
        try:
            print(f"\n🎨 Generating comprehensive visualizations...")
            
            # Create DataFrame for visualizations
            results_df = pd.DataFrame(processor.results)
            results_df = results_df.sort_values('Element')
            
            # Only proceed if we have data
            if not results_df.empty:
                visualizer = PeriodicTableVisualizer(results_df)
                
                # Create plots directory
                plots_dir = f"{output_dir}/plots"
                create_directory_if_not_exists(plots_dir)
                
                # Generate all visualization types
                visualizer.generate_all_visualizations(plots_dir)
                
                # Show interactive plots
                print(f"   Opening interactive plots... (Close plot windows to continue)")
                visualizer.show_all_plots()
                
                print(f"   📈 Visualization complete! Plots saved to: {plots_dir}")
            else:
                print(f"   ⚠️  No data available for visualization")
            
        except Exception as plot_error:
            print(f"⚠️  Visualization error (continuing without plots): {plot_error}")
        
        print(f"   Ready for crystallographic research! 🔬")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())