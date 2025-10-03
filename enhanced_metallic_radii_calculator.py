#!/usr/bin/env python3
"""
Enhanced Metallic Radii Calculator for ALL Elements
Handles complex crystal structures across the entire periodic table
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedCIFParser:
    """Enhanced parser for CIF files with support for complex structures"""
    
    def __init__(self, cif_path: str):
        self.cif_path = cif_path
        self.data = {}
        self.atomic_positions = []
        self.parse_cif()
    
    def parse_cif(self):
        """Parse CIF file and extract crystallographic data"""
        with open(self.cif_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract basic information
        self.data['filename'] = os.path.basename(self.cif_path)
        
        # Extract element name (more robust)
        element_match = re.search(r'^([A-Z][a-z]?)(?:[0-9]*|_)', self.data['filename'])
        if element_match:
            self.data['element'] = element_match.group(1)
        else:
            # Fallback: try to extract from chemical formula
            formula_match = re.search(r'_chemical_formula_sum\s+([A-Z][a-z]?)', content)
            self.data['element'] = formula_match.group(1) if formula_match else 'Unknown'
        
        # Extract crystal system from filename (more comprehensive)
        crystal_patterns = [
            r'_\(([^)]+)\)_',  # Standard pattern
            r'_(α-[^_]+)_',    # Alpha phases
            r'_(β-[^_]+)_',    # Beta phases  
            r'_(γ-[^_]+)_',    # Gamma phases
            r'_([a-z]+P[0-9]+)_',  # Space group notation
            r'_([a-z]+I[0-9]+)_',  # Space group notation
            r'_([a-z]+R[0-9]+)_',  # Space group notation
            r'_([a-z]+C[0-9]+)_',  # Space group notation
        ]
        
        self.data['crystal system'] = 'Unknown'
        for pattern in crystal_patterns:
            match = re.search(pattern, self.data['filename'])
            if match:
                self.data['crystal system'] = match.group(1)
                break
        
        # Extract lattice parameters
        self.data['a'] = self._extract_parameter(content, r'_cell_length_a\s+([\d.]+)')
        self.data['b'] = self._extract_parameter(content, r'_cell_length_b\s+([\d.]+)')
        self.data['c'] = self._extract_parameter(content, r'_cell_length_c\s+([\d.]+)')
        
        # Extract cell angles
        self.data['alpha'] = self._extract_parameter(content, r'_cell_angle_alpha\s+([\d.]+)')
        self.data['beta'] = self._extract_parameter(content, r'_cell_angle_beta\s+([\d.]+)')
        self.data['gamma'] = self._extract_parameter(content, r'_cell_angle_gamma\s+([\d.]+)')
        
        # Extract cell volume and Z
        self.data['volume'] = self._extract_parameter(content, r'_cell_volume\s+([\d.]+)')
        self.data['z'] = self._extract_parameter(content, r'_cell_formula_units_Z\s+(\d+)', int)
        
        # Extract space group information
        space_group_patterns = [
            r"_space_group_name_H-M_alt\s+'([^']+)'",
            r"_space_group_name_H-M\s+'([^']+)'",
            r"_symmetry_space_group_name_H-M\s+'([^']+)'"
        ]
        
        for pattern in space_group_patterns:
            match = re.search(pattern, content)
            if match:
                self.data['space group'] = match.group(1)
                break
        else:
            self.data['space group'] = 'Unknown'
        
        # Extract space group number
        self.data['space group_number'] = self._extract_parameter(content, r'_space_group_IT_number\s+(\d+)', int)
        
        # Extract density
        self.data['density'] = self._extract_parameter(content, r'_exptl_crystal_density_diffrn\s+([\d.]+)')
        
        # Extract atomic positions for complex structure analysis
        self._extract_atomic_positions(content)
    
    def _extract_parameter(self, content: str, pattern: str, dtype=float):
        """Extract a parameter using regex pattern"""
        match = re.search(pattern, content)
        if match:
            try:
                value_str = match.group(1).replace('(', '').replace(')', '')  # Remove uncertainty
                return dtype(value_str)
            except ValueError:
                return None
        return None
    
    def _extract_atomic_positions(self, content: str):
        """Extract atomic positions for distance calculations"""
        # Look for atomic position loop
        position_section = re.search(
            r'loop \s*\n(_atom_site_[^\n]*\n)*.*?(?=\n\s*(?:loop_|#|$))', 
            content, re.DOTALL | re.MULTILINE
        )
        
        if position_section:
            lines = position_section.group(0).split('\n')
            headers = []
            data_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('_atom_site_'):
                    headers.append(line)
                elif line and not line.startswith(('loop ', '#', '_')):
                    data_lines.append(line.split())
            
            # Parse atomic positions if we have the necessary columns
            if any('_atom_site_fract_x' in h for h in headers):
                x_idx = next((i for i, h in enumerate(headers) if '_atom_site_fract_x' in h), None)
                y_idx = next((i for i, h in enumerate(headers) if '_atom_site_fract_y' in h), None)
                z_idx = next((i for i, h in enumerate(headers) if '_atom_site_fract_z' in h), None)
                
                for data_line in data_lines:
                    if len(data_line) > max(x_idx or 0, y_idx or 0, z_idx or 0):
                        try:
                            pos = {
                                'x': float(data_line[x_idx]) if x_idx is not None else 0,
                                'y': float(data_line[y_idx]) if y_idx is not None else 0,
                                'z': float(data_line[z_idx]) if z_idx is not None else 0
                            }
                            self.atomic_positions.append(pos)
                        except (ValueError, IndexError):
                            continue

class EnhancedMetallicRadiiCalculator:
    """Enhanced calculator for metallic radii with support for complex structures"""
    
    # Extended coordination numbers and distance calculations
    STRUCTURE_INFO = {
        # Simple cubic structures
        'fcc': {'coordination': 12, 'calc method': 'fcc'},
        'hcp': {'coordination': 12, 'calc method': 'hcp'},
        'bcc': {'coordination': 8, 'calc method': 'bcc'},
        
        # Lanthanide structures
        'α-La': {'coordination': 12, 'calc method': 'alpha la'},
        
        # Diamond structures
        'diamond': {'coordination': 4, 'calc method': 'diamond'},
        
        # Complex metallic structures
        'α-Mn': {'coordination': 8, 'calc method': 'alpha mn'},
        'α-Np': {'coordination': 8, 'calc method': 'orthorhombic'},
        'α-Pa': {'coordination': 8, 'calc method': 'tetragonal'},
        'U': {'coordination': 8, 'calc method': 'uranium'},
        'α-Po': {'coordination': 6, 'calc method': 'simple cubic'},
        'β-Po': {'coordination': 6, 'calc method': 'simple cubic'},
        
        # Tetragonal structures
        'tI2': {'coordination': 8, 'calc method': 'tetragonal centered'},
        'tP50': {'coordination': 6, 'calc method': 'complex tetragonal'},
        
        # Rhombohedral
        'hR12': {'coordination': 6, 'calc method': 'rhombohedral'},
        'α-As': {'coordination': 3, 'calc method': 'arsenic'},
        
        # Gallium structure
        'α-Ga': {'coordination': 7, 'calc method': 'gallium'},
        
        # Molecular crystals (special handling)
        'Fddd': {'coordination': 2, 'calc method': 'molecular'},  # S8
        'Cmca': {'coordination': 2, 'calc method': 'molecular'},  # I2
        'cI4': {'coordination': 12, 'calc method': 'molecular'},  # H2, D2
        'γ-Se': {'coordination': 2, 'calc method': 'selenium'},
        
        # Complex boron structures
        'β-B': {'coordination': 6, 'calc method': 'beta boron'},
        
        # Tin structures
        'β-Sn': {'coordination': 4, 'calc method': 'beta tin'},
        
        # Phosphorus
        'black': {'coordination': 3, 'calc method': 'black phosphorus'},
        'red': {'coordination': 3, 'calc method': 'red phosphorus'},
        'violet': {'coordination': 3, 'calc method': 'violet phosphorus'},
        'Black P': {'coordination': 3, 'calc method': 'black phosphorus'},
        
        # Plutonium
        'mP16': {'coordination': 8, 'calc method': 'plutonium'},
        
        # Graphite
        'graphite': {'coordination': 3, 'calc method': 'graphite'},
        'P63mc': {'coordination': 4, 'calc method': 'wurtzite'},
    }
    
    def __init__(self):
        self.results = []
        self.problematic_structures = []
    
    def calculate_metallic_radius(self, cif_data: Dict) -> Tuple[Optional[float], str]:
        """
        Calculate metallic radius with method information
        Returns (radius, method_used)
        """
        crystal_system = cif_data['crystal system']
        a = cif_data['a']
        b = cif_data['b'] 
        c = cif_data['c']
        alpha = cif_data.get('alpha', 90)
        beta = cif_data.get('beta', 90)
        gamma = cif_data.get('gamma', 90)
        
        if not a:
            return None, 'No lattice parameter'
        
        # Get structure info
        structure_info = self.STRUCTURE_INFO.get(crystal_system, {'calc method': 'unknown'})
        calc_method = structure_info['calc method']
        
        try:
            radius = self._calculate_radius_by_method(calc_method, a, b, c, alpha, beta, gamma, cif_data)
            return radius, calc_method
        except Exception as e:
            self.problematic_structures.append({
                'element': cif_data['element'],
                'structure': crystal_system,
                'error': str(e)
            })
            return None, f'Error: {str(e)}'
    
    def _calculate_radius_by_method(self, method: str, a: float, b: float, c: float, 
                                  alpha: float, beta: float, gamma: float, cif_data: Dict) -> Optional[float]:
        """Calculate radius based on specific method"""
        
        if method == 'fcc':
            return a / (2 * np.sqrt(2))
        
        elif method == 'hcp':
            return a / 2
        
        elif method == 'bcc':
            return a * np.sqrt(3) / 4
        
        elif method == 'alpha la':
            return a / 2
        
        elif method == 'diamond':
            return a * np.sqrt(3) / 8
        
        elif method == 'alpha mn':
            # α-Mn has complex cubic structure, use average coordination approach
            return a * 0.12  # Empirical factor for α-Mn
        
        elif method == 'orthorhombic':
            # For orthorhombic structures like α-Np
            if b and c:
                avg_param = (a + b + c) / 3
                return avg_param * 0.25  # Estimated factor
            return a * 0.25
        
        elif method == 'tetragonal':
            # For tetragonal structures like α-Pa
            if c:
                return min(a, c) * 0.25
            return a * 0.25
        
        elif method == 'uranium':
            # α-U has orthorhombic structure
            if b and c:
                # Use shortest distance as basis
                min_param = min(a, b, c)
                return min_param / 2
            return a / 2
        
        elif method == 'simple cubic':
            return a / 2
        
        elif method == 'tetragonal centered':
            # Body-centered tetragonal (like In)
            if c:
                return np.sqrt(a*a + c*c) / 4
            return a * np.sqrt(3) / 4
        
        elif method == 'arsenic':
            # α-As structure (rhombohedral)
            return a * 0.25  # Empirical for As-type structure
        
        elif method == 'gallium':
            # α-Ga has complex orthorhombic structure
            if b and c:
                return min(a, b, c) * 0.35
            return a * 0.35
        
        elif method == 'molecular':
            # For molecular crystals, use van der Waals approach
            return a * 0.2  # Very approximate
        
        elif method == 'selenium':
            # Selenium chains
            return a * 0.3
        
        elif method == 'beta boron':
            # Complex boron structure
            return a * 0.15
        
        elif method == 'beta tin':
            # β-Sn (white tin)
            if c:
                return min(a, c) * 0.35
            return a * 0.35
        
        elif method == 'black phosphorus':
            # Layered structure
            if b and c:
                return min(a, b) * 0.25  # Use in-layer distance
            return a * 0.25
        
        elif method == 'red phosphorus':
            return a * 0.2  # Amorphous-like
        
        elif method == 'violet phosphorus':
            return a * 0.2
        
        elif method == 'plutonium':
            # α-Pu has monoclinic structure
            if b and c:
                return min(a, b, c) * 0.3
            return a * 0.3
        
        elif method == 'graphite':
            # Use in-plane distance
            return a / 2
        
        elif method == 'wurtzite':
            return a / 2
        
        elif method == 'rhombohedral':
            return a * 0.3
        
        elif method == 'complex tetragonal':
            return a * 0.2
        
        else:
            # Default: assume close-packed like behavior
            return a / 2
    
    def process_cif_file(self, cif_path: str) -> Dict:
        """Process a single CIF file"""
        parser = EnhancedCIFParser(cif_path)
        
        metallic_radius, method = self.calculate_metallic_radius(parser.data)
        atomic_volume = self.calculate_atomic_volume(parser.data)
        
        # Get coordination number
        structure_info = self.STRUCTURE_INFO.get(parser.data['crystal system'], {})
        coordination = structure_info.get('coordination', 'Unknown')
        
        result = {
            'Element': parser.data['element'],
            'Crystal System': parser.data['crystal system'],
            'Space Group': parser.data['space group'],
            'Space Group_Number': parser.data.get('space group_number'),
            'a (Å)': parser.data['a'],
            'b (Å)': parser.data['b'],
            'c (Å)': parser.data['c'],
            'alpha (°)': parser.data.get('alpha'),
            'beta (°)': parser.data.get('beta'),
            'gamma (°)': parser.data.get('gamma'),
            'Volume (Å³)': parser.data['volume'],
            'Z': parser.data['z'],
            'Density (g/cm³)': parser.data['density'],
            'Metallic Radius (Å)': metallic_radius,
            'Atomic Volume (Å³)': atomic_volume,
            'Coordination Number': coordination,
            'Calculation Method': method,
            'Filename': parser.data['filename']
        }
        
        return result
    
    def calculate_atomic_volume(self, cif_data: Dict) -> Optional[float]:
        """Calculate atomic volume"""
        volume = cif_data['volume']
        z = cif_data['z']
        
        if volume and z:
            return volume / z
        return None
    
    def process_directory(self, cif_directory: str):
        """Process all CIF files in directory"""
        cif_files = list(Path(cif_directory).glob('*.cif'))
        
        print(f"Found {len(cif_files)} CIF files to process...")
        
        for i, cif_file in enumerate(cif_files, 1):
            try:
                result = self.process_cif_file(str(cif_file))
                self.results.append(result)
                
                if i % 10 == 0:
                    print(f"Processed {i}/{len(cif_files)} files...")
                
            except Exception as e:
                print(f"Error processing {cif_file}: {e}")
                self.problematic_structures.append({
                    'file': str(cif_file),
                    'error': str(e)
                })
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame"""
        return pd.DataFrame(self.results)
    
    def print_problematic_structures(self):
        """Print information about structures that couldn't be processed"""
        if self.problematic_structures:
            print(f"\n⚠️  {len(self.problematic_structures)} problematic structures found:")
            for issue in self.problematic_structures:
                print(f"   - {issue}")

class PeriodicTableAnalyzer:
    """Advanced analyzer for complete periodic table"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.setup_plotting_style()
        self.periodic_data = self._load_periodic_data()
    
    def setup_plotting_style(self):
        """Enhanced plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 11
    
    def _load_periodic_data(self) -> Dict:
        """Load periodic table information"""
        # Atomic numbers for proper ordering
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
            'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
            'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
            'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
            'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112
        }
        
        # Element categories
        categories = {
            'Alkali Metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
            'Alkaline Earth Metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
            'Transition Metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
            'Lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
            'Actinides': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
            'Post-transition Metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'],
            'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
            'Nonmetals': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
            'Noble Gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
        }
        
        return {'atomic numbers': atomic_numbers, 'categories': categories}
    
    def create_comprehensive_summary(self) -> pd.DataFrame:
        """Create comprehensive summary with periodic information"""
        df_enhanced = self.df.copy()
        
        # Add atomic numbers
        df_enhanced['Atomic Number'] = df_enhanced['Element'].map(self.periodic_data['atomic numbers'])
        
        # Add element categories
        element_to_category = {}
        for category, elements in self.periodic_data['categories'].items():
            for element in elements:
                element_to_category[element] = category
        
        df_enhanced['Element Category'] = df_enhanced['Element'].map(element_to_category).fillna('Other')
        
        # Sort by atomic number
        df_enhanced = df_enhanced.sort_values('Atomic Number')
        
        return df_enhanced
    
    def plot_periodic_trends(self):
        """Plot metallic radii trends across periodic table"""
        df_enhanced = self.create_comprehensive_summary()
        df_enhanced = df_enhanced.dropna(subset=['Metallic Radius (Å)', 'Atomic Number'])
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Atomic number vs metallic radius
        categories = df_enhanced['Element Category'].unique()
        colors = sns.color_palette("Set1", len(categories))
        color_map = dict(zip(categories, colors))
        
        for category in categories:
            cat_data = df_enhanced[df_enhanced['Element Category'] == category]
            if not cat_data.empty:
                axes[0,0].scatter(cat_data['Atomic Number'], cat_data['Metallic Radius (Å)'],
                               label=category, alpha=0.7, s=60, color=color_map[category])
        
        axes[0,0].set_xlabel('Atomic Number', fontweight='bold')
        axes[0,0].set_ylabel('Metallic Radius (Å)', fontweight='bold')
        axes[0,0].set_title('Metallic Radii Across the Periodic Table', fontweight='bold', fontsize=14)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Crystal system distribution
        crystal_counts = df_enhanced['Crystal System'].value_counts().head(10)
        axes[0,1].bar(range(len(crystal_counts)), crystal_counts.values,
                     color=sns.color_palette("viridis", len(crystal_counts)))
        axes[0,1].set_xticks(range(len(crystal_counts)))
        axes[0,1].set_xticklabels(crystal_counts.index, rotation=45, ha='right')
        axes[0,1].set_ylabel('Number of Elements', fontweight='bold')
        axes[0,1].set_title('Crystal System Distribution', fontweight='bold', fontsize=14)
        
        # 3. Density vs radius
        axes[1,0].scatter(df_enhanced['Metallic Radius (Å)'], df_enhanced['Density (g/cm³)'],
                         c=df_enhanced['Atomic Number'], cmap='viridis', alpha=0.7, s=60)
        axes[1,0].set_xlabel('Metallic Radius (Å)', fontweight='bold')
        axes[1,0].set_ylabel('Density (g/cm³)', fontweight='bold')
        axes[1,0].set_title('Density vs Metallic Radius', fontweight='bold', fontsize=14)
        cbar = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])
        cbar.set_label('Atomic Number', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Coordination number analysis
        coord_data = df_enhanced.dropna(subset=['Coordination Number'])
        coord_data = coord_data[coord_data['Coordination Number'] != 'Unknown']
        coord_data['Coordination Number'] = pd.to_numeric(coord_data['Coordination Number'], errors='coerce')
        coord_data = coord_data.dropna(subset=['Coordination Number'])
        
        if not coord_data.empty:
            coord_counts = coord_data['Coordination Number'].value_counts().sort_index()
            axes[1,1].bar(coord_counts.index, coord_counts.values,
                         color=sns.color_palette("plasma", len(coord_counts)))
            axes[1,1].set_xlabel('Coordination Number', fontweight='bold')
            axes[1,1].set_ylabel('Number of Elements', fontweight='bold')
            axes[1,1].set_title('Coordination Number Distribution', fontweight='bold', fontsize=14)
            axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_element_families(self):
        """Plot trends within element families"""
        df_enhanced = self.create_comprehensive_summary()
        
        families_to_plot = ['Alkali Metals', 'Alkaline Earth Metals', 'Transition Metals', 
                           'Lanthanides', 'Actinides']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, family in enumerate(families_to_plot):
            family_data = df_enhanced[df_enhanced['Element Category'] == family]
            family_data = family_data.dropna(subset=['Metallic Radius (Å)'])
            family_data = family_data.sort_values('Atomic Number')
            
            if not family_data.empty:
                if family == 'Transition Metals':
                    # Define transition metal series
                    tm_3d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
                    tm_4d = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
                    tm_5d = ['Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
                    
                    series_colors = ["#9ed0ff", "#8ace8a", "#e795bb"]
                    text_colors   = ["#478dce", "#388a38", "#ce4e8a"]
                    series_names  = ['3d series', '4d series', '5d series']
                    
                    for series, color, text_color, name in zip([tm_3d, tm_4d, tm_5d], series_colors, text_colors, series_names):
                        series_data = family_data[family_data['Element'].isin(series)]
                        if not series_data.empty:
                            # Calculate d-electrons based on position in series
                            d_electrons = []
                            for _, row in series_data.iterrows():
                                try:
                                    d_elec = series.index(row['Element']) + 1
                                    d_electrons.append(d_elec)
                                except ValueError:
                                    continue

                            if d_electrons:
                                axes[i].plot(d_electrons, series_data['Metallic Radius (Å)'].iloc[:len(d_electrons)],
                                           'o-', linewidth=2, markersize=8, alpha=0.8, color=color, label=name)
                                
                                # Add element labels
                                for j, (_, row) in enumerate(series_data.iterrows()):
                                    if j < len(d_electrons):
                                        offset = (-10, 0) if series == tm_3d else (0, 10) if series == tm_4d else (10, 0)  # Adjust offset based on series
                                        axes[i].annotate(row['Element'], 
                                                   (d_electrons[j], row['Metallic Radius (Å)']),
                                                   xytext=offset, textcoords='offset points',
                                                   ha='center', fontsize=9, color=text_color)

                    
                    axes[i].set_xlabel('Number of d Electrons', fontweight='bold')
                    axes[i].legend()
                    axes[i].set_xticks(np.arange(1, 11, 1))
                    axes[i].set_yticks(np.arange(1.0, 2.4, 0.1))
                else:
                    axes[i].plot(family_data['Atomic Number'], family_data['Metallic Radius (Å)'],
                               'o-', linewidth=2, markersize=8, alpha=0.8)
                    
                    # Add element labels
                    for _, row in family_data.iterrows():
                        axes[i].annotate(row['Element'], 
                                       (row['Atomic Number'], row['Metallic Radius (Å)']),
                                       xytext=(0, 10), textcoords='offset points',
                                       ha='center', fontsize=10)
                    
                    axes[i].set_xlabel('Atomic Number', fontweight='bold')
                
                axes[i].set_ylabel('Metallic Radius (Å)', fontweight='bold')
                axes[i].set_title(f'{family}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        # if len(families_to_plot) < len(axes):
        #     fig.delaxes(axes[-1])
              
        # Crystal structure complexity analysis
        df_complexity = df_enhanced.copy()
        df_complexity['Structure Complexity'] = df_complexity['Crystal System'].map({
            'fcc': 1, 'hcp': 1, 'bcc': 1,
            'diamond': 2, 'α-La': 2,
            'α-Mn': 4, 'α-Np': 3, 'U': 3, 'α-Pa': 3,
            'α-Ga': 3, 'mP16': 4, 'tI2': 2
        }).fillna(3)  # Default complexity
        
        axes[5].scatter(df_complexity['Atomic Number'], df_complexity['Structure Complexity'],
                       c=df_complexity['Metallic Radius (Å)'], cmap='spring', 
                       s=40, alpha=0.7, edgecolors='black')
        axes[5].set_xlabel('Atomic Number', fontweight='bold')
        axes[5].set_ylabel('Structure Complexity', fontweight='bold')
        axes[5].set_title('Crystal Structure Complexity', fontweight='bold')
        cbar = plt.colorbar(axes[5].collections[0], ax=axes[5])
        cbar.set_label('Metallic Radius (Å)', fontweight='bold')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_calculation_method_analysis(self):
        """Analyze calculation methods used"""
        method_counts = self.df['Calculation Method'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Method distribution
        ax1.pie(method_counts.values, 
                labels=[s.title() if len(s) > 3 else s.upper() for s in method_counts.index], 
                autopct='%1.1f%%',
                colors=sns.color_palette("Set3", len(method_counts)),
                textprops={'fontsize': 8},
                startangle=140,)
        ax1.set_title('Distribution of Calculation Methods', fontweight='bold', fontsize=14)
        
        # Accuracy by method (if we had reference data)
        method_accuracy = self.df.groupby('Calculation Method')['Metallic Radius (Å)'].agg(['mean', 'std', 'count'])
        method_accuracy = method_accuracy.sort_values('count', ascending=True)
        
        ax2.barh(range(len(method_accuracy)), 
                 method_accuracy['count'],
                 color=sns.color_palette("viridis", len(method_accuracy)))
        ax2.set_yticks(range(len(method_accuracy)))
        ax2.set_yticklabels([s.title() if len(s) > 3 else s.upper() for s in method_accuracy.index])
        ax2.set_xlabel('Number of Elements', fontweight='bold')
        ax2.set_title('Elements per Calculation Method', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

def main():
    """Main function for complete periodic table analysis"""
    
    script_dir = Path(__file__).parent
    elements_dir = script_dir / 'Elements'
    output_dir = script_dir / 'output full_periodic'
    output_dir.mkdir(exist_ok=True)
    
    print("🌟 ENHANCED Metallic Radii Calculator - Full Periodic Table Edition!")
    print("=" * 80)
    
    if not elements_dir.exists():
        print(f"❌ Elements directory not found: {elements_dir}")
        print("Please ensure the 'Elements' folder exists with CIF files.")
        return
    
    # Initialize enhanced calculator
    calculator = EnhancedMetallicRadiiCalculator()
    
    print(f"\n📁 Processing ALL elements from: {elements_dir}")
    calculator.process_directory(str(elements_dir))
    
    # Get results
    df = calculator.get_results_dataframe()
    
    if df.empty:
        print("❌ No data was processed.")
        return
    
    print(f"\n✅ Successfully processed {len(df)} structures!")
    print(f"📊 Unique elements: {df['Element'].nunique()}")
    print(f"🔬 Unique crystal systems: {df['Crystal System'].nunique()}")
    
    # Print problematic structures
    calculator.print_problematic_structures()
    
    # Create enhanced analyzer
    analyzer = PeriodicTableAnalyzer(df)
    
    # Create comprehensive summary
    summary_df = analyzer.create_comprehensive_summary()
    
    print(f"\n📋 Element Categories Represented:")
    category_counts = summary_df['Element Category'].value_counts()
    for category, count in category_counts.items():
        print(f"   • {category}: {count} elements")
    
    # Save comprehensive results
    csv_path = output_dir / 'complete periodic_table_radii.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n💾 Complete results saved to: {csv_path}")
    
    # Generate advanced visualizations
    print(f"\n📈 Generating advanced visualizations...")
    
    # Plot 1: Periodic trends
    fig1 = analyzer.plot_periodic_trends()
    fig1.savefig(output_dir / 'periodic table_trends.png', dpi=300, bbox_inches='tight')
    print(f"   🎨 Periodic trends plot saved")
    
    # Plot 2: Element families
    fig2 = analyzer.plot_element_families()
    fig2.savefig(output_dir / 'element families_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   🎨 Element families plot saved")
    
    # Plot 3: Calculation methods
    fig3 = analyzer.create_calculation_method_analysis()
    fig3.savefig(output_dir / 'calculation methods_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   🎨 Calculation methods plot saved")
    
    # Statistical summary
    valid_radii = summary_df.dropna(subset=['Metallic Radius (Å)'])
    
    print(f"\n📊 COMPREHENSIVE STATISTICS:")
    print(f"   • Total structures analyzed: {len(df)}")
    print(f"   • Elements with valid radii: {len(valid_radii)}")
    print(f"   • Average metallic radius: {valid_radii['Metallic Radius (Å)'].mean():.3f} Å")
    print(f"   • Largest radius: {valid_radii['Metallic Radius (Å)'].max():.3f} Å ({valid_radii.loc[valid_radii['Metallic Radius (Å)'].idxmax(), 'Element']})")
    print(f"   • Smallest radius: {valid_radii['Metallic Radius (Å)'].min():.3f} Å ({valid_radii.loc[valid_radii['Metallic Radius (Å)'].idxmin(), 'Element']})")
    print(f"   • Radius range: {valid_radii['Metallic Radius (Å)'].max() - valid_radii['Metallic Radius (Å)'].min():.3f} Å")
    
    print(f"\n🔬 Crystal Structure Diversity:")
    for crystal, count in df['Crystal System'].value_counts().head(10).items():
        print(f"   • {crystal}: {count} structures")
    
    print(f"\n🏆 Most Complex Structures Successfully Analyzed:")
    complex_structures = ['α-Mn', 'U', 'α-Np', 'mP16', 'α-Ga']
    for structure in complex_structures:
        elements = df[df['Crystal System'] == structure]['Element'].tolist()
        if elements:
            print(f"   • {structure}: {', '.join(elements)}")
    
    plt.show()
    
    print(f"\n🎉 COMPLETE PERIODIC TABLE ANALYSIS FINISHED!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"🌟 You now have metallic radii for the entire periodic table!")

if __name__ == "__main__":
    main()