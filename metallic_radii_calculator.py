#!/usr/bin/env python3
"""
Metallic Radii Calculator from CIF Files
Calculates metallic radii from crystallographic data of rare earth elements
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

class CIFParser:
    """Parser for CIF (Crystallographic Information File) format"""
    
    def __init__(self, cif_path: str):
        self.cif_path = cif_path
        self.data = {}
        self.parse_cif()
    
    def parse_cif(self):
        """Parse CIF file and extract relevant crystallographic data"""
        with open(self.cif_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic information
        self.data['filename'] = os.path.basename(self.cif_path)
        
        # Extract element name
        element_match = re.search(r'([A-Z][a-z]?)_', self.data['filename'])
        self.data['element'] = element_match.group(1) if element_match else 'Unknown'
        
        # Extract crystal system from filename
        crystal_match = re.search(r'_\(([^)]+)\)_', self.data['filename'])
        self.data['crystal_system'] = crystal_match.group(1) if crystal_match else 'Unknown'
        
        # Extract lattice parameters
        self.data['a'] = self._extract_parameter(content, r'_cell_length_a\s+([\d.]+)')
        self.data['b'] = self._extract_parameter(content, r'_cell_length_b\s+([\d.]+)')
        self.data['c'] = self._extract_parameter(content, r'_cell_length_c\s+([\d.]+)')
        
        # Extract cell angles
        self.data['alpha'] = self._extract_parameter(content, r'_cell_angle_alpha\s+([\d.]+)')
        self.data['beta'] = self._extract_parameter(content, r'_cell_angle_beta\s+([\d.]+)')
        self.data['gamma'] = self._extract_parameter(content, r'_cell_angle_gamma\s+([\d.]+)')
        
        # Extract cell volume and Z (number of formula units)
        self.data['volume'] = self._extract_parameter(content, r'_cell_volume\s+([\d.]+)')
        self.data['z'] = self._extract_parameter(content, r'_cell_formula_units_Z\s+(\d+)', int)
        
        # Extract space group
        space_group_match = re.search(r"_space_group_name_H-M_alt\s+'([^']+)'", content)
        self.data['space_group'] = space_group_match.group(1) if space_group_match else 'Unknown'
        
        # Extract density
        self.data['density'] = self._extract_parameter(content, r'_exptl_crystal_density_diffrn\s+([\d.]+)')
    
    def _extract_parameter(self, content: str, pattern: str, dtype=float):
        """Extract a parameter using regex pattern"""
        match = re.search(pattern, content)
        if match:
            try:
                return dtype(match.group(1))
            except ValueError:
                return None
        return None

class MetallicRadiiCalculator:
    """Calculate metallic radii from crystallographic data"""
    
    # Coordination numbers for different crystal structures
    COORDINATION_NUMBERS = {
        'fcc': 12,     # Face-centered cubic
        'hcp': 12,     # Hexagonal close-packed
        'bcc': 8,      # Body-centered cubic
        'α-La': 12,    # α-Lanthanum structure (similar to hcp)
    }
    
    def __init__(self):
        self.results = []
    
    def calculate_metallic_radius(self, cif_data: Dict) -> Optional[float]:
        """
        Calculate metallic radius from CIF data
        
        The metallic radius is calculated as half the nearest-neighbor distance
        """
        crystal_system = cif_data['crystal_system']
        a = cif_data['a']
        b = cif_data['b'] 
        c = cif_data['c']
        
        if not all([a, b, c]):
            return None
        
        # Calculate nearest neighbor distance based on crystal structure
        if crystal_system == 'fcc':
            # In FCC, nearest neighbors are at a/√2
            nearest_neighbor_distance = a / np.sqrt(2)
        elif crystal_system == 'hcp':
            # In HCP, nearest neighbors are at distance a
            nearest_neighbor_distance = a
        elif crystal_system == 'bcc':
            # In BCC, nearest neighbors are at a√3/2
            nearest_neighbor_distance = a * np.sqrt(3) / 2
        elif crystal_system == 'α-La':
            # α-Lanthanum structure, similar to hcp
            nearest_neighbor_distance = a
        else:
            # Default case - assume simple cubic
            nearest_neighbor_distance = a
        
        # Metallic radius is half the nearest neighbor distance
        metallic_radius = nearest_neighbor_distance / 2
        
        return metallic_radius
    
    def calculate_atomic_volume(self, cif_data: Dict) -> Optional[float]:
        """Calculate atomic volume from unit cell volume and Z"""
        volume = cif_data['volume']
        z = cif_data['z']
        
        if volume and z:
            return volume / z
        return None
    
    def process_cif_file(self, cif_path: str) -> Dict:
        """Process a single CIF file and calculate metallic radius"""
        parser = CIFParser(cif_path)
        
        metallic_radius = self.calculate_metallic_radius(parser.data)
        atomic_volume = self.calculate_atomic_volume(parser.data)
        
        result = {
            'Element': parser.data['element'],
            'Crystal_System': parser.data['crystal_system'],
            'Space_Group': parser.data['space_group'],
            'a (Å)': parser.data['a'],
            'b (Å)': parser.data['b'],
            'c (Å)': parser.data['c'],
            'Volume (Å³)': parser.data['volume'],
            'Z': parser.data['z'],
            'Density (g/cm³)': parser.data['density'],
            'Metallic_Radius (Å)': metallic_radius,
            'Atomic_Volume (Å³)': atomic_volume,
            'Coordination_Number': self.COORDINATION_NUMBERS.get(parser.data['crystal_system'], 'Unknown'),
            'Filename': parser.data['filename']
        }
        
        return result
    
    def process_directory(self, cif_directory: str):
        """Process all CIF files in a directory"""
        cif_files = Path(cif_directory).glob('*.cif')
        
        for cif_file in cif_files:
            try:
                result = self.process_cif_file(str(cif_file))
                self.results.append(result)
                print(f"Processed: {result['Element']} ({result['Crystal_System']})")
            except Exception as e:
                print(f"Error processing {cif_file}: {e}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame"""
        return pd.DataFrame(self.results)

class MetallicRadiiAnalyzer:
    """Analyze and visualize metallic radii data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table with key properties"""
        summary_cols = [
            'Element', 'Crystal_System', 'Metallic_Radius (Å)', 
            'Atomic_Volume (Å³)', 'Coordination_Number', 'a (Å)', 'Density (g/cm³)'
        ]
        
        summary_df = self.df[summary_cols].copy()
        summary_df = summary_df.sort_values('Metallic_Radius (Å)', ascending=False)
        
        return summary_df
    
    def plot_metallic_radii_by_element(self):
        """Plot metallic radii by element"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by metallic radius
        df_sorted = self.df.sort_values('Metallic_Radius (Å)', ascending=True)
        
        # Create color map based on crystal system
        crystal_systems = df_sorted['Crystal_System'].unique()
        colors = sns.color_palette("Set2", len(crystal_systems))
        color_map = dict(zip(crystal_systems, colors))
        
        bar_colors = [color_map[cs] for cs in df_sorted['Crystal_System']]
        
        bars = ax.bar(df_sorted['Element'], df_sorted['Metallic_Radius (Å)'], 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Element', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metallic Radius (Å)', fontsize=14, fontweight='bold')
        ax.set_title('Metallic Radii of Rare Earth Elements', fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, df_sorted['Metallic_Radius (Å)']):
            if value:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Create legend for crystal systems
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[cs], 
                                       edgecolor='black', alpha=0.8, label=cs) 
                          for cs in crystal_systems]
        ax.legend(handles=legend_elements, title='Crystal System', 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    def plot_radius_vs_atomic_volume(self):
        """Plot metallic radius vs atomic volume"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        crystal_systems = self.df['Crystal_System'].unique()
        colors = sns.color_palette("Set1", len(crystal_systems))
        
        for i, cs in enumerate(crystal_systems):
            cs_data = self.df[self.df['Crystal_System'] == cs]
            ax.scatter(cs_data['Atomic_Volume (Å³)'], cs_data['Metallic_Radius (Å)'],
                      c=[colors[i]], label=cs, s=100, alpha=0.7, edgecolors='black')
            
            # Add element labels
            for _, row in cs_data.iterrows():
                if pd.notna(row['Metallic_Radius (Å)']):
                    ax.annotate(row['Element'], 
                              (row['Atomic_Volume (Å³)'], row['Metallic_Radius (Å)']),
                              xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Atomic Volume (Å³)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metallic Radius (Å)', fontsize=14, fontweight='bold')
        ax.set_title('Metallic Radius vs Atomic Volume', fontsize=16, fontweight='bold')
        ax.legend(title='Crystal System')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_crystal_system_distribution(self):
        """Plot distribution of crystal systems"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        crystal_counts = self.df['Crystal_System'].value_counts()
        ax1.bar(crystal_counts.index, crystal_counts.values, 
               color=sns.color_palette("Set2", len(crystal_counts)))
        ax1.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Elements', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Crystal Systems', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(crystal_counts.values, labels=crystal_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette("Set2", len(crystal_counts)))
        ax2.set_title('Crystal System Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_lattice_parameters(self):
        """Plot lattice parameters comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # a parameter
        df_sorted = self.df.sort_values('a (Å)', ascending=True)
        axes[0,0].bar(df_sorted['Element'], df_sorted['a (Å)'], 
                     color='skyblue', alpha=0.8, edgecolor='black')
        axes[0,0].set_ylabel('a (Å)', fontweight='bold')
        axes[0,0].set_title('Lattice Parameter a', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # c parameter
        df_c_sorted = self.df.sort_values('c (Å)', ascending=True)
        axes[0,1].bar(df_c_sorted['Element'], df_c_sorted['c (Å)'], 
                     color='lightcoral', alpha=0.8, edgecolor='black')
        axes[0,1].set_ylabel('c (Å)', fontweight='bold')
        axes[0,1].set_title('Lattice Parameter c', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # c/a ratio for hexagonal systems
        hex_data = self.df[self.df['Crystal_System'].isin(['hcp', 'α-La'])].copy()
        hex_data['c/a_ratio'] = hex_data['c (Å)'] / hex_data['a (Å)']
        hex_data = hex_data.sort_values('c/a_ratio', ascending=True)
        
        if not hex_data.empty:
            axes[1,0].bar(hex_data['Element'], hex_data['c/a_ratio'], 
                         color='lightgreen', alpha=0.8, edgecolor='black')
            axes[1,0].set_ylabel('c/a ratio', fontweight='bold')
            axes[1,0].set_title('c/a Ratio (Hexagonal Systems)', fontweight='bold')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].axhline(y=1.633, color='red', linestyle='--', alpha=0.7, 
                             label='Ideal HCP (1.633)')
            axes[1,0].legend()
        
        # Density vs Metallic Radius
        crystal_systems = self.df['Crystal_System'].unique()
        colors = sns.color_palette("Set1", len(crystal_systems))
        
        for i, cs in enumerate(crystal_systems):
            cs_data = self.df[self.df['Crystal_System'] == cs]
            axes[1,1].scatter(cs_data['Metallic_Radius (Å)'], cs_data['Density (g/cm³)'],
                            c=[colors[i]], label=cs, s=80, alpha=0.7, edgecolors='black')
        
        axes[1,1].set_xlabel('Metallic Radius (Å)', fontweight='bold')
        axes[1,1].set_ylabel('Density (g/cm³)', fontweight='bold')
        axes[1,1].set_title('Density vs Metallic Radius', fontweight='bold')
        axes[1,1].legend(title='Crystal System')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def main():
    """Main function to run the metallic radii analysis"""
    
    # Set up paths
    script_dir = Path(__file__).parent
    cif_dir = script_dir / 'cif'
    output_dir = script_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    print("🔬 Metallic Radii Calculator for Rare Earth Elements")
    print("=" * 60)
    
    # Initialize calculator and process CIF files
    calculator = MetallicRadiiCalculator()
    
    print(f"\n📁 Processing CIF files from: {cif_dir}")
    calculator.process_directory(str(cif_dir))
    
    # Get results dataframe
    df = calculator.get_results_dataframe()
    
    if df.empty:
        print("❌ No data was processed. Check CIF files and directory path.")
        return
    
    print(f"\n✅ Successfully processed {len(df)} elements")
    
    # Create analyzer
    analyzer = MetallicRadiiAnalyzer(df)
    
    # Create and save summary table
    summary_df = analyzer.create_summary_table()
    
    print("\n📊 Summary Table:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Save detailed results to CSV
    csv_path = output_dir / 'metallic_radii_results.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n💾 Detailed results saved to: {csv_path}")
    
    # Save summary table
    summary_path = output_dir / 'metallic_radii_summary.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.3f')
    print(f"💾 Summary table saved to: {summary_path}")
    
    # Generate and save plots
    print("\n📈 Generating plots...")
    
    # Plot 1: Metallic radii by element
    fig1 = analyzer.plot_metallic_radii_by_element()
    fig1.savefig(output_dir / 'metallic_radii_by_element.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Metallic radii plot saved")
    
    # Plot 2: Radius vs atomic volume
    fig2 = analyzer.plot_radius_vs_atomic_volume()
    fig2.savefig(output_dir / 'radius_vs_atomic_volume.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Radius vs volume plot saved")
    
    # Plot 3: Crystal system distribution
    fig3 = analyzer.plot_crystal_system_distribution()
    fig3.savefig(output_dir / 'crystal_system_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Crystal system distribution plot saved")
    
    # Plot 4: Lattice parameters
    fig4 = analyzer.plot_lattice_parameters()
    fig4.savefig(output_dir / 'lattice_parameters_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Lattice parameters plot saved")
    
    # Display key statistics
    print("\n📈 Key Statistics:")
    print(f"   • Average metallic radius: {df['Metallic_Radius (Å)'].mean():.3f} Å")
    print(f"   • Largest metallic radius: {df['Metallic_Radius (Å)'].max():.3f} Å ({df.loc[df['Metallic_Radius (Å)'].idxmax(), 'Element']})")
    print(f"   • Smallest metallic radius: {df['Metallic_Radius (Å)'].min():.3f} Å ({df.loc[df['Metallic_Radius (Å)'].idxmin(), 'Element']})")
    print(f"   • Range: {df['Metallic_Radius (Å)'].max() - df['Metallic_Radius (Å)'].min():.3f} Å")
    
    # Crystal system statistics
    print(f"\n🔬 Crystal System Distribution:")
    for cs, count in df['Crystal_System'].value_counts().items():
        print(f"   • {cs}: {count} elements")
    
    plt.show()
    
    print(f"\n🎉 Analysis complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()