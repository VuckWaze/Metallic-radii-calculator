"""
Visualization module for Enhanced Metallic Radii Calculator.

Provides comprehensive plotting and analysis capabilities including
periodic trends, element families, crystal structures, and advanced
statistical visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from config import METALLIC_ELEMENT_CATEGORIES

class PeriodicTableVisualizer:
    """
    Advanced visualizer for complete periodic table analysis.
    
    Creates publication-quality plots for metallic radii trends,
    element families, crystal structures, and statistical analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer with results DataFrame.
        
        Args:
            df: DataFrame containing metallic radii calculation results
        """
        self.df = df.copy()
        self.setup_plotting_style()
        self.periodic_data = self._load_periodic_data()
        self.enhanced_df = self._enhance_dataframe()
    
    def setup_plotting_style(self) -> None:
        """Set up professional plotting style."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _load_periodic_data(self) -> Dict:
        """Load periodic table information for enhanced analysis."""
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
        
        return {'atomic_numbers': atomic_numbers, 'categories': categories}
    
    def _enhance_dataframe(self) -> pd.DataFrame:
        """Enhance DataFrame with periodic table information."""
        df_enhanced = self.df.copy()
        
        # Add atomic numbers
        df_enhanced['Atomic Number'] = df_enhanced['Element'].map(self.periodic_data['atomic_numbers'])
        
        # Add element categories
        element_to_category = {}
        for category, elements in self.periodic_data['categories'].items():
            for element in elements:
                element_to_category[element] = category
        
        df_enhanced['Element Category'] = df_enhanced['Element'].map(element_to_category).fillna('Other')
        
        # Sort by atomic number
        df_enhanced = df_enhanced.sort_values('Atomic Number')
        
        return df_enhanced
    
    def plot_periodic_trends(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive periodic trends analysis.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        df_plot = self.enhanced_df.dropna(subset=['Primary Radius (Å)', 'Atomic Number'])
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Atomic number vs metallic radius by category
        categories = df_plot['Element Category'].unique()
        colors = sns.color_palette("Set1", len(categories))
        color_map = dict(zip(categories, colors))
        
        for category in categories:
            cat_data = df_plot[df_plot['Element Category'] == category]
            if not cat_data.empty:
                axes[0,0].scatter(cat_data['Atomic Number'], cat_data['Primary Radius (Å)'],
                               label=category, alpha=0.7, s=60, color=color_map[category])
        
        axes[0,0].set_xlabel('Atomic Number', fontweight='bold')
        axes[0,0].set_ylabel('Metallic Radius (Å)', fontweight='bold')
        axes[0,0].set_title('Metallic Radii Across the Periodic Table', fontweight='bold', fontsize=14)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Crystal system distribution
        crystal_counts = df_plot['Crystal System'].value_counts().head(10)
        axes[0,1].bar(range(len(crystal_counts)), crystal_counts.values,
                     color=sns.color_palette("viridis", len(crystal_counts)))
        axes[0,1].set_xticks(range(len(crystal_counts)))
        axes[0,1].set_xticklabels(crystal_counts.index, rotation=45, ha='right')
        axes[0,1].set_ylabel('Number of Elements', fontweight='bold')
        axes[0,1].set_title('Crystal System Distribution', fontweight='bold', fontsize=14)
        
        # 3. Density vs radius
        density_data = df_plot.dropna(subset=['Density (g/cm³)'])
        if not density_data.empty:
            scatter = axes[1,0].scatter(density_data['Primary Radius (Å)'], density_data['Density (g/cm³)'],
                                      c=density_data['Atomic Number'], cmap='viridis', alpha=0.7, s=60)
            axes[1,0].set_xlabel('Metallic Radius (Å)', fontweight='bold')
            axes[1,0].set_ylabel('Density (g/cm³)', fontweight='bold')
            axes[1,0].set_title('Density vs Metallic Radius', fontweight='bold', fontsize=14)
            cbar = plt.colorbar(scatter, ax=axes[1,0])
            cbar.set_label('Atomic Number', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Coordination number analysis
        coord_data = df_plot.dropna(subset=['Coordination Number'])
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
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   🎨 Periodic trends plot saved to: {save_path}")
        
        return fig
    
    def plot_element_families(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trends within element families.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        families_to_plot = ['Alkali Metals', 'Alkaline Earth Metals', 'Transition Metals', 
                           'Lanthanides', 'Actinides']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, family in enumerate(families_to_plot):
            family_data = self.enhanced_df[self.enhanced_df['Element Category'] == family]
            family_data = family_data.dropna(subset=['Primary Radius (Å)'])
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
                            radii = []
                            for _, row in series_data.iterrows():
                                try:
                                    d_elec = series.index(row['Element']) + 1
                                    d_electrons.append(d_elec)
                                    radii.append(row['Primary Radius (Å)'])
                                except ValueError:
                                    continue

                            if d_electrons:
                                axes[i].plot(d_electrons, radii,
                                           'o-', linewidth=2, markersize=8, alpha=0.8, color=color, label=name)
                                
                                # Add element labels
                                for j, (d_elec, radius) in enumerate(zip(d_electrons, radii)):
                                    element = series_data.iloc[j]['Element']
                                    offset = (-10, 0) if series == tm_3d else (0, 10) if series == tm_4d else (10, 0)
                                    axes[i].annotate(element, 
                                               (d_elec, radius),
                                               xytext=offset, textcoords='offset points',
                                               ha='center', fontsize=9, color=text_color)
                    
                    axes[i].set_xlabel('Number of d Electrons', fontweight='bold')
                    axes[i].legend()
                    axes[i].set_xticks(np.arange(1, 11, 1))
                else:
                    # Regular family plot
                    axes[i].plot(family_data['Atomic Number'], family_data['Primary Radius (Å)'],
                               'o-', linewidth=2, markersize=8, alpha=0.8)
                    
                    # Add element labels
                    for _, row in family_data.iterrows():
                        axes[i].annotate(row['Element'], 
                                       (row['Atomic Number'], row['Primary Radius (Å)']),
                                       xytext=(0, 10), textcoords='offset points',
                                       ha='center', fontsize=10)
                    
                    axes[i].set_xlabel('Atomic Number', fontweight='bold')
                
                axes[i].set_ylabel('Metallic Radius (Å)', fontweight='bold')
                axes[i].set_title(f'{family}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        # Radius range by element family - Box plot analysis
        family_radius_data = []
        family_labels = []
        family_colors = []
        
        # Get families with sufficient data (at least 3 elements)
        all_families = ['Alkali Metals', 'Alkaline Earth Metals', 'Transition Metals', 
                       'Lanthanides', 'Actinides', 'Post-transition Metals']
        
        color_palette = sns.color_palette("Set2", len(all_families))
        
        for i, family in enumerate(all_families):
            family_data = self.enhanced_df[self.enhanced_df['Element Category'] == family]
            family_radii = family_data['Primary Radius (Å)'].dropna()
            
            if len(family_radii) >= 2:  # Need at least 2 points for a meaningful box plot
                family_radius_data.append(family_radii.values)
                family_labels.append(family)
                family_colors.append(color_palette[i])
        
        if family_radius_data:
            # Create box plot
            box_plot = axes[5].boxplot(family_radius_data, 
                                     labels=family_labels,
                                     patch_artist=True,
                                     showmeans=True,
                                     meanline=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], family_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style the plot
            axes[5].set_ylabel('Metallic Radius (Å)', fontweight='bold')
            axes[5].set_title('Metallic Radius Distribution by Element Family', fontweight='bold')
            axes[5].tick_params(axis='x', rotation=45)
            axes[5].grid(True, alpha=0.3, axis='y')
            
            # Add statistical annotations
            for i, (radii, label) in enumerate(zip(family_radius_data, family_labels)):
                n_elements = len(radii)
                axes[5].text(i+1, max(radii) + 0.05, f'n={n_elements}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            # Fallback if no family data available
            axes[5].text(0.5, 0.5, 'Insufficient family data\nfor box plot analysis', 
                       ha='center', va='center', transform=axes[5].transAxes,
                       fontsize=12, fontweight='bold')
            axes[5].set_title('Radius Distribution by Family', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   🎨 Element families plot saved to: {save_path}")
        
        return fig
    
    def plot_calculation_methods(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize calculation methods used.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Create method distribution based on available data
        methods = []
        for _, row in self.df.iterrows():
            if pd.notna(row.get('Corrected Volume Radius (Å)')):
                methods.append('Volume-based (Corrected)')
            elif pd.notna(row.get('Traditional Radius (Å)')):
                methods.append('Traditional Structure-specific')
            else:
                methods.append('Unknown')
        
        method_series = pd.Series(methods)
        method_counts = method_series.value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Method distribution pie chart
        colors = sns.color_palette("Set3", len(method_counts))
        wedges, texts, autotexts = ax1.pie(method_counts.values, 
                                          labels=method_counts.index, 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          textprops={'fontsize': 10},
                                          startangle=140)
        ax1.set_title('Distribution of Calculation Methods', fontweight='bold', fontsize=14)
        
        # Method accuracy comparison
        traditional_radii = self.df['Traditional Radius (Å)'].dropna()
        volume_radii = self.df['Corrected Volume Radius (Å)'].dropna()
        
        if len(traditional_radii) > 0 and len(volume_radii) > 0:
            ax2.hist([traditional_radii, volume_radii], 
                    bins=20, alpha=0.7, 
                    label=['Traditional', 'Volume-based (Corrected)'],
                    color=['skyblue', 'lightcoral'])
            ax2.set_xlabel('Metallic Radius (Å)', fontweight='bold')
            ax2.set_ylabel('Number of Elements', fontweight='bold')
            ax2.set_title('Radius Distribution by Method', fontweight='bold', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Fallback: show method usage by crystal system
            crystal_method_counts = self.df['Crystal System'].value_counts().head(8)
            ax2.barh(range(len(crystal_method_counts)), 
                    crystal_method_counts.values,
                    color=sns.color_palette("viridis", len(crystal_method_counts)))
            ax2.set_yticks(range(len(crystal_method_counts)))
            ax2.set_yticklabels(crystal_method_counts.index)
            ax2.set_xlabel('Number of Elements', fontweight='bold')
            ax2.set_title('Elements per Crystal Structure', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   🎨 Calculation methods plot saved to: {save_path}")
        
        return fig
    
    def plot_comparison_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between traditional and volume-based methods.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Filter data with both traditional and volume-based calculations
        comparison_data = self.df.dropna(subset=['Traditional Radius (Å)', 'Corrected Volume Radius (Å)'])
        
        if comparison_data.empty:
            print("⚠️  No data available for method comparison")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Direct comparison scatter plot
        axes[0,0].scatter(comparison_data['Traditional Radius (Å)'], 
                         comparison_data['Corrected Volume Radius (Å)'],
                         alpha=0.7, s=60, color='steelblue')
        
        # Add perfect correlation line
        min_val = min(comparison_data['Traditional Radius (Å)'].min(), 
                     comparison_data['Corrected Volume Radius (Å)'].min())
        max_val = max(comparison_data['Traditional Radius (Å)'].max(), 
                     comparison_data['Corrected Volume Radius (Å)'].max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Agreement')
        
        axes[0,0].set_xlabel('Traditional Radius (Å)', fontweight='bold')
        axes[0,0].set_ylabel('Volume-based Radius (Å)', fontweight='bold')
        axes[0,0].set_title('Method Comparison: Traditional vs Volume-based', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Difference analysis
        differences = comparison_data['Corrected Volume Radius (Å)'] - comparison_data['Traditional Radius (Å)']
        axes[0,1].hist(differences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].axvline(differences.mean(), color='red', linestyle='--', 
                         label=f'Mean: {differences.mean():.3f} Å')
        axes[0,1].set_xlabel('Difference (Volume - Traditional) (Å)', fontweight='bold')
        axes[0,1].set_ylabel('Number of Elements', fontweight='bold')
        axes[0,1].set_title('Difference Distribution', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Relative error by element category
        comparison_enhanced = comparison_data.merge(
            self.enhanced_df[['Element', 'Element Category']], on='Element', how='left'
        )
        
        relative_errors = (differences / comparison_data['Traditional Radius (Å)']) * 100
        comparison_enhanced['Relative Error (%)'] = relative_errors
        
        categories_with_data = comparison_enhanced['Element Category'].value_counts()
        categories_to_plot = categories_with_data[categories_with_data >= 2].index[:6]
        
        if len(categories_to_plot) > 0:
            category_data = [comparison_enhanced[comparison_enhanced['Element Category'] == cat]['Relative Error (%)'] 
                           for cat in categories_to_plot]
            axes[1,0].boxplot(category_data, labels=categories_to_plot)
            axes[1,0].set_xlabel('Element Category', fontweight='bold')
            axes[1,0].set_ylabel('Relative Error (%)', fontweight='bold')
            axes[1,0].set_title('Method Agreement by Element Category', fontweight='bold')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Crystal system accuracy
        crystal_systems = comparison_data['Crystal System'].value_counts()
        systems_to_plot = crystal_systems[crystal_systems >= 2].index[:8]
        
        if len(systems_to_plot) > 0:
            system_errors = []
            system_labels = []
            for system in systems_to_plot:
                system_data = comparison_data[comparison_data['Crystal System'] == system]
                if len(system_data) >= 2:
                    errors = (system_data['Corrected Volume Radius (Å)'] - system_data['Traditional Radius (Å)']).abs()
                    system_errors.append(errors.mean())
                    system_labels.append(system)
            
            if system_errors:
                axes[1,1].bar(range(len(system_errors)), system_errors, 
                             color=sns.color_palette("viridis", len(system_errors)))
                axes[1,1].set_xticks(range(len(system_labels)))
                axes[1,1].set_xticklabels(system_labels, rotation=45, ha='right')
                axes[1,1].set_ylabel('Mean Absolute Error (Å)', fontweight='bold')
                axes[1,1].set_title('Method Agreement by Crystal System', fontweight='bold')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   🎨 Method comparison plot saved to: {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, output_dir: str) -> None:
        """
        Generate all visualizations and save to specified directory.
        
        Args:
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📈 Generating comprehensive visualizations...")
        
        # Generate all plots
        self.plot_periodic_trends(str(output_path / 'periodic_trends.png'))
        self.plot_element_families(str(output_path / 'element_families.png'))
        self.plot_calculation_methods(str(output_path / 'calculation_methods.png'))
        
        # Only generate comparison if we have both types of data
        comparison_data = self.df.dropna(subset=['Traditional Radius (Å)', 'Corrected Volume Radius (Å)'])
        if not comparison_data.empty:
            self.plot_comparison_analysis(str(output_path / 'method_comparison.png'))
        
        print(f"✨ All visualizations saved to: {output_path}")
    
    def show_all_plots(self) -> None:
        """Display all plots in interactive mode."""
        print("🎨 Generating interactive plots...")
        
        self.plot_periodic_trends()
        self.plot_element_families()
        self.plot_calculation_methods()
        
        # Only show comparison if data is available
        comparison_data = self.df.dropna(subset=['Traditional Radius (Å)', 'Corrected Volume Radius (Å)'])
        if not comparison_data.empty:
            self.plot_comparison_analysis()
        
        plt.show()
    
    def get_summary_statistics(self) -> Dict:
        """Get comprehensive summary statistics for visualization."""
        stats = {
            'total_elements': len(self.df),
            'elements_with_radii': len(self.df.dropna(subset=['Primary Radius (Å)'])),
            'unique_crystal_systems': self.df['Crystal System'].nunique(),
            'element_categories': self.enhanced_df['Element Category'].value_counts().to_dict(),
            'radius_range': {
                'min': self.df['Primary Radius (Å)'].min(),
                'max': self.df['Primary Radius (Å)'].max(),
                'mean': self.df['Primary Radius (Å)'].mean(),
                'std': self.df['Primary Radius (Å)'].std()
            }
        }
        
        return stats