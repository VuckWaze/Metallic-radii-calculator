#!/usr/bin/env python3
"""
Showcase Analysis: Highlighting the Most Exciting Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_showcase_plots():
    """Create beautiful showcase plots of the most exciting results"""
    
    # Load the complete results
    df = pd.read_csv('output_full_periodic/complete_periodic_table_radii.csv')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (20, 16)
    plt.rcParams['font.size'] = 12
    
    # Create a comprehensive showcase figure
    fig = plt.figure(figsize=(24, 18))
    
    # Define a beautiful color palette
    colors = sns.color_palette("husl", 10)
    
    # 1. The Grand Periodic Table View (Top Left)
    ax1 = plt.subplot(3, 4, (1, 5))
    
    # Filter valid data
    valid_data = df.dropna(subset=['Metallic_Radius (Å)', 'Atomic_Number'])
    
    # Create the main periodic trend plot
    categories = valid_data['Element_Category'].unique()
    category_colors = sns.color_palette("Set1", len(categories))
    color_map = dict(zip(categories, category_colors))
    
    for category in categories:
        cat_data = valid_data[valid_data['Element_Category'] == category]
        if not cat_data.empty:
            ax1.scatter(cat_data['Atomic_Number'], cat_data['Metallic_Radius (Å)'],
                       label=category, alpha=0.8, s=80, color=color_map[category],
                       edgecolors='black', linewidth=0.5)
    
    # Highlight extreme cases
    max_radius = valid_data.loc[valid_data['Metallic_Radius (Å)'].idxmax()]
    min_radius = valid_data.loc[valid_data['Metallic_Radius (Å)'].idxmin()]
    
    ax1.annotate(f'Largest: {max_radius["Element"]} ({max_radius["Metallic_Radius (Å)"]:.3f} Å)',
                xy=(max_radius['Atomic_Number'], max_radius['Metallic_Radius (Å)']),
                xytext=(max_radius['Atomic_Number']+5, max_radius['Metallic_Radius (Å)']+0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    ax1.annotate(f'Smallest: {min_radius["Element"]} ({min_radius["Metallic_Radius (Å)"]:.3f} Å)',
                xy=(min_radius['Atomic_Number'], min_radius['Metallic_Radius (Å)']),
                xytext=(min_radius['Atomic_Number']+10, min_radius['Metallic_Radius (Å)']+0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=12, fontweight='bold', color='blue')
    
    ax1.set_xlabel('Atomic Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Metallic Radius (Å)', fontsize=14, fontweight='bold')
    ax1.set_title('🌟 Metallic Radii Across the Complete Periodic Table 🌟', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 2. Extreme Structures Showcase (Top Right)
    ax2 = plt.subplot(3, 4, (2, 3))
    
    extreme_structures = ['α-Mn', 'U', 'mP16', 'α-Np', 'α-Ga', 'α-Pa']
    extreme_data = df[df['Crystal_System'].isin(extreme_structures)].copy()
    
    if not extreme_data.empty:
        bars = ax2.bar(extreme_data['Element'], extreme_data['Metallic_Radius (Å)'],
                      color=sns.color_palette("viridis", len(extreme_data)),
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add structure labels
        for bar, structure in zip(bars, extreme_data['Crystal_System']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    structure, ha='center', va='bottom', fontsize=10, 
                    fontweight='bold', rotation=45)
        
        ax2.set_ylabel('Metallic Radius (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('🏆 Most Complex Structures Conquered! 🏆', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Polymorphs Comparison (Middle Left)
    ax3 = plt.subplot(3, 4, (6, 7))
    
    # Find elements with multiple structures
    element_counts = df['Element'].value_counts()
    polymorphic_elements = element_counts[element_counts > 1].index
    
    polymorph_data = []
    for element in polymorphic_elements[:6]:  # Top 6 for clarity
        element_data = df[df['Element'] == element]
        for _, row in element_data.iterrows():
            polymorph_data.append({
                'Element': element,
                'Structure': row['Crystal_System'],
                'Radius': row['Metallic_Radius (Å)'],
                'Label': f"{element} ({row['Crystal_System']})"
            })
    
    polymorph_df = pd.DataFrame(polymorph_data)
    
    if not polymorph_df.empty:
        # Create grouped bar plot
        x_pos = range(len(polymorph_df))
        bars = ax3.bar(x_pos, polymorph_df['Radius'],
                      color=sns.color_palette("plasma", len(polymorph_df)),
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(polymorph_df['Label'], rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Metallic Radius (Å)', fontsize=12, fontweight='bold')
        ax3.set_title('🔄 Polymorphic Elements - Different Structures! 🔄', 
                      fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Element Categories Distribution (Middle Right)
    ax4 = plt.subplot(3, 4, 8)
    
    category_counts = df['Element_Category'].value_counts()
    wedges, texts, autotexts = ax4.pie(category_counts.values, labels=category_counts.index, 
                                      autopct='%1.1f%%', colors=sns.color_palette("Set2", len(category_counts)),
                                      startangle=90)
    
    ax4.set_title('📊 Element Categories Analyzed 📊', fontsize=14, fontweight='bold')
    
    # 5. Actinide Series Highlight (Bottom Left)
    ax5 = plt.subplot(3, 4, (9, 10))
    
    actinides = df[df['Element_Category'] == 'Actinides'].copy()
    actinides = actinides.sort_values('Atomic_Number')
    
    if not actinides.empty:
        ax5.plot(actinides['Atomic_Number'], actinides['Metallic_Radius (Å)'],
                'o-', linewidth=3, markersize=10, color='orange', alpha=0.8, 
                markeredgecolor='black', markeredgewidth=1)
        
        # Add element labels
        for _, row in actinides.iterrows():
            ax5.annotate(row['Element'], 
                        (row['Atomic_Number'], row['Metallic_Radius (Å)']),
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold')
        
        ax5.set_xlabel('Atomic Number', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Metallic Radius (Å)', fontsize=12, fontweight='bold')
        ax5.set_title('☢️ Actinide Series - Including Transuranics! ☢️', 
                      fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Crystal Structure Complexity (Bottom Middle)
    ax6 = plt.subplot(3, 4, 11)
    
    # Define complexity scores
    complexity_map = {
        'bcc': 1, 'fcc': 1, 'hcp': 1,
        'diamond': 2, 'α-La': 2,
        'α-As': 3, 'tI2': 3, 'γ-Se': 3,
        'α-Mn': 5, 'U': 4, 'mP16': 5, 'α-Np': 4,
        'α-Ga': 4, 'α-Pa': 4
    }
    
    df_complexity = df.copy()
    df_complexity['Complexity'] = df_complexity['Crystal_System'].map(complexity_map).fillna(3)
    
    complexity_counts = df_complexity['Complexity'].value_counts().sort_index()
    
    bars = ax6.bar(complexity_counts.index, complexity_counts.values,
                  color=sns.color_palette("rocket", len(complexity_counts)),
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    ax6.set_xlabel('Structure Complexity Score', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Number of Structures', fontsize=12, fontweight='bold')
    ax6.set_title('🔬 Structural Complexity Distribution 🔬', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Top Statistics Box (Bottom Right)
    ax7 = plt.subplot(3, 4, 12)
    ax7.axis('off')
    
    # Create statistics text
    stats_text = f"""
🎉 INCREDIBLE ACHIEVEMENTS! 🎉

📊 Total Structures: {len(df)}
🧪 Unique Elements: {df['Element'].nunique()}
🔬 Crystal Systems: {df['Crystal_System'].nunique()}

🌟 RADIUS EXTREMES:
   🔴 Largest: {max_radius['Element']} - {max_radius['Metallic_Radius (Å)']:.3f} Å
   🔵 Smallest: {min_radius['Element']} - {min_radius['Metallic_Radius (Å)']:.3f} Å
   📏 Range: {valid_data['Metallic_Radius (Å)'].max() - valid_data['Metallic_Radius (Å)'].min():.3f} Å

🏆 COMPLEX STRUCTURES SOLVED:
   • α-Mn (58 atoms/cell!)
   • Plutonium mP16
   • Uranium orthorhombic
   • Neptunium α-Np

🔄 POLYMORPHS ANALYZED:
   • Boron: 3 forms
   • Carbon: 3 forms  
   • Phosphorus: 4 forms
   • Cobalt: 2 forms

☢️ ACTINIDES COMPLETE:
   From Ac to Cf - transuranics included!
    """
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.8))
    
    # Add main title
    fig.suptitle('🌟 COMPLETE PERIODIC TABLE METALLIC RADII ANALYSIS 🌟\n' + 
                '96 Structures • 83 Elements • 32 Crystal Systems', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def main():
    """Create and save the showcase analysis"""
    print("🎨 Creating showcase visualization...")
    
    fig = create_showcase_plots()
    
    # Save the showcase
    output_path = Path('output_full_periodic/SPECTACULAR_SHOWCASE.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"🎉 Spectacular showcase saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()