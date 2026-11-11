"""Plots the 3D geometry based on the Table of Offsets."""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import yaml
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.cad_geometry import CadGeometry

def main():
    cad = CadGeometry()
    off = cad.offsets

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Wings
    for side in ["wing_right", "wing_left"]:
        w = off[side]
        # Leading Edge
        ax.plot([w["root_le"][0], w["tip_le"][0]], 
                [w["root_le"][1], w["tip_le"][1]], 
                [w["root_le"][2], w["tip_le"][2]], 'k-', linewidth=3)
        # Trailing Edge
        ax.plot([w["root_te"][0], w["tip_te"][0]], 
                [w["root_te"][1], w["tip_te"][1]], 
                [w["root_te"][2], w["tip_te"][2]], 'k-', linewidth=3)
        # Tip and Root chords
        ax.plot([w["tip_le"][0], w["tip_te"][0]], [w["tip_le"][1], w["tip_te"][1]], [w["tip_le"][2], w["tip_te"][2]], 'k-', linewidth=3)
        ax.plot([w["root_le"][0], w["root_te"][0]], [w["root_le"][1], w["root_te"][1]], [w["root_le"][2], w["root_te"][2]], 'k-', linewidth=3)

    # 2. Plot V-Tail
    vt = off["vtail"]
    # Root chord
    ax.plot([vt["root_le"][0], vt["root_te"][0]], [vt["root_le"][1], vt["root_te"][1]], [vt["root_le"][2], vt["root_te"][2]], 'purple', linewidth=3)
    # Left surface
    ax.plot([vt["root_le"][0], vt["tip_left_le"][0]], [vt["root_le"][1], vt["tip_left_le"][1]], [vt["root_le"][2], vt["tip_left_le"][2]], 'purple', linewidth=3)
    # Right surface
    ax.plot([vt["root_le"][0], vt["tip_right_le"][0]], [vt["root_le"][1], vt["tip_right_le"][1]], [vt["root_le"][2], vt["tip_right_le"][2]], 'purple', linewidth=3)

    # 3. Plot Fuselage
    f = off["fuselage"]
    ax.plot([f["nose_tip"][0], f["tail_tip"][0]], [f["nose_tip"][1], f["tail_tip"][1]], [f["nose_tip"][2], f["tail_tip"][2]], 'gray', linewidth=6, alpha=0.5)

    # 4. Plot Motors
    for m in off["motors"]:
        color = 'blue' if m["tilts"] else 'red'
        ax.scatter(m["pos"][0], m["pos"][1], m["pos"][2], c=color, s=100, marker='^')
        ax.text(m["pos"][0], m["pos"][1], m["pos"][2], f" M{m['id']}", color=color, fontsize=9)

    # 5. Plot CG
    ax.scatter(0, 0, 0, c='green', s=200, marker='o', label="CG")

    # Formatting
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Right)')
    ax.set_zlabel('Z (Down)')
    ax.set_title('Blue UAM: Table of Offsets (3D CAD Skeleton)')
    
    # Invert Z so UP is UP on screen (NED convention)
    ax.invert_zaxis()
    
    # Set limits
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 0.5])

    plt.tight_layout()
    plt.savefig("cad_offsets_3d.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    main()
