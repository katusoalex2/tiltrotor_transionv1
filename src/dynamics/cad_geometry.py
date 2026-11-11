"""Cad Geometry Engine: XFLR5 Properties + 3D Mesh Generation."""
import numpy as np
import yaml

class CadGeometry:
    def __init__(self, config_path="configs/geometry_offsets.yaml"):
        try:
            with open(config_path, "r") as f:
                self.offsets = yaml.safe_load(f)
        except FileNotFoundError:
            self.offsets = {}

    def compute_wing_properties(self, root_le, root_te, tip_le, tip_te):
        """Calculates MAC and AC from raw coordinates."""
        b_2 = abs(tip_le[1] - root_le[1])
        b = b_2 * 2.0
        c_r = np.linalg.norm(np.array(root_le) - np.array(root_te))
        c_t = np.linalg.norm(np.array(tip_le) - np.array(tip_te))
        
        taper = c_t / c_r
        mac = (2/3) * c_r * (1 + taper + taper**2) / (1 + taper)
        
        sweep_rad = np.arctan2((tip_le[0] - root_le[0]), b_2)
        x_mac = root_le[0] + (b_2 / 3.0) * ((1 + 2*taper) / (1 + taper)) * np.tan(sweep_rad)
        ac_x = x_mac + 0.25 * mac
        
        return {"mac": mac, "ac_x": ac_x, "span": b, "area": b * ((c_r + c_t) / 2.0)}

    def calculate_tilted_motors(self, tilt_angle_deg):
        tilt_rad = np.deg2rad(tilt_angle_deg)
        pivot_x, pivot_z = 0.1, 0.0
        tilted_motors = []
        motors = self.offsets.get('motors', [[0.3, 0.4, 0.0, True], [0.3, -0.4, 0.0, True], [-0.3, 0.4, 0.0, False], [-0.3, -0.4, 0.0, False]])
        
        for i, m in enumerate(motors):
            if len(m) == 4:
                pos_x, pos_y, pos_z, tilts = m
            else:
                pos_x, pos_y, pos_z = m; tilts = i < 2
            
            if tilts:
                rel_x = pos_x - pivot_x
                rel_z = pos_z - pivot_z
                new_rel_x = rel_x * np.cos(tilt_rad) + rel_z * np.sin(tilt_rad)
                new_rel_z = -rel_x * np.sin(tilt_rad) + rel_z * np.cos(tilt_rad)
                tilted_motors.append({"id": i+1, "pos": [pivot_x + new_rel_x, pos_y, pivot_z + new_rel_z], "thrust_dir": [np.sin(tilt_rad), 0, -np.cos(tilt_rad)], "tilts": True})
            else:
                tilted_motors.append({"id": i+1, "pos": [pos_x, pos_y, pos_z], "thrust_dir": [0, 0, -1], "tilts": False})
        return tilted_motors

    def generate_wing_mesh(self, wing_name, coords, cs_chord_pct=0.20, cs_span_pct=0.80):
        """Generates main wing and control surface 3D meshes."""
        if wing_name not in self.offsets: return np.array([]), np.array([])
        xu, yu, xl, yl = coords
        w = self.offsets[wing_name]
        
        root_le, root_te = np.array(w["root_le"]), np.array(w["root_te"])
        tip_le, tip_te = np.array(w["tip_le"]), np.array(w["tip_te"])
        
        root_chord = np.linalg.norm(root_le - root_te)
        tip_chord = np.linalg.norm(tip_le - tip_te)
        root_dir = (root_le - root_te) / root_chord if root_chord > 0 else np.array([1,0,0])
        tip_dir = (tip_le - tip_te) / tip_chord if tip_chord > 0 else np.array([1,0,0])
        
        main_verts, cs_verts = [], []
        hinge_x_idx = np.argmin(np.abs(xu - (1.0 - cs_chord_pct)))
        
        for i in range(len(xu)):
            is_cs = i <= hinge_x_idx
            p_root = root_te + xu[i] * root_dir * root_chord
            p_root[2] += yu[i] * root_chord
            p_root_l = root_te + xl[i] * root_dir * root_chord
            p_root_l[2] += yl[i] * root_chord
            
            p_tip = tip_te + xu[i] * tip_dir * tip_chord
            p_tip[2] += yu[i] * tip_chord
            p_tip_l = tip_te + xl[i] * tip_dir * tip_chord
            p_tip_l[2] += yl[i] * tip_chord
            
            if is_cs: cs_verts.extend([p_root, p_root_l, p_tip, p_tip_l])
            else: main_verts.extend([p_root, p_root_l, p_tip, p_tip_l])
                
        return np.array(main_verts), np.array(cs_verts)

    def generate_vtail_mesh(self, coords):
        """Generates V-Tail 3D meshes."""
        if "vtail" not in self.offsets: return np.array([]), np.array([])
        xu, yu, xl, yl = coords
        vt = self.offsets["vtail"]
        
        root_le, root_te = np.array(vt["root_le"]), np.array(vt["root_te"])
        tip_le_l, tip_le_r = np.array(vt["tip_left_le"]), np.array(vt["tip_right_le"])
        
        root_chord = np.linalg.norm(root_le - root_te)
        tip_chord = root_chord * 0.5
        root_dir = (root_le - root_te) / root_chord if root_chord > 0 else np.array([1,0,0])
        
        verts_left, verts_right = [], []
        
        for i in range(len(xu)):
            p_root = root_te + xu[i] * root_dir * root_chord
            p_root[2] += yu[i] * root_chord
            p_root_l = root_te + xl[i] * root_dir * root_chord
            p_root_l[2] += yl[i] * root_chord
            
            tip_te_l = tip_le_l - root_dir * tip_chord
            p_tip_l = tip_te_l + xu[i] * (tip_le_l - tip_te_l)
            p_tip_l[2] += yu[i] * tip_chord
            p_tip_ll = tip_te_l + xl[i] * (tip_le_l - tip_te_l)
            p_tip_ll[2] += yl[i] * tip_chord
            verts_left.extend([p_root, p_root_l, p_tip_l, p_tip_ll])
            
            tip_te_r = tip_le_r - root_dir * tip_chord
            p_tip_r = tip_te_r + xu[i] * (tip_le_r - tip_te_r)
            p_tip_r[2] += yu[i] * tip_chord
            p_tip_rr = tip_te_r + xl[i] * (tip_le_r - tip_te_r)
            p_tip_rr[2] += yl[i] * tip_chord
            verts_right.extend([p_root, p_root_l, p_tip_r, p_tip_rr])
            
        return np.array(verts_left), np.array(verts_right)
