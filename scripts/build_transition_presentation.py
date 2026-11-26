"""Build a comprehensive presentation of the tiltrotor transition model."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "Comprehensive_Tiltrotor_Project_Presentation.pptx"

NAVY = RGBColor(16, 31, 55)
BLUE = RGBColor(35, 108, 190)
TEAL = RGBColor(20, 153, 160)
PALE = RGBColor(239, 246, 251)
INK = RGBColor(38, 49, 63)
MUTED = RGBColor(91, 107, 124)
WHITE = RGBColor(255, 255, 255)
ORANGE = RGBColor(226, 130, 40)
RED = RGBColor(183, 56, 56)
GREEN = RGBColor(54, 133, 87)

SLIDE_W = 13.333
SLIDE_H = 7.5


def set_background(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_box(slide, x, y, w, h, fill=PALE, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=18,
    color=INK,
    bold=False,
    font="Aptos",
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.08,
):
    shape = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraphs = text.split("\n")
    for index, content in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(5)
        run = paragraph.add_run()
        run.text = content
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def add_formula(slide, x, y, w, h, text, size=17):
    add_box(slide, x, y, w, h, fill=PALE, line=RGBColor(209, 224, 237), radius=True)
    add_text(
        slide,
        x + 0.14,
        y + 0.08,
        w - 0.28,
        h - 0.16,
        text,
        size=size,
        color=NAVY,
        font="Cambria Math",
        valign=MSO_ANCHOR.MIDDLE,
    )


def add_header(slide, title, section="MODEL-BASED TRANSITION"):
    add_text(slide, 0.62, 0.18, 8.7, 0.22, section.upper(), size=9, color=TEAL, bold=True)
    add_text(slide, 0.62, 0.42, 12.0, 0.55, title, size=27, color=NAVY, bold=True)
    add_box(slide, 0.62, 1.02, 12.05, 0.035, fill=TEAL)


def add_footer(slide, page):
    add_text(
        slide,
        0.62,
        7.16,
        11.4,
        0.18,
        "Tiltrotor transition study  |  Engineering model, not flight certification",
        size=8,
        color=MUTED,
    )
    add_text(
        slide,
        12.15,
        7.14,
        0.52,
        0.2,
        f"{page:02d}",
        size=9,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )


def content_slide(prs, title, section="MODEL-BASED TRANSITION"):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    add_header(slide, title, section)
    add_footer(slide, len(prs.slides))
    return slide


def add_image_fit(slide, path, x, y, w, h):
    with Image.open(path) as image:
        image_w, image_h = image.size
    scale = min(w / image_w, h / image_h)
    width = Inches(image_w * scale)
    height = Inches(image_h * scale)
    left = Inches(x + (w - image_w * scale) / 2.0)
    top = Inches(y + (h - image_h * scale) / 2.0)
    return slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def add_graph_slide(prs, title, image_name, note, section="OPTIMIZATION RESULTS"):
    slide = content_slide(prs, title, section)
    image_path = ROOT / image_name
    if not image_path.is_file():
        image_path = FIGURES / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Required graph not found: {image_path}")
    add_image_fit(slide, image_path, 0.72, 1.12, 11.9, 5.55)
    add_text(slide, 0.82, 6.72, 11.7, 0.3, note, size=10, color=MUTED)
    return slide


def add_two_column(
    slide,
    left_title,
    left_text,
    right_title,
    right_text,
    y=1.35,
    h=5.45,
):
    add_box(slide, 0.72, y, 5.82, h, fill=PALE, radius=True)
    add_box(slide, 6.78, y, 5.82, h, fill=PALE, radius=True)
    add_text(slide, 0.95, y + 0.23, 5.35, 0.42, left_title, size=18, color=BLUE, bold=True)
    add_text(slide, 1.0, y + 0.73, 5.22, h - 0.95, left_text, size=15, color=INK)
    add_text(slide, 7.02, y + 0.23, 5.3, 0.42, right_title, size=18, color=BLUE, bold=True)
    add_text(slide, 7.08, y + 0.73, 5.22, h - 0.95, right_text, size=15, color=INK)


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    prs.core_properties.title = "Comprehensive Tiltrotor Transition Project Presentation"
    prs.core_properties.subject = "Project dynamics, control, corridors, gusts, tracking, and Monte Carlo results"
    prs.core_properties.author = "Transition Modeling Project"
    prs.core_properties.keywords = "tiltrotor, transition, corridor, optimization, forces"

    # 1. Cover
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide, NAVY)
    add_box(slide, 0.0, 0.0, 0.16, SLIDE_H, fill=TEAL)
    add_text(slide, 0.86, 1.0, 10.9, 0.35, "MODEL-BASED FLIGHT TRANSITION", size=12, color=TEAL, bold=True)
    add_text(
        slide,
        0.86,
        1.7,
        11.4,
        1.45,
        "Tiltrotor Transition\nProject Review",
        size=34,
        color=WHITE,
        bold=True,
    )
    add_text(
        slide,
        0.9,
        3.55,
        10.8,
        0.9,
        "Dynamics • control • transition corridors • tracking • gusts • Monte Carlo • optimization",
        size=18,
        color=RGBColor(205, 221, 237),
    )
    add_box(slide, 0.9, 5.35, 5.6, 0.04, fill=TEAL)
    add_text(slide, 0.9, 5.62, 8.4, 0.4, "Comprehensive engineering presentation", size=13, color=WHITE)
    add_text(slide, 0.9, 6.94, 10.5, 0.24, "Results are simulation outputs; they are not flight-test validation.", size=10, color=RGBColor(190, 202, 217))

    # 2. Executive summary
    slide = content_slide(prs, "What this presentation covers", "OVERVIEW")
    add_two_column(
        slide,
        "Model and optimization",
        "• A 6-DOF Newton–Euler simulator combines rotor, wing, and V-tail forces.\n\n"
        "• A cascade PID tracks altitude/speed references from a corridor CSV.\n\n"
        "• Dryden-like turbulence, gust sweeps, Monte Carlo controller comparisons, and corridor optimizers provide offline analysis.\n\n"
        "• A separate CasADi/IPOPT point-mass model optimizes a 15 s transition.",
        "Headline result",
        "• Project rotor angle: 90° upward in hover and 0° forward in cruise.\n\n"
        "• The legacy reduced model targets 12 m/s and 100 m altitude. The corrected pitch-trim analysis rejects that endpoint under the configured travel limit.\n\n"
        "• Existing full-simulation plots also show substantial tracking error and gust response.\n\n"
        "• Monte Carlo evidence varies by script and run configuration.\n\n"
        "• No output in this project establishes flight readiness.",
    )

    # 3. Sign conventions
    slide = content_slide(prs, "Frames, signs, and tilt convention", "MODEL DEFINITION")
    add_formula(
        slide,
        0.8,
        1.35,
        11.7,
        1.0,
        "Project rotor angle:  90° (rotors vertical / hover)  →  0° (rotors forward / cruise)\n"
        "Body: Forward–Right–Down (FRD)     Earth: North–East–Down (NED)",
        size=18,
    )
    add_text(
        slide,
        0.95,
        2.7,
        5.6,
        3.8,
        "The project and the supplied sketch measure the rotor angle above the forward axis. The equations below use the internal rotor deflection β:\n\n"
        "β = 90° − project rotor angle\n\n"
        "β increases from 0° in hover to 90° with the rotors forward.",
        size=17,
    )
    add_image_fit(slide, FIGURES / "tilt_convention_conversion.png", 6.7, 2.15, 5.7, 4.25)

    # 4. Tilt schedule
    slide = content_slide(prs, "Smooth cosine tilt schedule", "REFERENCE PROFILE")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.15,
        "β(t) = 0,                         t < t_hold\n"
        "β(t) = 45° [1 − cos(π(t − t_hold)/t_d)],  t_hold ≤ t < t_hold + t_d\n"
        "β(t) = 90°,                       t ≥ t_hold + t_d",
        size=17,
    )
    add_text(
        slide,
        0.92,
        2.62,
        5.1,
        3.5,
        "The cosine schedule has zero tilt rate at its start and finish. Its peak rate is:\n\n"
        "max |β̇| = 45π/t_d  deg/s\n\n"
        "These equations use model deflection β. The plotted rotor angle equals 90° − β. The example uses a 2 s hold and 10 s transition.",
        size=16,
    )
    add_image_fit(slide, FIGURES / "tilt_schedule_comparison.png", 6.1, 2.25, 6.3, 4.25)

    # 5. Rotor forces
    slide = content_slide(prs, "Rotor thrust decomposition", "FORCE MODEL")
    add_formula(
        slide,
        0.82,
        1.35,
        11.65,
        1.05,
        "T_i = k_T ω_i²\n"
        "F_rotor,body = [ T_t sin β,  0,  −T_t cos β − T_f ]ᵀ",
        size=20,
    )
    add_two_column(
        slide,
        "Meaning",
        "T_t is the combined thrust from the two tilting rotors. T_f is the combined thrust from the two fixed rotors.\n\n"
        "At β = 0°, tilting-rotor thrust acts upward. At β = 90°, it acts forward. Fixed-rotor thrust remains vertical.",
        "Optimization parameterization",
        "Each pair has a total thrust bound of 0–40 N (equivalent to 0–20 N per rotor for two identical rotors).\n\n"
        "The optimized decision variables are pair thrusts, not motor angular rates. The underlying rotor law is quadratic, T = k_Tω².",
        y=2.62,
        h=4.18,
    )

    # 6. Aerodynamics
    slide = content_slide(prs, "Lift, drag, and dynamic pressure", "FORCE MODEL")
    add_formula(
        slide,
        0.82,
        1.32,
        11.65,
        1.42,
        "q̄ = ½ρV²     ;     L = q̄ S C_L     ;     D = q̄ S C_D\n"
        "C_L = C_L0 = 0.1     ;     C_D = C_D0 + C_L²/(π e AR)\n"
        "F_aero,body = [−D, 0, −L]ᵀ",
        size=18,
    )
    add_text(
        slide,
        0.95,
        3.05,
        11.3,
        2.9,
        "For the optimized curve set: ρ = 1.225 kg/m³, S = 1.5 m², C_D0 = 0.02, e = 0.8, AR = 6. The optimizer holds C_L at C_L0, corresponding to a zero-angle-of-attack approximation in this reduced model.\n\n"
        "Rotor wash is deliberately excluded from the optimizer so the force equations align with the supplied sketch. The project’s separate 6-DOF aerodynamic model includes an induced-velocity / rotor-wash approximation and a stall-limited lift curve.",
        size=16,
    )

    # 7. Full aerodynamic model
    slide = content_slide(prs, "Additional aerodynamic model used by the 6-DOF simulator", "6-DOF MODEL CONTEXT")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.42,
        "T_per = T_total/N_r     ;     v_i = √[T_per/(2ρA_r)]\n"
        "u_eff = u + v_i sin β     ;     w_eff = w + v_i cos β\n"
        "V_eff = √(u_eff² + v² + w_eff²)     ;     α_eff = atan2(w_eff, u_eff)",
        size=17,
    )
    add_formula(
        slide,
        0.82,
        2.95,
        11.65,
        1.55,
        "C_L = C_L0 + C_Lα α_eff,                |α_eff| < α_stall\n"
        "C_L = [C_L0 + C_Lα sgn(α_eff)α_stall] cos²(α_eff − sgn(α_eff)α_stall), otherwise\n"
        "C_D = C_D0 + C_L²/(π·0.8·AR)     ;     q̄ = ½ρV_eff²",
        size=16,
    )
    add_formula(
        slide,
        0.82,
        4.72,
        11.65,
        0.83,
        "M_aero = r_ac × F_aero + [0, q̄S c(C_m0 + C_mαα_eff), 0]ᵀ",
        size=16,
    )
    add_text(
        slide,
        0.98,
        5.72,
        11.1,
        1.0,
        "The 6-DOF AeroModel adds rotor-wash velocity before evaluating lift and drag, then computes wing pitching moment from aerodynamic-center offset plus C_m0 + C_mαα_eff. These effects are not used in the reduced optimizer’s lift/drag curves.",
        size=15,
    )

    # 9. Coordinate transform
    slide = content_slide(prs, "Body forces to NED acceleration", "FORCE MODEL")
    add_formula(
        slide,
        0.82,
        1.32,
        11.65,
        1.35,
        "γ = atan2(v_z, √(v_x² + 10⁻⁸))     ;     θ = γ\n"
        "F_x,N = cos θ F_x,b − sin θ F_z,b\n"
        "F_z,N = sin θ F_x,b + cos θ F_z,b",
        size=17,
    )
    add_formula(
        slide,
        0.82,
        2.92,
        11.65,
        1.15,
        "a_x = F_x,N / m     ;     a_up = −F_z,N / m − g",
        size=19,
    )
    add_text(
        slide,
        0.98,
        4.35,
        11.2,
        1.85,
        "The reduced optimizer sets pitch equal to flight-path angle, equivalent to a zero-AoA assumption. NED z is positive down, so upward acceleration is the negative of down-axis acceleration minus gravity. The repository’s full 6-DOF model uses the complete 3-2-1 Euler rotation matrix rather than this pitch-only rotation.",
        size=16,
    )

    # 8. Full Euler rotation
    slide = content_slide(prs, "Full 3-2-1 attitude transformation", "6-DOF MODEL CONTEXT")
    add_formula(
        slide,
        0.82,
        1.38,
        11.65,
        2.15,
        "R_b→N = [  cθcψ,  sφsθcψ − cφsψ,  cφsθcψ + sφsψ ]\n"
        "        [  cθsψ,  sφsθsψ + cφcψ,  cφsθsψ − sφcψ ]\n"
        "        [  −sθ,              sφcθ,              cφcθ ]\n\n"
        "F_N = R_b→N F_b     ;     a_N = F_N/m + [0, 0, g]ᵀ",
        size=16,
    )
    add_text(
        slide,
        0.98,
        3.85,
        11.1,
        2.2,
        "Here c(·) and s(·) mean cosine and sine. This is the matrix implemented in src/dynamics/frames.py. The image’s x/y/z acceleration equations are scalar expansions of this matrix multiplication. The optimized 2D model sets roll = yaw = 0 and uses pitch θ = γ.",
        size=16,
    )

    # 10. Euler kinematics
    slide = content_slide(prs, "Rigid-body attitude kinematics in the 6-DOF model", "6-DOF MODEL CONTEXT")
    add_formula(
        slide,
        0.82,
        1.42,
        11.65,
        1.85,
        "[φ̇, θ̇, ψ̇]ᵀ = E(φ, θ) [p, q, r]ᵀ\n"
        "E(φ,θ) = [ 1,  sinφ tanθ,  cosφ tanθ ;\n"
        "              0,       cosφ,       −sinφ ;\n"
        "              0,  sinφ secθ,  cosφ secθ ]",
        size=18,
    )
    add_text(
        slide,
        0.98,
        3.65,
        11.1,
        1.8,
        "The attitude rates are obtained from body angular rates (p, q, r) using 3-2-1 Euler kinematics. The full simulator advances rigid-body rotation and changing inertia; the reduced trajectory optimizer does not optimize attitude or angular rates.",
        size=16,
    )

    # 11. State equations and discretization
    slide = content_slide(prs, "Longitudinal state equations and discretization", "OPTIMIZATION MODEL")
    add_formula(
        slide,
        0.82,
        1.3,
        11.65,
        1.2,
        "v̇_x = a_x     ;     v̇_z = a_up     ;     ḣ = v_z     ;     β̇ = u_β",
        size=20,
    )
    add_formula(
        slide,
        0.82,
        2.78,
        11.65,
        1.65,
        "Δt = 15 s / 60 = 0.25 s\n"
        "v_x[k+1] = v_x[k] + a_x[k] Δt\n"
        "v_z[k+1] = v_z[k] + a_up[k] Δt\n"
        "h[k+1] = h[k] + ½(v_z[k] + v_z[k+1])Δt\n"
        "β[k+1] = β[k] + u_β[k]Δt",
        size=17,
    )
    add_text(
        slide,
        0.98,
        4.72,
        11.2,
        1.45,
        "The velocity and tilt states use forward Euler integration; altitude uses a trapezoidal update with adjacent vertical-velocity samples. The optimizer has 61 state nodes and 60 control intervals.",
        size=16,
    )

    # 12. Optimization variables and constraints
    slide = content_slide(prs, "Optimization variables and constraints", "OPTIMIZATION MODEL")
    add_two_column(
        slide,
        "Decision variables",
        "States at k = 0…60:\n"
        "• forward velocity v_x[k]\n"
        "• vertical velocity v_z[k]\n"
        "• altitude h[k]\n"
        "• tilt β[k]\n\n"
        "Controls:\n"
        "• tilt rate u_β[k]\n"
        "• tilting-pair thrust T_t[k]\n"
        "• fixed-pair thrust T_f[k]",
        "Boundary and path constraints",
        "Start: v_x = v_z = 0, h = 100 m, β = 0°.\n\n"
        "Finish: v_x = 12 m/s, v_z = 0, h = 100 m, β = 90°; terminal a_x = a_up = 0.\n\n"
        "Bounds: h = 90…110 m; v_x = 0…12 m/s and nondecreasing; v_z = ±4 m/s; |a_x| ≤ 3 and |a_up| ≤ 2 m/s²; |u_β| ≤ 0.35 rad/s; each pair thrust = 0…40 N.\n\n"
        "Tilt-rate start and finish are set to zero.",
    )

    # 13. Objective
    slide = content_slide(prs, "Optimization objective", "OPTIMIZATION MODEL")
    add_formula(
        slide,
        0.82,
        1.25,
        11.65,
        1.15,
        "min J = 10 L_max + 0.1 J_T + 0.5 J_h + 1.0 J_v + 0.1 J_rate + 0.1 J_rate-smooth + 0.2 J_thrust-smooth",
        size=15,
    )
    add_formula(
        slide,
        0.82,
        2.62,
        11.65,
        2.25,
        "L_max ≥ 0     ;     L_max ≥ h₀ − h[k]\n"
        "J_T = Δt Σ[(T_t/40)² + (T_f/40)²]\n"
        "J_h = Δt Σ[(h − 100)/10]²\n"
        "J_v = Δt Σ[(v_x − v_ref)/12]²     ;     v_ref[k] = 12 t[k]/15\n"
        "J_rate = Δt Σ[u_β/0.35]²\n"
        "J_rate-smooth = Σ[Δu_β/0.35]²\n"
        "J_thrust-smooth = Σ[(ΔT_t/40)² + (ΔT_f/40)²]",
        size=16,
    )
    add_text(
        slide,
        0.98,
        5.15,
        11.0,
        1.0,
        "The objective trades maximum altitude loss against normalized thrust effort, altitude tracking, speed-reference tracking, tilt-rate effort, and control smoothness. Coefficients are chosen for this engineering study; they are not tuned from flight data.",
        size=15,
    )

    # 14. Optimized trajectory
    add_graph_slide(
        prs,
        "Optimized tilt and flight-state curves",
        "optimized_longitudinal_transition.png",
        "CasADi/IPOPT optimum over 15 s, shown against linear and cosine reference schedules.",
    )

    # 15. Optimized force outputs
    add_graph_slide(
        prs,
        "Forces and accelerations along the optimized trajectory",
        "optimized_transition_force_curves.png",
        "Pair thrusts, lift/drag, body-axis resultant forces, and NED-derived forward/upward acceleration.",
    )

    # 16. Force sweep graph
    add_graph_slide(
        prs,
        "Unoptimized force sweep: tilt and airspeed effects",
        "transition_force_curves.png",
        "Fixed hover RPM snapshots at 0–20 m/s; included for physical intuition, not as the optimizer’s solution.",
        section="MODEL CROSS-CHECK",
    )

    # 17. Operating maps
    add_graph_slide(
        prs,
        "Unoptimized operating maps",
        "transition_force_model_maps.png",
        "Fixed-hover-RPM maps of forward/upward acceleration and pitch moment; these are not feasible-corridor boundaries.",
        section="MODEL CROSS-CHECK",
    )

    # 18. Full 6DOF model context
    slide = content_slide(prs, "How the reduced optimizer relates to the 6-DOF simulator", "MODEL SCOPE")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.68,
        "ṗ_N = v_N     ;     v̇_N = R_b→N F_b/m + g_N\n"
        "I ω̇ = M − ω × (Iω) − Īω\n"
        "M_rotor = Σ(r_i × F_i)\n"
        "F_total,b = ΣF_rotor,b + F_aero,b + F_tail,b",
        size=17,
    )
    add_two_column(
        slide,
        "Repository 6-DOF model",
        "Implements 3D body-to-NED force rotation, rotor force and moment sums, aerodynamic lift/drag, tail forces, attitude kinematics, and rigid-body angular dynamics.\n\n"
        "Rotor wash affects the aerodynamic model in this simulator.",
        "This presentation’s optimized run",
        "Uses a separate 2D longitudinal point-mass model with fixed C_L0, zero wind, and no propwash or pitch-moment balance.\n\n"
        "Therefore, its optimized curves are not a 6-DOF optimal-control result and do not inherit all constraints from the simulator corridor filters.",
        y=3.18,
        h=3.62,
    )

    # 19. Results
    slide = content_slide(prs, "Optimized run: results at a glance", "RESULTS")
    metrics = [
        ("Horizon", "15.0 s  |  60 intervals"),
        ("Legacy target", "0° rotor angle  |  12 m/s forward  |  100 m altitude"),
        ("Altitude range", "100.00–100.44 m  |  0 m loss"),
        ("Peak tilt rate", "13.78°/s  |  limit 20.05°/s"),
        ("Pair-thrust range", "Tilting: 2.73–24.98 N  |  Fixed: 23.99–35.82 N"),
        ("Solver", "CasADi Opti + IPOPT  |  convergence verified"),
    ]
    for index, (label, value) in enumerate(metrics):
        row = index // 2
        col = index % 2
        x = 0.82 + col * 6.0
        y = 1.42 + row * 1.55
        add_box(slide, x, y, 5.65, 1.18, fill=PALE, line=RGBColor(215, 228, 239), radius=True)
        add_text(slide, x + 0.18, y + 0.14, 5.28, 0.26, label.upper(), size=10, color=TEAL, bold=True)
        add_text(slide, x + 0.18, y + 0.49, 5.26, 0.52, value, size=17, color=NAVY, bold=True)
    add_text(
        slide,
        0.95,
        6.15,
        11.35,
        0.48,
        "The optimized path reaches the specified endpoint while respecting the reduced model’s explicit bounds. The small altitude rise is allowed by the objective; only altitude loss is explicitly minimized.",
        size=13,
        color=MUTED,
    )

    # 20. Master simulation architecture
    slide = content_slide(prs, "Master simulation: end-to-end loop and configuration", "PROJECT SIMULATION")
    add_two_column(
        slide,
        "Per-step workflow",
        "1. Read the tilt/altitude corridor from CSV.\n"
        "2. Scale the forward-speed reference with tilt progress.\n"
        "3. Compute altitude, speed, and pitch control commands.\n"
        "4. Generate seeded, colored wind perturbations.\n"
        "5. Advance the 6-DOF rigid-body model and log the state.",
        "Saved run configuration",
        "20 s horizon; Δt = 0.01 s; mass = 5 kg; wing area = 1.5 m²; rotor coefficient k_T = 10⁻⁵.\n\n"
        "Dryden-like gust standard deviations: (3.5, 3.5, 2.1) m/s; reference airspeed = 12 m/s; random seed = 99.\n\n"
        "The master script temporarily perturbs velocity with wind around the dynamics step. This is a simplified disturbance injection, not a fully coupled air-relative wind model.",
        y=1.48,
        h=4.95,
    )

    # 21. Cascade PID
    slide = content_slide(prs, "Master controller: cascade PID and thrust mapping", "CONTROL")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.32,
        "e = r − y     ;     u = clip(Kₚe + KᵢI − K_d Δy/Δt, −u_max, u_max)\n"
        "I[k+1] = clip(I[k] + eΔt, I_min, I_max)     (integrator bounds keep output feasible)",
        size=17,
    )
    add_formula(
        slide,
        0.82,
        2.83,
        11.65,
        1.48,
        "θ_des = clip(PID_h(h_ref − h) − PID_V(V_ref − V), −0.35, 0.35) rad\n"
        "δ_v = −PID_θ(θ_des − θ)     ;     T_cmd = mg + 0.5(h_ref − h)\n"
        "ω_cmd = clip(√[max(T_cmd, 0)/k_T], 0, 3000)",
        size=17,
    )
    add_text(
        slide,
        0.98,
        4.65,
        11.2,
        1.2,
        "The three PID loops use derivative-on-measurement and bounded integrators. The master run schedules altitude, speed, and pitch; the separate INDI and INDI+DOB controllers appear in Monte Carlo comparisons, not as the master-run controller.",
        size=15,
    )

    # 22. Corridor reference and tracking metrics
    slide = content_slide(prs, "Reference corridor and tracking metrics", "CONTROL")
    add_formula(
        slide,
        0.82,
        1.3,
        11.65,
        1.28,
        "β_ref(t) = interp(t; t_i, β_i)     ;     h_ref(t) = interp(t; t_i, h_i)\n"
        "p(t) = clip(β_ref/β_max, 0, 1)     ;     V_ref(t) = p(t)V_max",
        size=18,
    )
    add_formula(
        slide,
        0.82,
        2.85,
        11.65,
        1.45,
        "e_h[k] = h[k] − h_ref[k]     ;     RMSE_h = √(Σe_h²/N)\n"
        "e_V[k] = V[k] − V_ref[k]     ;     loss_max = h₀ − min_k h[k]\n"
        "pitch_RMS = √(Σθ[k]²/N)",
        size=18,
    )
    add_text(
        slide,
        0.98,
        4.65,
        11.1,
        1.25,
        "References are linearly interpolated from the corridor file. These metrics support offline comparisons; meeting one metric alone does not establish a feasible or safe transition.",
        size=15,
    )

    # 23. Master simulation graph
    add_graph_slide(
        prs,
        "Master simulation: altitude, pitch, and gust history",
        "master_simulation_final.png",
        "Saved 20 s master run. The plotted altitude falls below the 90 m reference floor and pitch excursions exceed ±10°; this is a stress-test result, not a passing corridor.",
        section="PROJECT SIMULATION",
    )

    # 24. Altitude tracking graph
    add_graph_slide(
        prs,
        "Altitude and speed tracking verification",
        "tracking_verification.png",
        "Saved comparison of optimized-reference and constant-rate profiles. The plotted response reaches roughly 87 m and speed near 50 m/s while its reference ends at 90 m and 12 m/s; tracking requires revalidation.",
        section="TRACKING",
    )

    # 25. Gust models
    slide = content_slide(prs, "Wind disturbance models used in the studies", "GUST MODELING")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.82,
        "T_corr = τ · 15/max(V, 0.1)     ;     a = exp(−Δt/T_corr)\n"
        "w[k] = a w[k−1] + η[k]     ;     η ~ N(0, σ²(1 − a²))",
        size=19,
    )
    add_formula(
        slide,
        0.82,
        3.38,
        11.65,
        1.1,
        "g(t) = ½[1 − cos(2π(t − t₀)/D)]g_peak,  t₀ ≤ t ≤ t₀ + D; otherwise 0\n"
        "Wind vector: w = [u_g, v_g, w_g]ᵀ",
        size=17,
    )
    add_text(
        slide,
        0.98,
        4.75,
        11.15,
        1.35,
        "The stochastic implementation is a seeded first-order colored-noise (Gauss–Markov) approximation labeled Dryden-like. It is not a full multi-axis Dryden spectral filter or a validated atmospheric model. σ is set per axis; the Monte Carlo scripts vary gust intensity separately.",
        size=15,
    )

    # 26. Gust graph
    add_graph_slide(
        prs,
        "Gust-model examples and frequency content",
        "gust_models.png",
        "Constant-wind and seeded colored-noise examples. The displayed PSD is an exploratory diagnostic, not proof that the implementation matches a canonical Dryden spectrum.",
        section="GUST MODELING",
    )

    # 27. Monte Carlo methodology
    slide = content_slide(prs, "Monte Carlo gust study: protocol and statistics", "MONTE CARLO")
    add_two_column(
        slide,
        "Current two-controller script",
        "scripts/monte_carlo_gusts.py runs 200 paired PID/INDI trials with gust σ sampled uniformly from 0.5–5.0 m/s and RNG seed 2024.\n\n"
        "Each trial records altitude loss, pitch RMS, V-tail saturation events, and crash outcome. Paired controllers share the sampled gust condition.",
        "Statistics reported",
        "Crash rate = N_crash/N\n"
        "Mean loss = (1/N)Σ loss_i\n"
        "P95 loss = quantile(loss, 0.95)\n"
        "Controller comparison: two-sample t-test on valid trials; p < 0.05 is treated as a difference, not as proof of safety or causation.\n\n"
        "Crash handling and excluded sentinel values must be checked before interpreting the distributions.",
        y=1.48,
        h=4.95,
    )

    # 28. Monte Carlo graph
    add_graph_slide(
        prs,
        "Monte Carlo distributions: PID versus INDI",
        "monte_carlo_gusts.png",
        "Existing histogram/scatter output. The plots contain outliers and controller saturation; do not infer a winner from a single mean or from this PNG without reproducing the exact run.",
        section="MONTE CARLO",
    )

    # 29. Statistical comparison graph
    add_graph_slide(
        prs,
        "Monte Carlo tails, gust bins, and control effort",
        "statistical_comparison.png",
        "Saved analysis includes gust-bin means, altitude-loss CDF, loss-versus-saturation Pareto points, and pitch distributions. Tail behavior and crash counts matter alongside central tendency.",
        section="MONTE CARLO",
    )

    # 30. Reported three-controller results
    slide = content_slide(prs, "Historical three-controller comparison (reported results)", "MONTE CARLO")
    add_text(
        slide,
        0.9,
        1.28,
        11.5,
        0.65,
        "PROJECT_REPORT.md summarizes a separate 100-trial-per-controller experiment with gust σ = 0.5–5.5 m/s. These values are not the current 200-trial PID/INDI script output.",
        size=15,
    )
    headers = ["Controller", "Crashes", "Mean altitude loss"]
    rows = [
        ["PID", "1 / 100", "11.51 m"],
        ["INDI", "23 / 100", "31.38 m"],
        ["INDI + DOB", "21 / 100", "30.89 m"],
    ]
    x0, y0 = 1.2, 2.15
    widths = [3.7, 3.1, 3.5]
    for col, header in enumerate(headers):
        add_box(slide, x0 + sum(widths[:col]), y0, widths[col] - 0.08, 0.62, fill=NAVY, line=NAVY, radius=False)
        add_text(slide, x0 + sum(widths[:col]) + 0.12, y0 + 0.12, widths[col] - 0.3, 0.36, header, size=14, color=WHITE, bold=True)
    for row_index, row in enumerate(rows):
        y = y0 + 0.72 + row_index * 0.82
        for col, value in enumerate(row):
            add_box(slide, x0 + sum(widths[:col]), y, widths[col] - 0.08, 0.68, fill=PALE, line=RGBColor(215, 228, 239), radius=False)
            add_text(slide, x0 + sum(widths[:col]) + 0.12, y + 0.12, widths[col] - 0.3, 0.4, value, size=16, color=NAVY, bold=(col == 0))
    add_text(
        slide,
        1.2,
        5.55,
        10.6,
        0.8,
        "The report also records a 12.90 m maximum altitude loss and 13.65° maximum pitch in its master simulation. These are historical report values; reproduce the scripts and seeds before using them as current performance claims.",
        size=14,
        color=MUTED,
    )

    # 31. Analytical corridor envelope
    slide = content_slide(prs, "Analytical speed/thrust envelope used for corridor checks", "CORRIDOR MODELING")
    add_formula(
        slide,
        0.82,
        1.28,
        11.65,
        1.55,
        "T_z,max(β) = T_max[n_f + n_t cosβ]     ;     T_x,max(β) = T_max n_t sinβ\n"
        "L_req = max(mg − T_z,max, 0)     ;     V_stall = √[2L_req/(ρS C_L,max)]",
        size=17,
    )
    add_formula(
        slide,
        0.82,
        3.08,
        11.65,
        1.7,
        "D(V) = A V² + B/V²     ;     A = ½ρS C_D0\n"
        "B = L_req²/[½ρSπeAR]     ;     Δ = T_x,max² − 4AB\n"
        "V_±² = [T_x,max ± √Δ]/(2A)",
        size=18,
    )
    add_text(
        slide,
        0.98,
        5.08,
        11.1,
        1.0,
        "A speed interval exists only when Δ ≥ 0; its lower bound is also constrained by stall speed. The broader corridor filters additionally check trim, static-stability signs, tilt reachability, thrust, motor slew, electrical limits, and gust reserve. This algebraic envelope is not a nonlinear 6-DOF reachable set.",
        size=14,
    )

    # 32. Robust corridor formulation
    slide = content_slide(prs, "Robust corridor prototype: simplified vertical dynamics", "CORRIDOR MODELING")
    add_formula(
        slide,
        0.82,
        1.3,
        11.65,
        1.65,
        "T_z = k_T ω_hover² cosβ     ;     V_x = V_max β/(π/2)\n"
        "L = ½ρS C_L V_x² sinβ     ;     a_z = (T_z + L − mg)/m\n"
        "h[k+1] = h[k] + v_z[k]Δt     ;     v_z[k+1] = v_z[k] + a_z[k]Δt",
        size=17,
    )
    add_formula(
        slide,
        0.82,
        3.18,
        11.65,
        1.05,
        "β[k+1] = β[k] + β̇[k]Δt     ;     J = max_k(h₀ − h[k])\n"
        "0 ≤ β ≤ π/2     ;     |β̇| ≤ 0.35 rad/s (nominal)",
        size=17,
    )
    add_text(
        slide,
        0.98,
        4.55,
        11.2,
        1.4,
        "The robust script applies 0.85 × 0.35 = 0.2975 rad/s (17.05°/s) in code. The source comment and plot legend disagree about the percentage/rate, and the plotted robust altitude floor does not match the code’s nominal altitude-bound calculation. Treat this as a prototype formulation.",
        size=14,
    )

    # 33. Robust corridor graph
    add_graph_slide(
        prs,
        "Nominal versus robust corridor prototype",
        "robust_corridor.png",
        "The stored profiles include non-monotonic tilt motion; several chart labels conflict with implementation bounds. They are diagnostic optimizer outputs, not validated safe corridors.",
        section="CORRIDOR MODELING",
    )

    # 34. Multi-corridor objectives
    slide = content_slide(prs, "Multi-corridor objectives: fast, safe, and smooth", "CORRIDOR MODELING")
    add_formula(
        slide,
        0.82,
        1.35,
        11.65,
        1.75,
        "J_fast = L_max + 0.1 Σ[(β_max − β_k)Δt]\n"
        "J_safe = L_max     ;     L_max ≥ h₀ − h[k]\n"
        "J_smooth = Σ β̇[k]²",
        size=20,
    )
    add_text(
        slide,
        0.98,
        3.48,
        11.15,
        2.0,
        "These are objective variants on the simplified altitude model, not a validated Pareto set. The implementation does not enforce monotonic tilt, and the plotted schedules show reverse tilt segments. A profile called “safe” is not certified as safe merely because altitude loss is in its objective.",
        size=16,
    )

    # 35. Multi-corridor graph
    add_graph_slide(
        prs,
        "Fast, safe, and smooth corridor profiles",
        "multi_corridor.png",
        "Stored profiles include substantial tilt reversals and altitude deviations. Use this figure to inspect prototype objective trade-offs, not to select a flight schedule.",
        section="CORRIDOR MODELING",
    )

    # 36. Sensitivity study
    slide = content_slide(prs, "Transition-duration sensitivity study", "SENSITIVITY")
    add_formula(
        slide,
        0.82,
        1.35,
        11.65,
        1.58,
        "For each tested duration T: loss_max(T) = 100 − min_t h(t)\n"
        "θ_max(T) = max_t |θ(t)|     ;     θ_RMS(T) = √[Σθ(t)²/N]",
        size=19,
    )
    add_text(
        slide,
        0.98,
        3.25,
        11.15,
        2.2,
        "The sweep compares simulated responses over several transition durations. It is a sensitivity experiment—not an optimizer—and should be interpreted only where the underlying run remains numerically and physically credible. The existing plot’s 25–30 s altitude-loss values grow to tens of thousands of metres, indicating divergence or invalid state evolution in those cases.",
        size=16,
    )

    # 37. Sensitivity graph
    add_graph_slide(
        prs,
        "Sensitivity to transition duration",
        "sensitivity_sweep.png",
        "The extreme altitude-loss values and pitch-limit exceedances reveal failed/unphysical cases at long durations; do not read this plot as a duration recommendation.",
        section="SENSITIVITY",
    )

    # 38. Validation and integration status
    slide = content_slide(prs, "Verification, generated controller code, and integration status", "PROJECT STATUS")
    add_two_column(
        slide,
        "Evidence recorded in project documentation",
        "• The final-deliverables report lists 121 passing tests.\n\n"
        "• A CasADi controller code-generation path and Python/C parity checks are present.\n\n"
        "• Simulation figures and CSV outputs support offline analysis.",
        "Not established",
        "• No aircraft hardware or flight-test validation.\n\n"
        "• PX4-related material is a mapping/export scaffold, not a qualified firmware integration.\n\n"
        "• V-tail moment arms, lateral-directional response, static margin, and gust recovery remain unverified.\n\n"
        "• This presentation build does not rerun the project test suite.",
        y=1.48,
        h=4.95,
    )

    # 39. Data lineage and interpretation
    slide = content_slide(prs, "How to read the results in this presentation", "EVIDENCE & LIMITATIONS")
    add_text(
        slide,
        0.9,
        1.35,
        11.5,
        4.95,
        "• The optimized 15 s curves are from a reduced longitudinal point-mass model; they are not optimized 6-DOF trajectories.\n\n"
        "• Master-simulation, tracking, sensitivity, robust-corridor, and multi-corridor plots are existing saved outputs. Several show bound violations, divergence, reverse tilt, or labels inconsistent with source code.\n\n"
        "• Monte Carlo artifacts use different experiment versions: the current two-controller script specifies 200 paired trials and σ = 0.5–5.0 m/s; the report’s three-controller summary uses 100 trials per controller and σ = 0.5–5.5 m/s.\n\n"
        "• Do not combine their statistics or treat historical report values as reproduced by this presentation build. Regenerate each result from its exact script, configuration, and seed before making performance claims.",
        size=16,
    )

    # 40. Limitations and next steps
    slide = content_slide(prs, "Interpretation, limitations, and next validation steps", "ENGINEERING REVIEW")
    add_two_column(
        slide,
        "What the project establishes",
        "• A working set of simulation, controller, force-model, and optimization prototypes.\n\n"
        "• A numerical optimum for one explicitly reduced 2D model and its stated constraints.\n\n"
        "• Useful evidence about tracking and robustness gaps that need investigation.",
        "Next validation steps",
        "• Reconcile script outputs, labels, and report tables; add feasibility and monotonic-tilt checks.\n\n"
        "• Correct wind coupling and stabilize/revalidate altitude and speed tracking.\n\n"
        "• Add actuator/motor/electrical and full moment constraints; calibrate with measured data.\n\n"
        "• Validate in SIL/HIL before any controlled flight-test process.",
    )

    # 41. Formula glossary and sources
    slide = content_slide(prs, "Symbols, constants, and implementation references", "APPENDIX")
    add_text(
        slide,
        0.86,
        1.33,
        5.75,
        5.35,
        "Symbols\n"
        "β — internal deflection from vertical; project rotor angle equals 90° − β\n"
        "T_t, T_f — total tilting/fixed pair thrust (N)\n"
        "v_x, v_z — forward/upward velocity (m/s)\n"
        "h — altitude (m); γ — flight-path angle\n"
        "F_x,b, F_z,b — body-axis force (N)\n"
        "F_x,N, F_z,N — NED force (N)\n"
        "L, D — lift and drag (N); q̄ — dynamic pressure\n"
        "ρ — density; S — wing area; AR — aspect ratio\n"
        "e — Oswald efficiency; g — gravity\n"
        "k_T — rotor thrust coefficient; ω — rotor speed",
        size=14,
    )
    add_text(
        slide,
        6.85,
        1.33,
        5.55,
        5.35,
        "Implementation sources\n"
        "• scripts/optimize_longitudinal_transition.py\n"
        "  Optimization equations, constraints, objective, and plots\n\n"
        "• src/dynamics/tiltrotor.py\n"
        "  6-DOF force and rigid-body dynamics\n\n"
        "• src/dynamics/frames.py\n"
        "  Body/NED coordinate transformations\n\n"
        "• src/aero/aerodynamics.py\n"
        "  Lift, drag, rotor-wash, and aerodynamic moment model\n\n"
        "• src/control/reference.py\n"
        "  Cosine tilt reference schedule",
        size=14,
    )

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build_deck())
