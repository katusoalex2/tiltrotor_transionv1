const study = "figures/transition_study/";
const refined = study + "refined_corridor/";
const follow = refined + "followup/";
const review = refined + "transition_review/";

export const slides = [
  {kind:"cover", section:"ENGINEERING REVIEW", title:"Forward-only tiltrotor\ntransition corridor", sub:"Model verification, corridor construction, and closed-loop results", detail:"5 kg V-tail reference aircraft  /  Hover 90° to cruise 0°", source:review+"summary.json", notes:"Engineering development review of the bundled aircraft and the latest offline simulation campaign. All geometry and handling conclusions apply to the configured model. This deck records completed work and the evidence still needed for aircraft validation."},

  {section:"Overview", title:"The strongest simulated transition", intro:"Reserve-aware 12 improved disturbance-case performance among the three controllers compared.", kind:"metrics", metrics:[["32 s","Nominal transition"],["30.0 m/s","Terminal true airspeed"],["0.15 m","Peak altitude deviation"],["9 / 26","Common cases passed"]], close:"The selected replay passed all four held-out along-track gusts. A robust corridor remains unestablished.", source:review+"summary.json", notes:"Selection was a staged comparison, not a claim of mathematical global optimality. Candidate ranking first required the nominal inner screen, then maximized common cases passed, then minimized modeled nominal electrical energy. All candidates used the same scenario bank."},

  {section:"Overview", title:"Development and verification workflow", items:[
    ["Aircraft definition","Reconcile the configuration with geometry, coordinate frames, mass assumptions, and thrust vectors."],
    ["Static trim","Solve steady force and pitch equilibrium over the airspeed–angle grid."],
    ["Dynamic reachability","Optimize complete transitions and replay them under state and actuator limits."],
    ["Disturbance review","Replay each candidate against the same perturbations and log the first constraint crossed."],
    ["Validation status","Separate numerical verification from aircraft-specific aerodynamic, propulsion, and hardware validation."]
  ], source:"others/FINAL_DELIVERABLES.md", notes:"Each stage has an associated script, output tables, and an acceptance criterion. A projected two-dimensional map is a useful summary, but the time-resolved state and controls are retained in each replay."},

  {section:"Aircraft definition", title:"Geometry and mass properties", kind:"table", values:[
    ["Input","Configured value","Role in the model"],
    ["Mass and inertia","5.0 kg  /  I = diag(0.10, 0.10, 0.20) kg·m²","Weight and angular acceleration"],
    ["CG and body axes","CG at model origin  /  x forward, y right, z down","Reference for loads and moment arms"],
    ["Front hubs in hover","x = +0.30 m, y = ±0.40 m, z = 0","Two synchronized tilting motors"],
    ["Rear hubs","x = −0.30 m, y = ±0.40 m, z = 0","Two fixed upward-thrust motors"],
    ["Wing","2.0 m span, 0.40/0.20 m root/tip chord","0.60 m² area, aspect ratio 6.67"],
    ["V-tail","0.10 m² configured area, 26.6° panel inclination","Symmetric pitch input in this study"]
  ], widths:[241,466,431], close:"The airframe reference and mass properties are configured values. Measured CG and inertia remain open.", source:"UAM Configurations/aircraft.yaml", notes:"Crosschecked motor hubs against geometry_offsets.yaml and the model geometry logic. Panel inclination follows atan2(0.15, 0.30). The trapezoidal wing mean aerodynamic chord is about 0.311 m. Weight is 49.05 N with the configured gravity. No physical weighing or inertia measurement is implied."},

  {section:"Aircraft definition", title:"Rotor elevation convention", kind:"figure", image:"figures/tilt_convention_conversion.png", points:["Published elevation β is measured above the forward axis.","Hover β = 90°. Forward cruise β = 0°.","The plant uses deflection from vertical, δ = 90° − β.","Rear rotors remain fixed upward in the body frame."], source:"src/dynamics/nacelle.py  /  figures/tilt_convention_conversion.png", notes:"Use rotor elevation in slide axes, reports, and exported setpoints where explicitly labeled. Internally the thrust direction is expressed in deflection from vertical. A legacy tilt_deg column may use a different convention, so its source must be checked before reuse."},

  {section:"Aircraft definition", title:"Thrust vectors and pitch moments", kind:"formula", formulas:[
    ["Front pair in body axes","Ffront = Tf [cos β, 0, −sin β]"],
    ["Fixed rear pair in body axes","Frear = Tr [0, 0, −1]"],
    ["Pitch moment about the CG","My = (r × F)y = rz Fx − rx Fz"]
  ], close:"The front hubs move about the modeled pivot. Their moment arms are recomputed as the nacelles tilt.", source:"src/dynamics/rotor.py  /  src/dynamics/tiltrotor.py", notes:"Tf and Tr are pair thrust magnitudes. For the configured pivot at x = 0.10 m and a 0.20 m hover offset, a front hub moves from x = 0.30 m, z = 0 toward x = 0.10 m, z = +0.20 m at forward tilt. This changes pitch moment even when the forward force acts with no nominal vertical thrust component in body axes. Transform all forces before summing in the inertial frame."},

  {section:"Aircraft definition", title:"Fixed rear rotors change the available force", kind:"table", values:[
    ["Equal per-rotor limit T","Forward-only tilt","All four rotors tilt"],
    ["Maximum upward projection","2T(1 + sin β)","4T sin β"],
    ["Maximum forward projection","2T cos β","4T cos β"],
    ["At β = 0°","2T upward and 2T forward","0 upward and 4T forward"]
  ], widths:[386,383,369], close:"These level-attitude projections isolate the geometry. Corridor width also depends on trim, power, and control authority.", source:"others/FINAL_DELIVERABLES.md  /  thrust-vector derivation", notes:"This analytical comparison assumes equal maximum thrust on all motors and neglects aerodynamic loads and attitude changes. It explains why the fixed rear layout preserves vertical rotor support late in transition while reducing forward force at the same angle."},

  {section:"Aircraft model", title:"Rigid-body dynamics and air-relative motion", kind:"formula", formulas:[
    ["Translation in inertial axes","m a = R Fbody + m g"],
    ["Rotation in body axes","I ω̇ = Mbody − ω × (Iω)"],
    ["Symmetric pitch limit","Iyy q̇ = ΣMy"]
  ], close:"The planner uses five longitudinal states. The rigid-body replay advances full attitude and translational states under symmetric commands.", source:"src/dynamics/tiltrotor.py  /  src/dynamics/transition_trajectory.py", notes:"The five trajectory states are forward velocity, climb rate, height change, pitch and pitch rate. Controls are front nacelle deflection, front and rear pair motor speeds, and symmetric V-tail deflection. Aerodynamic incidence uses air-relative motion. The replay configuration and disturbances exercise a symmetric longitudinal plant subset, so the full integrator does not confer lateral validation."},

  {section:"Aircraft model", title:"Wing loads, stall, and rotor-wash blending", kind:"formula", formulas:[
    ["Lift and drag","L = q̄ S CL   /   D = q̄ S CD"],
    ["Dynamic pressure and induced drag","q̄ = ½ρV²   /   CD,i = CL² ÷ (π e AR)"],
    ["Effective incidence","Freestream motion and simplified rotor wash"]
  ], close:"The wing polar models finite-wing effects and post-stall behavior. The configured 15° stall reference is an assumption.", source:"src/aero/wing_polar.py  /  src/aero/aerodynamics.py", notes:"The configured inputs include lift slope 4.0 per radian, CL0 = 0.10, CD0 = 0.02 and Oswald efficiency = 0.80. The study changes the low-speed load onset from a sharp cutoff to a blend across 0.5–2.5 m/s. Startup propwash can yield high effective incidence below wingborne speed, so a separate low-speed flow guard is used. The low-order model does not establish aircraft stall behavior."},

  {section:"Aircraft model", title:"V-tail forces and pitch authority", items:[
    ["Geometry","Trapezoidal panels provide a geometry-derived quarter-chord moment arm about the reference CG."],
    ["Panel loads","Panel aerodynamic forces generate a vertical contribution and a pitch moment. The longitudinal replay uses equal panel commands."],
    ["Travel and slew","The model configures ±11.46° of travel. The study caps the command near ±10.46° and applies 60°/s of slew."],
    ["Authority reserve","Log both unused travel and the resulting pitch moment headroom. A degree of travel produces different authority at different airspeeds."]
  ], source:"src/configuration/aircraft_parameters.py  /  scripts/corridor_uncertainty.py", notes:"The project's tail calculation uses a shared longitudinal arm. Individual panel roll moments and asymmetric aerodynamic response remain outside this review. The study cap holds back approximately one degree of configured total travel. Reserve-aware 12 caps the feedback command another 1.6° inside that study cap."},

  {section:"Aircraft model", title:"Propulsion and electrical limits", items:[
    ["Independent pair thrust","Each rotor uses T = kT ω², with kT = 1.0 × 10⁻⁵ N/(rad/s)². Front and rear pair speeds vary independently."],
    ["RPM bounds","Study caps are about 17,731 RPM front and 16,821 RPM rear, per motor."],
    ["Electrical screen","Propeller torque and motor resistance estimate per-motor current and voltage. ESC efficiency and battery sag determine available headroom."],
    ["Rate limits","Replay enforces motor slew and tail slew. Nacelle actuation is screened against the configured 0.35 rad/s limit."],
    ["Data status","The model assumes a 6S pack, 920 RPM/V motors, and a 0.305 m propeller. Installed thrust and power still need bench measurements."]
  ], source:"UAM Configurations/aircraft.yaml  /  scripts/transition_study.py", notes:"The rear speed limit embeds the configured rear thrust reserve. Electrical results depend on approximate motor, ESC and battery models, including optimistic loaded-speed assumptions in the configuration. This is why reported energy and headroom should be treated as comparisons inside the model."},

  {section:"Model verification", title:"Hover and cruise anchor cases", kind:"table", values:[
    ["Check","Hover at 0 m/s, β = 90°","Cruise trim at 30 m/s, β = 0°"],
    ["Force balance","49.05 N upward total, 12.26 N per rotor","Wing, rotor and V-tail components close the vertical balance"],
    ["Motor speed","≈ 10,575 RPM per motor","7,253 front / 8,881 rear RPM per motor"],
    ["Pitch and tail","Opposing front and rear moments cancel","+0.03° pitch / −6.57° V-tail"],
    ["Interpretation","Analytic equal-thrust hover matches the trim test","One steady solution from the bounded grid solver"]
  ], widths:[218,394,526], close:"The 30 m/s cruise trim retains rear rotor support.", source:"tests/test_transition_dynamics.py  /  followup/static_trim_audit.csv", notes:"Hover RPM is 60/(2π) times sqrt(mg/(4 kT)), about 10,575. The 30 m/s, zero-degree trimmed row in static_trim_audit.csv gives the listed pitch, actuator deflection and separate RPM. A different dynamic trajectory can reach the same map coordinates with a different load split, since the trim problem is underdetermined."},

  {section:"Model verification", title:"Model parity and low-speed startup", items:[
    ["Independent calculations","Compare symbolic longitudinal loads with the numeric rigid-body model on matching coordinates and actuator states."],
    ["Force and moment sign checks","Verify explicit rotor and V-tail contributions against independent cross products and frame transformations."],
    ["Startup discontinuity","The prior sharp-cutoff replay ended at 1.22 s and 0.51 m/s with a pitch rate near −34°/s."],
    ["Revised load onset","A 0.5–2.5 m/s blend smooths the modeled load onset. The numeric and symbolic paths agree to approximately 2.1 × 10⁻¹² in the static audit’s component units."]
  ], source:study+"summary.json  /  "+follow+"summary.json", notes:"This is a code-to-code verification and sensitivity test. The sharp-cutoff replay and blended replay use the same aircraft reference but different low-speed aerodynamic load onset. A fixture or flight data would still be needed to validate the physical blending choice."},

  {section:"Static corridor", title:"Bounded equilibrium solve at each grid center", kind:"formula", formulas:[
    ["Inputs","True airspeed V and front rotor elevation β"],
    ["Decision variables","θ, front RPM, rear RPM, symmetric V-tail deflection"],
    ["Residuals","[aₓ/g, aᵤₚ/g, My/(mg MAC)]"]
  ], close:"The solver accepts a bounded solution when the largest absolute normalized residual is ≤ 0.005.", source:"scripts/refined_corridor.py  /  scripts/transition_study.py", notes:"SciPy bounded least squares uses an exact CasADi residual Jacobian. The starts try the nearest coarse-grid solution, the prior speed’s converged solution, a hover start, and a lower-thrust tilted start. The solver retains the lowest-residual result. An optimizer status flag alone is insufficient. Electrical and wingborne incidence are checked on the resulting candidate. Finite-rate constraints belong to trajectory replay."},

  {section:"Static corridor", title:"Static trim grid refinement", kind:"figure", image:refined+"static_trim.png", points:["Initial grid: 1 m/s × 5° across 665 points.","Refined grid: 0.5 m/s × 2° from 0 to 30 m/s and 0° to 90°.","Trim found at 2,024 of 2,806 refined centers.","All 310 shared coarse centers retained their classifications."], source:refined+"summary.json  /  static_trim.csv", notes:"The initial speed sweep extends to 34 m/s. The refined comparison window ends at 30 m/s. Their total point counts have different bounds, so compare classifiers at shared coordinates. The improved grid supports boundary inspection without establishing full numerical convergence."},

  {section:"Static corridor", title:"Trim boundaries and failed corners", kind:"table", values:[
    ["Audit item","Observed result","Meaning for the corridor"],
    ["2,806 centers assessed","782 without accepted trim","Bounded residual remained outside tolerance"],
    ["2,024 accepted trim centers","1,585 pass the wingborne screens","Checked for wingborne incidence and electrical feasibility"],
    ["Low-speed region","439 accepted trim centers below wingborne gate","Assessed under rotor-borne or mixed-flow logic"],
    ["30 m/s and β = 90°","≈ −1.36 N forward residual","The solver cannot balance the rearward force"],
    ["12 m/s and β = 0°","Original study did not find trim","Force and pitch balance fail despite electrical pass"]
  ], widths:[344,337,457], close:"A rejected numerical candidate identifies a boundary or a search gap. Solver diagnostics distinguish those cases.", source:follow+"static_trim_audit.csv  /  "+study+"summary.json", notes:"At 30 m/s with the front pair vertical the stored normalized residual is about 0.02767 and pitch is at the negative bound. At 12 m/s and zero degrees the original study records 3.01 N forward, −0.97 N vertical, and −1.29 N·m in the best rejected sample. These are model-specific examples of failed steady balance."},

  {section:"Dynamic corridor", title:"Trajectory optimization and collocation", items:[
    ["Reference","Start from hover trim, target 30 m/s and 0° with settled terminal altitude, acceleration, and pitch motion."],
    ["Transcription","Use scaled state and actuator variables. Enforce dynamics with Hermite midpoint states and Simpson integration over each interval."],
    ["Feasibility","Constrain node and midpoint limits. Introduce independent steady-trim variables at each node to require equilibrium at its speed and tilt."],
    ["Objective","Penalize height change, pitch rate, actuator movement and curvature. The margin plan adds a peak-loss term, modeled power, and tail use."],
    ["Solver","CasADi supplies derivatives to IPOPT. Alternative tilt schedules and targeted waypoints generate multiple route candidates."]
  ], source:"src/dynamics/transition_trajectory.py  /  scripts/corridor_followup.py", notes:"The latest candidates retain the fixed 32 s planning horizon. This study optimizes the route shape and relative margins for that horizon. It does not claim a global minimum-time solution or a robust trajectory optimization across all scenarios."},

  {section:"Dynamic corridor", title:"Closed-loop simulation and actuator allocation", items:[
    ["Reference interpolation","Use a cubic Hermite curve for the planned states and interpolate actuator commands across the replay step."],
    ["Tracking feedback","Forward speed, climb rate, altitude, pitch, and pitch rate errors drive acceleration and moment corrections."],
    ["Bounded allocation","A local load Jacobian maps the corrections into front RPM, rear RPM, and V-tail changes. Bounded least squares respects travel and slew."],
    ["Tilt scheduler","Follow the prescribed nacelle trajectory. Disturbed cases may include actuator lag with a rate cap."],
    ["Plant replay","Integrate the numeric rigid-body model and compare computed loads with the symbolic model. Screen the full trajectory and the terminal state."]
  ], source:"scripts/transition_study.py  /  replay_trajectory", notes:"The reference is time based. The three outputs for allocation are forward acceleration, upward acceleration, and pitch moment. Weights prioritize pitch and vertical behavior relative to forward tracking. The replay samples at 0.01 s nominal and 0.02 s in scenario studies. The controller does not look up an escape route from the map."},

  {section:"Dynamic corridor", title:"Hard limits and inner screening margins", kind:"table", values:[
    ["Quantity","Hard study check","Inner screen"],
    ["Altitude deviation","≤ 3.0 m","≤ 0.75 m"],
    ["Pitch attitude / pitch rate","≤ 25° / 25°/s","≤ 18° / 18°/s"],
    ["Nacelle slew","≤ 20.05°/s","≤ 85% of hard limit"],
    ["Motor / V-tail slew","≤ 6,000 RPM/s / 60°/s","Same rate limits"],
    ["Wingborne stall margin","≥ 3°","≥ 5°"],
    ["Differential pitch reserve","≥ 0.60 N·m","≥ 1.0 N·m"],
    ["Electrical and tail reserves","Physical model limits apply","≥ 15% RPM, ≥ 20% current, ≥ 0.5 V, ≥ 1.5° tail travel"]
  ], widths:[327,354,457], source:study+"summary.json  /  "+refined+"summary.json", notes:"Hard checks also include acceleration magnitudes up to 3 m/s² forward and 2 m/s² upward, ±3 m/s climb rate, V-tail travel and terminal tracking. Inner screening includes a 0.15 N·m wingborne tail moment reserve. At low speeds, appropriate freestream and effective-incidence guards are applied. These are study thresholds rather than measured operational limits."},

  {section:"Dynamic corridor", title:"Observed reachability on the refined grid", kind:"figure", image:refined+"dynamic_reachability.png", points:["A cell can contain a sampled state from a full constraint-checked route.","The retained state coordinates receive an exact trim recheck.","The refined campaign reports 779 witnessed cells and 874 directed edges.","Six targeted searches added two verified routes."], source:refined+"summary.json  /  corridor_evidence.csv", notes:"The initial coarse dynamic map found 9 verified routes from 12 attempts and 225 witnessed cells. The refined campaign retains nine nominal routes in its summary and adds two traceable targeted probes. Directed edges describe consecutive observations in complete witnesses and cannot be composed arbitrarily across full states."},

  {section:"Dynamic corridor", title:"Static overlap and path interpretation", items:[
    ["Static center","A point solve asks whether level equilibrium is possible at the grid center."],
    ["Dynamic witness","A time sample records a specific state, control history and prior path. Its cell label is a projection of that evidence."],
    ["Selected route","All 417 exact sampled coordinates passed trim rechecks. The replay visits 105 bins, including 84 in the previously mapped static and dynamic overlap."],
    ["Map gaps","The 21 remaining bins have evidence from the selected replay at sampled coordinates. The previous map’s overlap uses its own center and witness tests."],
    ["Scope","Full recoverability would require state-dependent reachability or a validated recovery policy."]
  ], source:review+"summary.json  /  "+refined+"corridor_evidence.csv", notes:"Exact trim rechecks were taken roughly every 0.1 s and at bin witnesses. A trim at a visited coordinate does not imply trim at every point in its bin. The route was evaluated against state and actuator constraints throughout the nominal replay."},

  {section:"Trajectory evidence", title:"Reserve-aware 12 nominal time history", kind:"figure", image:review+"transition_review.png", points:["Top left: selected route over the prior overlap.","Top right: front and rear vertical thrust, wing lift, and V-tail force.","Bottom left: V-tail travel and wingborne stall margins.","Bottom right: results of the paired disturbance comparison."], source:review+"transition_review.png  /  selected_nominal_trace.csv", notes:"The selected nominal route reaches 30.001 m/s at 0° with 0.1497 m maximum altitude deviation. Its minimum wingborne stall margin is 9.284° and minimum V-tail travel reserve is 5.602°. The wing supplies an increasing fraction of support while the rear pair remains loaded at cruise and the model’s V-tail produces downward vertical force."},

  {section:"Trajectory evidence", title:"Force and moment closure over the route", kind:"figure", image:follow+"route_12_balance.png", points:["Check ΣFup − mg − m aup at every recorded sample.","Check ΣMy − Iyy q̇ against the integrated pitch acceleration.","Route 12 closure: 2.6 × 10⁻¹² N vertical and 1.5 × 10⁻¹² N·m pitch.","The selected controller shares Route 12’s nominal trace."], source:follow+"route_12_balance.csv  /  summary.json", notes:"The sum of vertical loads includes front/rear thrust, vertical wing lift and drag, V-tail load, and gravity. In a dynamic route the net force can be nonzero. The relevant diagnostic is force minus inertial acceleration, and similarly moment minus inertia times angular acceleration. These residuals verify the bookkeeping, not aerodynamic fidelity."},

  {section:"Trajectory evidence", title:"Nominal margin and actuator trade-offs", kind:"table", values:[
    ["Metric","Reference 3","Reserve-aware 12"],
    ["Duration / terminal airspeed","32 s / 30.0 m/s","32 s / 30.0 m/s"],
    ["Peak altitude deviation","0.039 m","0.150 m"],
    ["Peak pitch rate / tilt rate","6.10°/s / 5.41°/s","8.40°/s / 7.85°/s"],
    ["Peak motor slew / V-tail slew","2,415 RPM/s / 16.87°/s","6,000 RPM/s / 58.64°/s"],
    ["Minimum stall / V-tail travel reserve","9.60° / 4.59°","9.28° / 5.60°"],
    ["Minimum differential pitch reserve","2.04 N·m","2.04 N·m"],
    ["Modeled electrical energy","0.162 Wh","0.150 Wh"]
  ], widths:[455,338,345], close:"The selected route reaches the modeled motor slew cap and approaches the tail slew cap.", source:review+"candidate_comparison.csv", notes:"The final candidate offers about 7.3% lower modeled nominal electrical energy than Reference 3 and improved case coverage. Reference 3 has smaller nominal altitude excursions and lower actuator rates. Values are rounded from candidate_comparison.csv. The operating limit of the actual installed actuators remains unmeasured."},

  {section:"Disturbance testing", title:"Matched scenario and gust design", items:[
    ["Shared bank","Each controller replays the same 26 cases. Six named cases and 16 seeded parameter draws form the original bank."],
    ["Parameter uncertainty","Vary mass, longitudinal CG, wing lift slope, drag, thrust coefficient, and tilt response."],
    ["Dryden-like cases","Add four seeded first-order along-track gusts. Use a 0.35 m/s input standard deviation and 1 s time constant with a tapered interval."],
    ["Independent holdout","After ranking, replay the selected controller with four fresh gust seeds on the nominal plant."],
    ["Acceptance","Require a complete trajectory with all applicable checks and settled terminal tracking."]
  ], source:"scripts/corridor_uncertainty.py  /  scripts/transition_review.py", notes:"Original random cases use seed 404. Comparison gusts use seeds 701–704 and holdout gusts use 901–904. The gust envelope ramps in at 3–5 s and ramps out by 29 s, isolating terminal settling. This longitudinal first-order approximation and small sample set do not estimate corridor failure probability."},

  {section:"Disturbance testing", title:"The V-tail command reserve improved test coverage", kind:"chart", categories:["Reference 3","Margin plan 12","Reserve-aware 12"], series:[["Passed", [4,4,9], "#147D72"],["Failed", [22,22,17], "#CCD8DF"]], close:"The selected route passed all four comparison gusts and all four held-out seeds. The original 22-case bank still contains 17 failures.", source:review+"candidate_comparison.csv  /  scenario_results.csv", notes:"Reserve-aware 12 uses the same optimized reference as Margin plan 12 and keeps another 1.6° of working command travel in reserve. Both share a nominal trace because the added clip is inactive in that run. Case coverage, then modeled nominal energy, determined the result after the nominal inner screen."},

  {section:"Disturbance testing", title:"First violations on the selected route", kind:"table", values:[
    ["First result","Count","Diagnostic interpretation"],
    ["Passed","9","Completed all required checks in the shared bank"],
    ["Differential pitch reserve","6","Rotor thrust redistribution no longer provides required pitch headroom"],
    ["Altitude deviation","4","Height error crossed the inner screen"],
    ["Flow guard","4","Freestream or effective wing incidence crossed its speed-dependent bound"],
    ["Pitch rate","3","Pitch motion crossed the allowable rate"]
  ], widths:[367,143,628], close:"The earlier four-plan campaign logged tail travel reserve first in 47 of 88 cases.", source:review+"scenario_results.csv  /  "+follow+"summary.json", notes:"Counts are first flagged constraints, one per shared case. Later violations can occur during the same replay. The earlier 47-of-88 result uses a different candidate ensemble and case bank, so the counts explain the design motivation rather than serving as a paired improvement metric."},

  {section:"Disturbance testing", title:"Candidate disturbance-tested corridor", kind:"figure", image:refined+"tested_margin_corridor.png", points:["A route must pass the full case bank.","Eligible bins must occur in every replay of that route.","No route met the criterion. The inner map has zero all-case cells.","A probabilistic corridor remains open."], source:refined+"tested_margin_corridor.png  /  "+review+"summary.json", notes:"The plotted map is the refined campaign’s published inner test map. Its all-case condition intersects bins across the nominal and disturbed trajectories of a route that passes the complete bank. Later reserve-aware testing improves case coverage without qualifying any route under the all-case rule. Neither a continuous uncertainty guarantee nor a flight safety envelope follows from the map."},

  {section:"Off-nominal cases", title:"Front- and rear-pair thrust-loss injections", kind:"table", values:[
    ["Scenario","Injection","First observed response"],
    ["Both front rotors lose thrust","12.0 s, about 10.04 m/s","Vertical acceleration limit crossed at injection"],
    ["Both rear rotors lose thrust","12.0 s, about 10.04 m/s","Vertical acceleration limit crossed at injection"]
  ], widths:[392,270,476], close:"The plant removes thrust while the existing controller continues its nominal tracking logic.", source:review+"pair_failure_results.csv  /  scripts/transition_review.py", notes:"The run implements plant-side force loss by scaling applied rotor speed by the square root of thrust availability. This preserves the T ∝ ω² relation and leaves command telemetry untouched. Both runs terminate at the hard limit and do not evaluate fault detection, recovery, asymmetric rotor failure or escape."},

  {section:"Validation status", title:"Evidence required before flight validation", items:[
    ["Airframe and actuators","Measure mass, CG, installed thrust and torque, battery sag, actuator speed, delays, and loaded travel."],
    ["Aerodynamics","Validate wing and V-tail coefficients, post-stall response, and rotor interference over transition conditions."],
    ["Robustness and recovery","Evaluate broader turbulence and crosswind, asymmetric failures, and a corridor-aware recovery law."],
    ["SITL and HIL","Build and verify the aircraft-specific PX4/Gazebo configuration. Compare software replay first, then assess hardware timing and saturation."],
    ["Acceptance","Use repeatable, independently checked evidence for margins, terminal tracking, and recovery across the required cases."]
  ], source:"others/FINAL_DELIVERABLES.md", notes:"The repository has a PX4 mapping scaffold and offline tests. No aircraft-specific autopilot run, hardware-in-the-loop session, or physical aircraft clearance is documented for this result. Treat the selected route as a simulation candidate."},

  {section:"Appendix", title:"Uncertainty ranges used in the study", kind:"table", values:[
    ["Parameter","Uniform draw range","Interpretation"],
    ["Mass scale / longitudinal CG","0.95–1.05 / −6 to +6 mm","Relative to the 5 kg reference and modeled CG"],
    ["Wing lift slope / drag scale","0.92–1.08 / 0.88–1.12","Multipliers on the nominal model"],
    ["Rotor thrust coefficient scale","0.92–1.05","Applies to the modeled plant"],
    ["Nacelle response time","0.03–0.20 s","First-order response with rate limit"],
    ["Prescribed gust / onset","−0.8 to +0.8 m/s / 5–22 s","Along-track wind input and start time"]
  ], widths:[376,347,415], close:"The study used 16 joint parameter draws with seed 404. These ranges are modeling assumptions.", source:"scripts/corridor_uncertainty.py  /  scenarios", notes:"Independent uniform draws generate each row’s input values. Named stress cases are separate from the draws. Dryden-like gust tests are a subsequent test family and retain the nominal parameter set, so their result does not address combined parameter and stochastic uncertainty."},

  {section:"Appendix", title:"Reproducing the analysis", kind:"table", values:[
    ["Command from the repository root","Evidence produced"],
    ["python -m scripts.transition_study","Baseline trim, route optimization, and replay"],
    ["python -m scripts.refined_corridor","Refined static tables and supporting route data"],
    ["python -m scripts.corridor_validation","Dynamic and tested inner maps with case results"],
    ["python -m scripts.probe_corridor_cells","Targeted searches from static-only regions"],
    ["python -m scripts.probe_corridor_cells --merge","Merge new witness routes into refined evidence"],
    ["python -m scripts.corridor_followup","Static audit, load closure, and diagnostics"],
    ["python -m scripts.transition_review","Candidate ranking, gusts, and pair-outage tests"]
  ], widths:[547,591], close:"The recorded regression run passed 200 tests. Two INDI test modules were excluded because their implementations were absent.", source:"others/FINAL_DELIVERABLES.md  /  recorded regression run", notes:"Run inside the project Python environment. The recorded regression command used pytest -p no:cacheprovider --ignore=tests/test_indi.py --ignore=tests/test_indi_dob.py -q. Solver and replay runtime depend on the local CPU and dependencies. The deck reports saved experiments, and does not execute analysis on open."},

  {section:"Appendix", title:"Source files and evidence records", kind:"table", values:[
    ["Evidence","Where to trace it"],
    ["Airframe and motion","aircraft.yaml, geometry_offsets.yaml, src/dynamics/tiltrotor.py"],
    ["Aerodynamic and optimization models","src/aero/wing_polar.py, src/dynamics/transition_trajectory.py"],
    ["Trim and map membership","static_trim.csv, followup/static_trim_audit.csv, corridor_evidence.csv"],
    ["Routes and sampled connections","nominal_route_checks.csv, observed_edges.csv, probe_witnesses.csv"],
    ["Route constraints and load balance","route_12_balance.csv and selected_nominal_trace.csv"],
    ["Disturbance and fault results","scenario_results.csv, candidate_comparison.csv, pair_failure_results.csv"]
  ], widths:[412,726], close:"Corridor evidence is under figures/transition_study/refined_corridor/.", source:"others/FINAL_DELIVERABLES.md  /  project results tree", notes:"Speaker notes on individual slides retain the assumptions and source files behind each numerical claim. Debugging and audit data remain in the original adjacent CSV and JSON outputs."}
];
