try:
    from abaqus import *
    from abaqusConstants import *
    import regionToolset
    import mesh
except:
    pass

# =====================================================================
# 1. USER PARAMETERS
# =====================================================================
# Geometry (Units: mm)
h_a = 1.0  # Adhesive thickness
h_b = 0.5  # Backing layer thickness (top)
h_s = 0.5  # Substrate layer thickness (bottom)
c_a = 150.0  # Total length coefficient
c_1 = 50.0  # Top notch length coefficient
c_2 = 50.0  # Bottom notch length coefficient

eps = 0.01 * h_a  # Small gap defining the explicit V-notch width
r_jint = 0.3 * min(h_a, h_b, h_s)  # Radius for J-integral partition

# Distance for the vertical partition lines (must be larger than r_jint)
x_off = 3.0 * r_jint

# Material (Units: MPa, N)
E_ad_kPa = 100.0  # Adhesive modulus in kPa
E_ad = E_ad_kPa / 1000.0  # Convert to MPa
mu = E_ad / 3.0  # Small strain shear modulus

# Yeoh coefficients (incompressible D=0)
C10 = 0.5 * mu
C20 = -0.0237 * mu
C30 = 0.00166 * mu

# Backing/Substrate properties
E_rigid = 10000.0  # 1 GPa = 1000 MPa
nu_rigid = 0.3

# Load
mode = "peel"  # "release" or "peel"
if mode not in ["release", "peel"]:
    raise ValueError("Invalid mode. Choose 'release' or 'peel'.")


def choose_traction(m):
    if m == "peel":
        return 30.0
    elif m == "release":
        return 0.3


applied_traction = choose_traction(mode)

# Mesh
n_elements_radius = 60
n_contour = 30

# Derived geometric coordinates
L = c_a * h_a
H = h_s + h_a + h_b
y1 = h_s + h_a  # Top notch y-coordinate
y2 = h_s  # Bottom notch y-coordinate
L1 = c_1 * h_a  # Top notch x-tip
L2 = c_2 * h_a  # Bottom notch x-tip

global_size = H / 20.0


def build_model():
    # =====================================================================
    # 2. MODEL & SKETCH INITIALIZATION
    # =====================================================================
    Mdb()
    model = mdb.models["Model-1"]
    model.setValues(description="Command Tape Fracture Model")

    s = model.ConstrainedSketch(name="tape_profile", sheetSize=2000.0)

    # Draw external boundary with explicit V-notches
    s.Line(point1=(0, H), point2=(0, y1 + eps))
    s.Line(point1=(0, y1 + eps), point2=(L1, y1))
    s.Line(point1=(L1, y1), point2=(0, y1 - eps))
    s.Line(point1=(0, y1 - eps), point2=(0, y2 + eps))
    s.Line(point1=(0, y2 + eps), point2=(L2, y2))
    s.Line(point1=(L2, y2), point2=(0, y2 - eps))
    s.Line(point1=(0, y2 - eps), point2=(0, 0))
    s.Line(point1=(0, 0), point2=(L, 0))
    s.Line(point1=(L, 0), point2=(L, H))
    s.Line(point1=(L, H), point2=(0, H))

    p = model.Part(name="Tape", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)

    # =====================================================================
    # 3. PARTITIONING (UPDATED WITH VERTICAL LINES)
    # =====================================================================
    s_part = model.ConstrainedSketch(name="partitions", sheetSize=2000.0)

    # Horizontal lines from notch tips to the right end
    s_part.Line(point1=(L1, y1), point2=(L, y1))
    s_part.Line(point1=(L2, y2), point2=(L, y2))

    # Circular regions around notch tips
    s_part.CircleByCenterPerimeter(center=(L1, y1), point1=(L1 + r_jint, y1))
    s_part.CircleByCenterPerimeter(center=(L2, y2), point1=(L2 + r_jint, y2))

    # Bounding Box Overlap Logic
    min_L = min(L1, L2)
    max_L = max(L1, L2)

    if (max_L - min_L) <= 2 * x_off:
        # Bounding boxes intersect -> Draw ONE unified bounding box
        box_left = min_L - x_off
        box_right = max_L + x_off
        s_part.Line(point1=(box_left, 0), point2=(box_left, H))
        s_part.Line(point1=(box_right, 0), point2=(box_right, H))
    else:
        # Bounding boxes do not intersect -> Draw TWO separate bounding boxes
        # Vertical lines bounding Top Notch
        s_part.Line(point1=(L1 - x_off, 0), point2=(L1 - x_off, H))
        s_part.Line(point1=(L1 + x_off, 0), point2=(L1 + x_off, H))

        # Vertical lines bounding Bottom Notch
        s_part.Line(point1=(L2 - x_off, 0), point2=(L2 - x_off, H))
        s_part.Line(point1=(L2 + x_off, 0), point2=(L2 + x_off, H))

    p.PartitionFaceBySketch(faces=p.faces, sketch=s_part)

    # =====================================================================
    # 4. MATERIALS & SECTIONS
    # =====================================================================
    mat_ad = model.Material(name="Adhesive")
    mat_ad.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=YEOH,
        volumetricResponse=VOLUMETRIC_DATA,
        table=((C10, C20, C30, 0.0, 0.0, 0.0),),
    )

    mat_stiff = model.Material(name="Adhesive_Linear")
    mat_stiff.Elastic(table=((E_ad, 0.495),))

    mat_stiff = model.Material(name="Stiff_Layer")
    mat_stiff.Elastic(table=((E_rigid, nu_rigid),))

    model.HomogeneousSolidSection(
        name="Sec-Adhesive", material="Adhesive", thickness=1.0
    )
    model.HomogeneousSolidSection(
        name="Sec-Stiff", material="Stiff_Layer", thickness=1.0
    )

    pts_ad, pts_top, pts_bot = [], [], []
    for f in p.faces:
        cx, cy, cz = f.pointOn[0]
        if cy > y1 + 1e-4:
            pts_top.append(((cx, cy, cz),))
        elif cy < y2 - 1e-4:
            pts_bot.append(((cx, cy, cz),))
        else:
            pts_ad.append(((cx, cy, cz),))

    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces.findAt(*pts_ad)),
        sectionName="Sec-Adhesive",
    )
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces.findAt(*pts_top)),
        sectionName="Sec-Stiff",
    )
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces.findAt(*pts_bot)),
        sectionName="Sec-Stiff",
    )

    # =====================================================================
    # 5. MESHING (STRUCTURED SWEPT MESH IN CRACK TIP REGIONS)
    # =====================================================================
    elemQuad = mesh.ElemType(elemCode=CPE4H, elemLibrary=STANDARD)
    elemTri = mesh.ElemType(elemCode=CPE3H, elemLibrary=STANDARD)
    p.setElementType(
        regions=regionToolset.Region(faces=p.faces), elemTypes=(elemQuad, elemTri)
    )

    pts_far = []
    pts_near = []
    pts_circles = []

    # 5a. Identify faces for meshing strategy
    min_L = min(L1, L2)
    max_L = max(L1, L2)
    boxes_intersect = (max_L - min_L) <= 2 * x_off

    for f in p.faces:
        cx, cy, cz = f.pointOn[0]
        dist1 = ((cx - L1) ** 2 + (cy - y1) ** 2) ** 0.5
        dist2 = ((cx - L2) ** 2 + (cy - y2) ** 2) ** 0.5

        # Sort faces into Circles, Unstructured vertical bands, or Structured far-field
        if dist1 < r_jint or dist2 < r_jint:
            pts_circles.append(((cx, cy, cz),))
        else:
            if boxes_intersect:
                # Unified bounding box bounds
                if min_L - x_off - 1e-4 < cx < max_L + x_off + 1e-4:
                    pts_near.append(((cx, cy, cz),))
                else:
                    pts_far.append(((cx, cy, cz),))
            else:
                # Separate bounding box bounds
                if (L1 - x_off - 1e-4 < cx < L1 + x_off + 1e-4) or (
                    L2 - x_off - 1e-4 < cx < L2 + x_off + 1e-4
                ):
                    pts_near.append(((cx, cy, cz),))
                else:
                    pts_far.append(((cx, cy, cz),))

    # Apply mesh shapes and techniques
    if pts_far:
        # Structured quads for the far-field
        p.setMeshControls(
            regions=p.faces.findAt(*pts_far), elemShape=QUAD, technique=STRUCTURED
        )
    if pts_near:
        # Free quad-dominated mesh for the transition zone
        p.setMeshControls(
            regions=p.faces.findAt(*pts_near), elemShape=QUAD_DOMINATED, technique=FREE
        )
    if pts_circles:
        # Use a swept mesh, but ALLOW triangles at the degenerate crack tip using QUAD_DOMINATED
        p.setMeshControls(
            regions=p.faces.findAt(*pts_circles),
            elemShape=QUAD_DOMINATED,
            technique=SWEEP,
        )

    # 5b. Global Seeding
    p.seedPart(size=global_size, deviationFactor=0.1, minSizeFactor=0.1)

    # 5c. Biased Seeding for the Crack Tip Circles
    bias_ratio = (
        5.0  # Ratio of the largest element size to the smallest element size at the tip
    )

    end1_pts = []
    end2_pts = []
    circ_pts = []

    for e in p.edges:
        cx, cy, cz = e.pointOn[0]
        dist1_mid = ((cx - L1) ** 2 + (cy - y1) ** 2) ** 0.5
        dist2_mid = ((cx - L2) ** 2 + (cy - y2) ** 2) ** 0.5

        # Check if the edge is the circular partition perimeter
        if abs(dist1_mid - r_jint) < 1e-4 or abs(dist2_mid - r_jint) < 1e-4:
            circ_pts.append(((cx, cy, cz),))
            continue

        # Get the edge's vertices to determine its topological direction
        v_indices = e.getVertices()
        if not v_indices:
            continue

        pt1 = p.vertices[v_indices[0]].pointOn[0]
        pt2 = p.vertices[v_indices[1]].pointOn[0]

        dist1_pt1 = ((pt1[0] - L1) ** 2 + (pt1[1] - y1) ** 2) ** 0.5
        dist1_pt2 = ((pt2[0] - L1) ** 2 + (pt2[1] - y1) ** 2) ** 0.5
        dist2_pt1 = ((pt1[0] - L2) ** 2 + (pt1[1] - y2) ** 2) ** 0.5
        dist2_pt2 = ((pt2[0] - L2) ** 2 + (pt2[1] - y2) ** 2) ** 0.5

        # If the edge is inside the circular domain and touches the tip
        if dist1_mid < r_jint or dist2_mid < r_jint:
            # Check if Vertex 1 is at the crack tip (Bias towards End 1)
            if dist1_pt1 < 1e-4 or dist2_pt1 < 1e-4:
                end1_pts.append(((cx, cy, cz),))
            # Check if Vertex 2 is at the crack tip (Bias towards End 2)
            elif dist1_pt2 < 1e-4 or dist2_pt2 < 1e-4:
                end2_pts.append(((cx, cy, cz),))

    # Apply biased seeds to radial lines
    if end1_pts:
        p.seedEdgeByBias(
            end1Edges=p.edges.findAt(*end1_pts),
            ratio=bias_ratio,
            number=n_elements_radius,
        )
    if end2_pts:
        p.seedEdgeByBias(
            end2Edges=p.edges.findAt(*end2_pts),
            ratio=bias_ratio,
            number=n_elements_radius,
        )

    # Apply a uniform seed to the circle's perimeter to constrain the outer sweep dimensions
    if circ_pts:
        # Mathematical approximation of the outer element size based on the bias progression
        outer_size = (2.0 * bias_ratio * r_jint) / (
            n_elements_radius * (bias_ratio + 1.0)
        )
        p.seedEdgeBySize(edges=p.edges.findAt(*circ_pts), size=outer_size)

    p.generateMesh()

    # =====================================================================
    # 6. ASSEMBLY, STEP, BCs, AND LOADS
    # =====================================================================
    a = model.rootAssembly
    inst = a.Instance(name="Tape-inst", part=p, dependent=ON)

    model.StaticStep(
        name="Pull_Step",
        previous="Initial",
        nlgeom=ON,
        initialInc=0.05,
        minInc=1e-6,
        maxInc=0.1,
    )

    # BC: Fix bottom edge
    pts_bot_edge = []
    for e in inst.edges:
        cx, cy, cz = e.pointOn[0]
        if abs(cy - 0.0) < 1e-4:
            pts_bot_edge.append(((cx, cy, cz),))

    if pts_bot_edge:
        model.EncastreBC(
            name="Fix_Bottom",
            createStepName="Initial",
            region=regionToolset.Region(edges=inst.edges.findAt(*pts_bot_edge)),
        )

    if mode == "peel":
        pts_left = []
        for e in inst.edges:
            cx, cy, cz = e.pointOn[0]
            if abs(cx) < 1e-4 and (cy > y1 + eps / 2):
                pts_left.append(((cx, cy, cz),))

    elif mode == "release":
        # Load: Pull the left edge of the middle part
        pts_left = []
        for e in inst.edges:
            cx, cy, cz = e.pointOn[0]
            if abs(cx) < 1e-4 and (y2 + eps / 2 < cy < y1 - eps / 2):
                pts_left.append(((cx, cy, cz),))

    if pts_left:
        # Applying a General Surface Traction pulling to the left (negative x-direction)
        model.SurfaceTraction(
            name="Pull_Left",
            createStepName="Pull_Step",
            region=regionToolset.Region(side1Edges=inst.edges.findAt(*pts_left)),
            magnitude=applied_traction,
            directionVector=((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            distributionType=UNIFORM,
            traction=GENERAL,
            follower=OFF,  # Fixes the load vector direction spatially even as the edge rotates
            resultant=ON,  # In the reference configuration
        )

    # =====================================================================
    # 7. CRACK DEFINITION & J-INTEGRAL OUTPUT REQUEST
    # =====================================================================
    # Crack 1: Top Notch
    v1 = inst.vertices.findAt(((L1, y1, 0.0),))
    if v1:
        q_vec_top = (((L1, y1, 0.0), (L1 + 1.0, y1, 0.0)),)

        a.engineeringFeatures.ContourIntegral(
            name="J-Top_Notch",
            crackFront=regionToolset.Region(vertices=v1),
            crackTip=regionToolset.Region(vertices=v1),
            extensionDirectionMethod=Q_VECTORS,
            qVectors=q_vec_top,
            symmetric=OFF,
        )

        model.HistoryOutputRequest(
            name="Hout_J-Top",
            createStepName="Pull_Step",
            contourIntegral="J-Top_Notch",
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
            numberOfContours=n_contour,
            contourType=J_INTEGRAL,
        )

    # Crack 2: Bottom Notch
    v2 = inst.vertices.findAt(((L2, y2, 0.0),))
    if v2:
        q_vec_bot = (((L2, y2, 0.0), (L2 + 1.0, y2, 0.0)),)

        a.engineeringFeatures.ContourIntegral(
            name="J-Bot_Notch",
            crackFront=regionToolset.Region(vertices=v2),
            crackTip=regionToolset.Region(vertices=v2),
            extensionDirectionMethod=Q_VECTORS,
            qVectors=q_vec_bot,
            symmetric=OFF,
        )

        model.HistoryOutputRequest(
            name="Hout_J-Bot",
            createStepName="Pull_Step",
            contourIntegral="J-Bot_Notch",
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
            numberOfContours=n_contour,
            contourType=J_INTEGRAL,
        )

    # =====================================================================
    # 8. ELEMENT STRAIN TRACKING
    # =====================================================================
    # Identify the coordinate for the center of the hanging adhesive portion
    if mode == "peel":
        x_track = max(L1, L2) + (L - max(L1, L2)) / 2
    elif mode == "release":
        x_track = 0.1 * min(L1, L2)
    y_track = h_s + 0.5 * h_a

    # Mesh elements do not have 'findAt'. We use a small bounding box instead.
    tol_x = global_size * 1.1  # X-tolerance
    tol_y = (
        global_size * 1.1
    )  # Y-tolerance kept tight to avoid grabbing the stiff layers

    strain_elems = inst.elements.getByBoundingBox(
        xMin=x_track - tol_x,
        yMin=y_track - tol_y,
        zMin=-0.1,
        xMax=x_track + tol_x,
        yMax=y_track + tol_y,
        zMax=0.1,
    )

    if len(strain_elems) > 0:
        # Create a set using just the first element found in our bounding box
        a.Set(elements=strain_elems[0:1], name="Center_Tracked_Element")
        model.HistoryOutputRequest(
            name="Hout_Strain_Center",
            createStepName="Pull_Step",
            variables=(
                "LE11",
                "LE22",
                "LE12",
                "U1",
                "U2",
                "SENER",
                "ELSE",
            ),  # Added SENER and ELSE
            region=a.sets["Center_Tracked_Element"],
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
        )

    print(
        "Model generated successfully! Switch to the Job module to create and submit your job."
    )


def get_h_and_nodes_from_inp(inp_file, el_id):
    """
    Parses an Abaqus .inp file to find the node connectivity for a given
    element ID, extracts their initial coordinates, and computes the
    vertical thickness (h) between the top and bottom nodes.
    """
    node_ids = []
    in_element_block = False

    # 1. Scan for the element connectivity
    with open(inp_file, "r") as f:
        for line in f:
            line_upper = line.strip().upper()
            if line_upper.startswith("*ELEMENT"):
                in_element_block = True
                continue
            elif line_upper.startswith("*"):
                in_element_block = False
                continue

            if in_element_block:
                parts = line.split(",")
                try:
                    if int(parts[0].strip()) == el_id:
                        # Extract the connected node IDs (ignoring the element ID)
                        node_ids = [int(p.strip()) for p in parts[1:] if p.strip()]
                        break
                except ValueError:
                    continue

    if not node_ids:
        raise ValueError("Element ID {} not found in {}".format(el_id, inp_file))

    if len(node_ids) != 4:
        raise ValueError(
            "Expected 4 nodes for element {}, found {}".format(el_id, len(node_ids))
        )

    # 2. Scan for the coordinates of those specific nodes
    nodes = {}
    in_node_block = False

    with open(inp_file, "r") as f:
        for line in f:
            line_upper = line.strip().upper()
            if line_upper.startswith("*NODE"):
                in_node_block = True
                continue
            elif line_upper.startswith("*"):
                in_node_block = False
                continue

            if in_node_block:
                parts = line.split(",")
                try:
                    n_id = int(parts[0].strip())
                    if n_id in node_ids:
                        nodes[n_id] = {
                            "id": n_id,
                            "x": float(parts[1].strip()),
                            "y": float(parts[2].strip()),
                        }
                except ValueError:
                    continue

    # 3. Geometrically sort the nodes to find the vertical pair
    node_list = list(nodes.values())

    # Sort by Y-coordinate
    node_list.sort(key=lambda n: n["y"])
    bottom_nodes = node_list[:2]
    top_nodes = node_list[2:]

    # Sort by X-coordinate to guarantee a vertically aligned pair
    bottom_nodes.sort(key=lambda n: n["x"])
    top_nodes.sort(key=lambda n: n["x"])

    bot_node = bottom_nodes[0]
    top_node = top_nodes[0]

    h = top_node["y"] - bot_node["y"]

    return h, top_node["id"], bot_node["id"]


def run_and_extract():
    import datetime
    import odbAccess

    # =====================================================================
    # 9. SUBMIT JOB AND WAIT FOR COMPLETION
    # =====================================================================
    # Generate a unique job name based on the current time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = "Tape_Fracture_" + mode + "_" + current_time

    print("\n--- Submitting Job: " + job_name + " ---")
    mdb.Job(name=job_name, model="Model-1", type=ANALYSIS, numCpus=1, numDomains=1)
    mdb.jobs[job_name].setValues(numThreadsPerMpiProcess=1, numCpus=10, numDomains=10)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

    if mdb.jobs[job_name].status != COMPLETED:
        print("\nWARNING: Job did not complete successfully.")
        print("Final Job Status: " + str(mdb.jobs[job_name].status))
        print(
            "Aborting data extraction. Please check the .msg or .dat files for errors."
        )
        return  # Exit the function early so it doesn't crash on odbAccess

    print("Job completed successfully!")

    # =====================================================================
    # 10. EXTRACT ODB DATA AND WRITE TO EXCEL-READABLE CSV
    # =====================================================================
    odb_path = job_name + ".odb"
    inp_path = job_name + ".inp"
    csv_filename = "/home/xlluo/abaqus/Commands/" + job_name + ".csv"

    print("Opening ODB to extract data...")
    try:
        odb = odbAccess.openOdb(path=odb_path, readOnly=True)
        step = odb.steps["Pull_Step"]

        # Dictionary to store data by exact time step: {time: {variable_name: value}}
        data_by_time = {}
        var_names = set()

        # Traverse History Regions to find J-integral, Strain, and Energy data
        for hr_name, hr in step.historyRegions.items():
            for ho_name, ho in hr.historyOutputs.items():

                col_name = None

                # Check for J-Integral
                if "J at" in ho_name:
                    col_name = ho_name

                # Check for Strain (LE) and Strain Energy (SENER, ELSE)
                target_vars = [
                    "LE11",
                    "LE22",
                    "LE12",
                    "SENER",
                    "ELSE",
                ]
                if any(var in ho_name for var in target_vars):
                    if "Element" in hr_name:  # Ensure it is our tracked element
                        col_name = ho_name + "_" + hr_name.split()[1]
                        element_id = hr_name.split(".")[1].split()[0]

                # If we identified a column, store its time-value pairs safely
                if col_name:
                    var_names.add(col_name)
                    for time_val, data_val in ho.data:
                        # Round time slightly to avoid floating point mismatch issues as keys
                        t_key = round(time_val, 6)
                        if t_key not in data_by_time:
                            data_by_time[t_key] = {}
                        data_by_time[t_key][col_name] = data_val

        target_element_id = int(element_id)
        h, top_node_id, bot_node_id = get_h_and_nodes_from_inp(
            inp_path, target_element_id
        )

        # 3. Locate the instance containing the element in the ODB
        inst_name = ""
        for name, instance in odb.rootAssembly.instances.items():
            matched = [el for el in instance.elements if el.label == target_element_id]
            if matched:
                inst_name = name
                break

        if not inst_name:
            raise ValueError(
                "Element ID {} found in .inp but not in ODB instances.".format(
                    target_element_id
                )
            )

        # 4. Extract from Field Output (Raw Frame Data)
        # Create a temporary node set to drastically speed up extraction
        temp_set_name = "TEMP_TRACKED_NODES_{}".format(target_element_id)
        if temp_set_name not in odb.rootAssembly.nodeSets:
            odb.rootAssembly.NodeSetFromNodeLabels(
                name=temp_set_name,
                nodeLabels=((inst_name, (int(top_node_id), int(bot_node_id))),),
            )
        tracked_region = odb.rootAssembly.nodeSets[temp_set_name]

        gamma_history = []

        # Iterate through all saved frames in the step
        for frame in step.frames:
            time = round(frame.frameValue, 6)

            # Check if displacements were written to this frame
            if "U" not in frame.fieldOutputs:
                continue

            u_field = frame.fieldOutputs["U"]

            # Extract displacement strictly for our two tracked nodes
            u_subset = u_field.getSubset(region=tracked_region).values

            val_top = None
            val_bot = None

            for val in u_subset:
                if val.nodeLabel == top_node_id:
                    val_top = val.data[0]  # data[0] is U1, data[1] is U2
                elif val.nodeLabel == bot_node_id:
                    val_bot = val.data[0]

            # If both nodes exist in this frame, compute the shear
            if val_top is not None and val_bot is not None:
                gamma = (val_top - val_bot) / h
                gamma_history.append((time, gamma))

        # 5. Append to the master data dictionary
        var_names.add("Gamma")
        for time_val, data_val in gamma_history:
            # Round time slightly to avoid floating point mismatch issues as keys
            t_key = round(time_val, 6)
            if t_key not in data_by_time:
                data_by_time[t_key] = {}
            data_by_time[t_key]["Gamma"] = data_val

        odb.close()

        # Write to CSV
        print("Writing extracted data to " + csv_filename + "...")
        with open(csv_filename, "w") as f:
            headers = sorted(list(var_names))
            f.write("Time," + ",".join(headers) + "\n")

            # Sort the time keys so the rows are written chronologically
            sorted_times = sorted(data_by_time.keys())

            for t in sorted_times:
                row = [str(t)]
                for h in headers:
                    # Fetch the value for this specific time.
                    # If it's missing (like J-integral at t=0), it defaults to 0.0.
                    if h in data_by_time[t]:
                        val = data_by_time[t][h]
                    else:
                        val = 0.0 if t == 0.0 else ""  # Mechanically, J is 0 at t=0

                    row.append(str(val))
                f.write(",".join(row) + "\n")

        print("\nSUCCESS: All data properly time-aligned and saved to " + csv_filename)

    except Exception as e:
        print("Error during ODB extraction: " + str(e))


if __name__ == "__main__":
    build_model()
    run_and_extract()
