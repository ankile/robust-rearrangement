task2idx = {
    "one_leg": 8,
    "square_table": 8,
    "lamp": 1,
    "round_table": 2,
    "desk": 3,
    "cabinet": 4,
    "stool": 5,
    "chair": 6,
    "drawer": 7,
    "place_shade": 9,
    "mug_rack": 10,
    "factory_peg_hole": 11,
    "factory_nut_bolt": 12,
    "bimanual_insertion": 13,
}


timesteps_per_phase = 200

task_phases = {
    "one_leg": 5,
    "square_table": 16,
    "lamp": 7,
    "round_table": 8,
    "desk": 16,
    "cabinet": 11,
    "stool": 11,
    "chair": 17,
    "drawer": 8,
    "mug_rack": 2,
    "factory_peg_hole": 2,
    "factory_nut_bolt": 2,
}


timesteps_per_part = 1_000  # Reduced from 1000 to 750 after moving to position actions
# Set back to 1_000 for now to match the original runs

task_parts = {
    "one_leg": 1,
    "square_table": 4,
    "lamp": 2,
    "round_table": 2,
    "desk": 4,
    "cabinet": 3,
    "stool": 3,
    "chair": 5,
    "drawer": 2,
    "mug_rack": 2,
}


def task_timeout(task, n_parts=None):
    assert task in task_parts, f"Task {task} not found"
    n_parts = task_parts[task] if n_parts is None else n_parts

    assert n_parts <= task_parts[task], f"Task {task} only has {task_parts[task]} parts"

    return n_parts * timesteps_per_part


idx2task = {v: k for k, v in task2idx.items()}

# Here we can with have task descriptions in natural language as well
complex_task_descriptions = {
    "lamp": [
        "Assemble the lamp",
        "Align the base, screw in the bulb, and place the lamp shade",
        "Screw the bulb into the base and put the shade on top",
        "Construct the lamp by first attaching its base, then installing the bulb, followed by setting the lampshade in its proper position.",
        "Begin by positioning the base, proceed to insert the bulb into the socket, and finish by adding the lampshade.",
        "First, fix the base in place, then mount the bulb, and finally, cap it with the lampshade.",
        "Initiate the lamp assembly by securing the base, then twist the bulb into the socket, and conclude by placing the shade over it.",
        "Start with stabilizing the base, follow up by embedding the bulb, and complete by draping the shade over the lamp.",
        "Establish the lamp's foundation by setting up the base, illuminate it by screwing in the bulb, and crown it with the lampshade.",
        "Prepare the lamp by first arranging the base, then efficiently screwing the bulb in place, and ultimately positioning the shade on it.",
        "First ensure the base is properly aligned, then carefully insert the bulb, and top it off with the lampshade.",
        "Kick-off the assembly by setting the base right, then twist in the bulb carefully, and gently rest the shade on the structure.",
        "Embark on assembling the lamp by first making sure the base is steady, then fix the bulb in its holder, and adorn it with the lampshade.",
    ],
}

simple_task_descriptions = {
    "one_leg": [
        "Screw in the leg into the square tabletop",
        "Put together the one-legged piece",
        "Construct the one-leg item",
        "Build the one-legged structure",
        "Piece together the one-leg furniture",
        "Assemble the components of the one-leg piece",
        "Join the parts of the one-legged item",
        "Combine the pieces of the one-leg structure",
        "Set up the one-legged furniture",
        "Arrange the one-leg assembly",
    ],
    "lamp": [
        "Assemble the lamp",
        "Put together the lamp",
        "Construct the lamp",
        "Build the lamp",
        "Piece together the lamp",
        "Assemble the lamp components",
        "Join the parts of the lamp",
        "Combine the lamp pieces",
        "Set up the lamp",
        "Arrange the lamp assembly",
    ],
    "round_table": [
        "Assemble the round table",
        "Put together the round table",
        "Construct the round table",
        "Build the round table",
        "Piece together the round table",
        "Assemble the round table components",
        "Join the parts of the round table",
        "Combine the round table pieces",
        "Set up the round table",
        "Arrange the round table assembly",
    ],
    "desk": [
        "Assemble the desk",
        "Put together the desk",
        "Construct the desk",
        "Build the desk",
        "Piece together the desk",
        "Assemble the desk components",
        "Join the parts of the desk",
        "Combine the desk pieces",
        "Set up the desk",
        "Arrange the desk assembly",
    ],
    "square_table": [
        "Assemble the square table",
        "Put together the square table",
        "Construct the square table",
        "Build the square table",
        "Piece together the square table",
        "Assemble the square table components",
        "Join the parts of the square table",
        "Combine the square table pieces",
        "Set up the square table",
        "Arrange the square table assembly",
    ],
    "cabinet": [
        "Assemble the cabinet",
        "Put together the cabinet",
        "Construct the cabinet",
        "Build the cabinet",
        "Piece together the cabinet",
        "Assemble the cabinet components",
        "Join the parts of the cabinet",
        "Combine the cabinet pieces",
        "Set up the cabinet",
        "Arrange the cabinet assembly",
    ],
    "stool": [
        "Assemble the stool",
        "Put together the stool",
        "Construct the stool",
        "Build the stool",
        "Piece together the stool",
        "Assemble the stool components",
        "Join the parts of the stool",
        "Combine the stool pieces",
        "Set up the stool",
        "Arrange the stool assembly",
    ],
    "chair": [
        "Assemble the chair",
        "Put together the chair",
        "Construct the chair",
        "Build the chair",
        "Piece together the chair",
        "Assemble the chair components",
        "Join the parts of the chair",
        "Combine the chair pieces",
        "Set up the chair",
        "Arrange the chair assembly",
    ],
}
