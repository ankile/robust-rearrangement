furniture2idx = {
    "one_leg": 0,
    "lamp": 1,
    "round_table": 2,
    "desk": 3,
    "square_table": 4,
    "cabinet": 5,
    "stool": 6,
    "chair": 7,
}

idx2furniture = {v: k for k, v in furniture2idx.items()}

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
}
