import copy

# ORDER OF THE NAMES HERE MATTERS!!!


one_leg_config = {
    "furniture": "square_table",
    "names": [
        "square_table_top",
        "square_table_leg1",
        "square_table_leg2",
        "square_table_leg3",
        "square_table_leg4",
    ],
    "usd_names": [
        "square_table_top_no_tag.usda",
        "square_table_leg1_no_tag.usda",
        "square_table_leg2_no_tag.usda",
        "square_table_leg3_no_tag.usda",
        "square_table_leg4_no_tag.usda",
    ],
    "obj_names": [
        "square_table_top.obj",
        "square_table_leg1.obj",
        "square_table_leg2.obj",
        "square_table_leg3.obj",
        "square_table_leg4.obj",
    ],
    "prim_paths": [
        f"/World/SquareTableTop",
        "/World/SquareTableLeg1",
        "/World/SquareTableLeg2",
        "/World/SquareTableLeg3",
        "/World/SquareTableLeg4",
    ],
}
square_table_config = copy.deepcopy(one_leg_config)


lamp_config = {
    "furniture": "lamp",
    "names": ["lamp_base", "lamp_bulb", "lamp_hood"],
    "usd_names": ["lamp_base.usda", "lamp_bulb.usda", "lamp_hood.usda"],
    "obj_names": ["lamp_base.obj", "lamp_bulb.obj", "lamp_hood.obj"],
    "prim_paths": [f"/World/LampBase", "/World/LampBulb", "/World/LampHood"],
}


round_table_config = {
    "furniture": "round_table",
    "names": ["round_table_top", "round_table_leg", "round_table_base"],
    "usd_names": [
        "round_table_top.usda",
        "round_table_leg.usda",
        "round_table_base.usda",
    ],
    "obj_names": [
        "round_table_top.obj",
        "round_table_leg.obj",
        "round_table_base.obj",
    ],
    "prim_paths": [
        f"/World/RoundTableTop",
        "/World/RoundTableLeg",
        "/World/RoundTableBase",
    ],
}


mug_rack_config = {
    "furniture": "mug_rack",
    "names": ["rack", "mug"],
    "usd_names": [
        "rack.usda",
        "mug.usda",
    ],
    "obj_names": [
        "mugrack/mugrack2.obj",
        "muggood/muggood.obj",
    ],
    "prim_paths": [
        "/World/Rack",
        "/World/Mug",
    ],
}


part_config_dict = {
    "one_leg": one_leg_config,
    "lamp": lamp_config,
    "round_table": round_table_config,
    "square_table": square_table_config,
    "mug_rack": mug_rack_config,
}
