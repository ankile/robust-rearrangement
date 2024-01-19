import pickle


def save_raw_rollout(
    robot_states, imgs1, imgs2, actions, rewards, success, furniture, output_path
):
    observations = list()

    for robot_state, image1, image2 in zip(robot_states, imgs1, imgs2):
        observations.append(
            {
                "robot_state": robot_state,
                "color_image1": image1,
                "color_image2": image2,
            }
        )

    data = {
        "observations": observations,
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "success": success,
        "furniture": furniture,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)
