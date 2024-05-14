from polymetis import GripperInterface
gripper = GripperInterface(ip_address="173.16.0.1")
gripper.goto(0.08, 0.05, 0.1, blocking=False)
# from IPython import embed; embed()
