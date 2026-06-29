"""Guard: the patched H1-2 handless URDF must remain urdf_parser_py-parseable.

scripts_phaseA/patch_h1_2_urdf.py produces the patched copy; this test asserts
the result loads via urdf_parser_py.URDF.from_xml_file and exposes the expected
27 revolute joints (legs 2*6 + waist 1 + arms 2*7). A regression here means the
trainer's FK init (CachedTargetLinkKinematics -> urdf_parser_py) will crash
before any training step.

Run:
  PYTHONPATH=/cpfs/user/yaoyifei/codespace/gbc/GBC:/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA \
    python GBC/tests/test_h1_2_urdf_parses.py
"""

from pathlib import Path

PATCHED_URDF = Path(
    "/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/assets_patched/unitree_h1_2/h1_2_handless.urdf"
)
EXPECTED_REVOLUTE_DOF = 27


def test_patched_h1_2_urdf_parses():
    from urdf_parser_py.urdf import URDF

    assert PATCHED_URDF.exists(), (
        f"patched URDF missing at {PATCHED_URDF}; run "
        "scripts_phaseA/patch_h1_2_urdf.py to produce it."
    )
    robot = URDF.from_xml_file(str(PATCHED_URDF))
    n_revolute = sum(1 for j in robot.joints if j.type == "revolute")
    assert n_revolute == EXPECTED_REVOLUTE_DOF, (
        f"expected {EXPECTED_REVOLUTE_DOF} revolute joints, found {n_revolute}; "
        "H1-2 handless URDF structure may have changed."
    )


if __name__ == "__main__":
    test_patched_h1_2_urdf_parses()
    print("OK: H1-2 patched URDF parses and has 27 revolute joints")
