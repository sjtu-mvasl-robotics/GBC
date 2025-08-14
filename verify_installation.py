#!/usr/bin/env python3

import sys
import importlib
import importlib.metadata
import traceback

def check_with_message(module_name, component_name, additional_check=None):
    """
    Check if a module can be imported and optionally perform additional checks.
    Returns a tuple of (success, message)
    """
    try:
        module = importlib.import_module(module_name)
        if additional_check:
            check_result, check_message = additional_check(module)
            if not check_result:
                return False, f"⚠️ {component_name}: {check_message}"
        return True, f"✅ {component_name}: OK"
    except ImportError as e:
        return False, f"❌ {component_name}: Failed to import {module_name}. Error: {str(e)}"
    except Exception as e:
        return False, f"❌ {component_name}: Unexpected error: {str(e)}"

def check_rsl_rl_version(module):
    """Check if rsl_rl has the dev tag in its version."""
    try:
        version = importlib.metadata.version('rsl_rl')
        if 'dev' not in version:
            return False, f"Installed version {version} does not have 'dev' tag. Please install the development version using 'pip install -e .'"
        return True, None
    except importlib.metadata.PackageNotFoundError:
        return False, "Package metadata not found"

def verify_gbc_core():
    """Verify GBC core components."""
    print("Checking GBC Core components...")
    
    checks = [
        ("GBC", "GBC base package"),
        ("GBC.utils.base.base_fk", "RobotKinematics module", 
         lambda m: (hasattr(m, "RobotKinematics"), "RobotKinematics class not found")),
        ("GBC.utils.data_preparation.create_smplh", "SMPLHFitter module",
         lambda m: (hasattr(m, "SMPLHFitter"), "SMPLHFitter class not found")),
        ("GBC.utils.data_preparation.pose_transformer_trainer", "PoseFormerTrainer module", 
         lambda m: (hasattr(m, "PoseFormerTrainer"), "PoseFormerTrainer class not found")),
        ("GBC.utils.data_preparation.amass_action_converter", "AMASSActionConverter module",
         lambda m: (hasattr(m, "AMASSActionConverter"), "AMASSActionConverter class not found")),
    ]
    
    success = True
    for module_name, component_name, *additional in checks:
        additional_check = additional[0] if additional else None
        check_success, message = check_with_message(module_name, component_name, additional_check)
        print(message)
        success = success and check_success
    
    return success

def verify_smpl_support():
    """Verify SMPL-related packages."""
    print("\nChecking SMPL support...")
    
    checks = [
        ("human_body_prior", "human_body_prior package"),
        ("body_visualizer", "body_visualizer package"),
    ]
    
    success = True
    for module_name, component_name in checks:
        check_success, message = check_with_message(module_name, component_name)
        print(message)
        success = success and check_success
    
    return success

def verify_isaac_lab():
    """Verify Isaac Lab installation."""
    print("\nChecking Isaac Lab installation...")
    
    try:
        from isaaclab.app import AppLauncher
        print("✅ Isaac Lab: OK")
        return True
    except ImportError:
        print("ℹ️ Isaac Lab: Not installed (optional)")
        return True
    except Exception as e:
        print(f"⚠️ Isaac Lab: Unexpected error: {str(e)}")
        return False

def verify_rsl_rl():
    """Verify rsl_rl installation."""
    print("\nChecking rsl_rl installation...")
    
    check_success, message = check_with_message("rsl_rl", "rsl_rl package", check_rsl_rl_version)
    print(message)
    
    return check_success

def main():
    """Run all verification checks."""
    print("=== GBC Installation Verification ===\n")
    
    gbc_ok = verify_gbc_core()
    smpl_ok = verify_smpl_support()
    isaac_ok = verify_isaac_lab()
    rsl_rl_ok = verify_rsl_rl()
    
    print("\n=== Verification Summary ===")
    print(f"GBC Core: {'OK' if gbc_ok else 'FAILED'}")
    print(f"SMPL Support: {'OK' if smpl_ok else 'FAILED'}")
    print(f"Isaac Lab: {'OK' if isaac_ok else 'NOT INSTALLED (optional)'}")
    print(f"rsl_rl: {'OK' if rsl_rl_ok else 'FAILED or INCORRECT VERSION'}")
    
    if gbc_ok and smpl_ok and isaac_ok and rsl_rl_ok:
        print("\n✅ All components verified successfully!")
        return 0
    else:
        print("\n⚠️ Some components failed verification. Check the messages above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
