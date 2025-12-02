#!/usr/bin/env python3
"""
Configuration validation script for Qwen pretraining.

This script validates training configuration against hardware constraints
and provides recommendations for optimal settings.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import TrainingConfig
from src.utils.config_utils import load_config_from_args, print_config_summary
from src.utils.hardware_validator import HardwareValidator


def main():
    """Main entry point for configuration validation."""
    try:
        print("=" * 60)
        print("QWEN PRETRAINING - CONFIGURATION VALIDATOR")
        print("=" * 60)
        
        # Load configuration
        config = load_config_from_args()
        
        # Print configuration
        print_config_summary(config)
        
        # Create hardware validator
        validator = HardwareValidator()
        
        # Run comprehensive validation
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        result = validator.run_pre_training_checks(config)
        
        # Print detailed report
        validator.print_validation_report(result)
        
        # Show system report
        print("\n" + "=" * 60)
        print("SYSTEM REPORT")
        print("=" * 60)
        
        system_report = validator.get_system_report()
        
        print(f"Platform: {system_report['system_info']['platform']}")
        print(f"Python: {system_report['system_info']['python_version']}")
        print(f"PyTorch: {system_report['system_info']['pytorch_version']}")
        print(f"CUDA: {system_report['system_info']['cuda_version']}")
        
        if system_report['system_info']['gpu_info']:
            for i, gpu in enumerate(system_report['system_info']['gpu_info']):
                print(f"GPU {i}: {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")
        
        print(f"RAM: {system_report['system_info']['ram_total_gb']:.1f}GB total, "
              f"{system_report['system_info']['ram_available_gb']:.1f}GB available")
        print(f"CPU: {system_report['system_info']['cpu_count']} cores")
        print(f"Disk: {system_report['system_info']['disk_free_gb']:.1f}GB free")
        
        # Show recommendations
        if system_report['recommendations']:
            print(f"\n💡 HARDWARE-SPECIFIC RECOMMENDATIONS:")
            for i, rec in enumerate(system_report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Show adjusted configuration if available
        if result.adjusted_config:
            print("\n" + "=" * 60)
            print("RECOMMENDED CONFIGURATION")
            print("=" * 60)
            print_config_summary(result.adjusted_config)
            
            # Save adjusted config
            adjusted_config_path = "config_adjusted.json"
            result.adjusted_config.save_to_file(adjusted_config_path)
            print(f"\n💾 Adjusted configuration saved to: {adjusted_config_path}")
            print(f"   Use: python train.py --config-file {adjusted_config_path}")
        
        # Final verdict
        print("\n" + "=" * 60)
        if result.is_valid:
            print("✅ CONFIGURATION IS VALID FOR TRAINING")
            if result.warnings:
                print("⚠️  Proceed with caution due to warnings above")
        else:
            print("❌ CONFIGURATION NEEDS ADJUSTMENT BEFORE TRAINING")
            if result.adjusted_config:
                print("🔧 Use the recommended configuration above")
        print("=" * 60)
        
        return 0 if result.is_valid else 1
        
    except Exception as e:
        print(f"\nValidation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())