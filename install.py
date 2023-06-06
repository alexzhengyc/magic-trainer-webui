import launch
import os
import pkg_resources
from packaging import version as packaging_version
import subprocess

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                operator = '=='
            elif '>=' in package:
                operator = '>='
            elif '<=' in package:
                operator = '<='
            else:
                continue  # Add more conditions for other comparison operators.

            package_name, package_version = package.split(operator)

            try:
                installed_version_str = pkg_resources.get_distribution(package_name).version
                installed_version = packaging_version.parse(installed_version_str)
            except pkg_resources.DistributionNotFound:
                launch.run_pip(f"install {package_name}=={package_version}", f"magic-trainer-webui requirement: {package_name}")
                continue

            package_version = packaging_version.parse(package_version)

            if operator == '==':
                if installed_version != package_version:
                    message = f"package not matched! changing version from {installed_version} to {package_version}"
                    launch.run_pip(f"install {package_name}=={package_version}", f"magic-trainer-webui requirement: {package_name} {message}")
            elif operator == '>=':
                if installed_version < package_version:
                    message = f"package is too old! changing version from {installed_version} to {package_version}"
                    launch.run_pip(f"install {package_name}=={package_version}", f"magic-trainer-webui requirement: {package_name} {message}")
            elif operator == '<=':
                if installed_version > package_version:
                    message = f"package is too new! changing version from {installed_version} to {package_version}"
                    launch.run_pip(f"install {package_name}=={package_version}", f"magic-trainer-webui requirement: {package_name} {message}")

                
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')


# exam if the build dir and dist dir exists
ext_dir = os.path.dirname(os.path.realpath(__file__))
# go to the parent dir of ext_dir
webui_ext_dir = os.path.dirname(ext_dir)
WEBUI_DIR = os.path.dirname(webui_ext_dir)

# def get_webui_path():
#     return WEBUI_DIR

if not os.path.exists(os.path.join(ext_dir, "kohya_ss_revised", "build")) or not os.path.exists(os.path.join(ext_dir, "kohya_ss_revised", "dist")):
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "kohya_ss_revised"))
    subprocess.run(["python", "setup.py", "install"])
    os.chdir(os.path.dirname(os.path.realpath(__file__)))