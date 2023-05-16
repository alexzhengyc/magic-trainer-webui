import launch
import os
import pkg_resources
import subprocess

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    launch.run_pip(f"install {package}", f"magic-trainer-webui requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"magic-trainer-webui requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')

# exam if the build dir and dist dir exists
ext_dir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(ext_dir, "kohya_ss", "build")) or not os.path.exists(os.path.join(ext_dir, "kohya_ss", "dist")):
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "kohya_ss"))
    subprocess.run(["python", "setup.py", "install"])
    os.chdir(os.path.dirname(os.path.realpath(__file__)))