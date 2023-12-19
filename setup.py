import sys

from setuptools import find_packages, setup


def parse_requirements(filename="requirements.txt"):
    # Read requirements.txt, ignore comments
    try:
        requirement_list = list()
        with open(filename, "rb") as f:
            lines = f.read().decode("utf-8").split("\n")
        for line in lines:
            line = line.strip()
            if "#" in line:
                # remove package starting with '#'
                line = line[: line.find("#")].strip()
            if line:
                if line.startswith("opencv-python"):
                    # in case of conda installed opencv, skip installing with pip
                    try:
                        import cv2

                        print(cv2.__version__)
                        continue
                    except Exception:
                        pass

                if line.startswith("torch"):
                    if sys.platform == "darwin":
                        line = "torch"

                if line.startswith("mmcv"):
                    if sys.platform == "darwin":
                        line = "mmcv"

                requirement_list.append(line)

    except Exception:
        print(f"'{filename}' not found!")
        requirement_list = list()
    return requirement_list


required_packages = parse_requirements()

setup(
    name="flare",
    version="0.1",
    description="Codebase for FLARE22 competition",
    author="MIA group",
    packages=find_packages(),
    install_requires=required_packages,
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
        "https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html",
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
    entry_points={"console_scripts": ["flare_test=flare.det.test:main"]},
)
