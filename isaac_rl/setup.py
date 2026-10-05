"""Cài package `bipedal` — môi trường RL cho robot hai chân.

Bản cũ (source/transformer_nam/setup.py) đọc metadata từ config/extension.toml,
tức file đó là mẫu của IsaacLab extension template còn nguyên tiêu đề
"Extension Template". Project không dùng cơ chế Omniverse extension, nên
metadata viết thẳng vào đây và extension.toml đã xoá — bớt một file và bớt
phụ thuộc vào gói `toml` lúc build.

Cài (editable, code sửa là có hiệu lực ngay):

    pip uninstall -y transformer_nam    # nếu còn bản cài cũ
    pip install -e isaac_rl
"""

from setuptools import find_packages, setup

setup(
    name="bipedal",
    version="0.2.0",
    description="Môi trường RL IsaacLab cho robot hai chân (OFFICIALdesign, NewSimple)",
    packages=find_packages(include=["bipedal", "bipedal.*"]),
    install_requires=["psutil"],
    python_requires=">=3.10",
    zip_safe=False,
)
