from setuptools import setup, find_packages

setup(
    name="stroke_prognosis",          # 项目名称（自定义，需唯一）
    version="0.1",                    # 版本号
    author="Your Name",               # 作者信息（可选）
    description="Multimodal stroke prognosis prediction",  # 项目描述（可选）
    packages=find_packages(),         # 自动发现所有Python包（关键！）
    install_requires=[                # 项目依赖（可选，可自动读取requirements.txt）
        # "torch==2.0.1",
        # "pytorch-tabnet==4.9",
        # 建议直接通过requirements.txt管理依赖，此处可留空
    ],
    include_package_data=True,        # 包含非代码文件（如图像、配置文件等，可选）
)