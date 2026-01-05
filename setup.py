from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ohtk",  # 包名
    version="0.2.0",    # 版本号
    author="LiuHengjiang", # 作者
    author_email="liuhengjiang@outlook.com",  # 作者邮箱
    description="A framework for occupational health toolkits",  # 项目描述
    long_description=long_description,  # 长描述（通常从 README.md 读取）
    long_description_content_type="text/markdown",  # 长描述格式
    url="",  # 项目主页
    packages=find_packages(),  # 自动查找所有包
    install_requires=install_requires,
    classifiers=[              # 分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",   # Python 版本要求
    # entry_points={             # 命令行工具入口
    #     "console_scripts": [
    #         "my_command=my_project.module1:main",  # 将 my_project.module1 的 main 函数注册为命令行工具
    #     ],
    # },
)