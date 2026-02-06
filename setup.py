import os
from setuptools import setup, find_packages

# 从_version.py读取版本
version_file = os.path.join(os.path.dirname(__file__), 'ohtk', '_version.py')
with open(version_file) as f:
    exec(f.read())

# 读取依赖
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# 读取README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ohtk",
    version=__version__,  # 从_version.py读取
    author="LiuHengjiang",
    author_email="liuhengjiang@outlook.com",
    description="A framework for occupational health toolkits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/occupational_health_toolkits",  # 添加项目URL
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",   # 更精确的版本要求
    keywords="occupational health, data analysis, research",  # 添加关键词
    # entry_points={             # 命令行工具入口
    #     "console_scripts": [
    #         "my_command=my_project.module1:main",  # 将 my_project.module1 的 main 函数注册为命令行工具
    #     ],
    # },
)